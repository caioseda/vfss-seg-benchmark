'''
Numerical equivalence against the reference implementation.

The unit tests in `test_ssl_methods.py` pin *our* invariants. These tests pin something else: that
the port is faithful -- that our loss produces the same number as HiLab-git/SSL4MIS
(commit 06df6047a59aba9988ced8331998b5957ecb356b, MIT) on identical inputs.

The reference snippets below are transcribed from that repository as literally as possible
(`code/utils/losses.py` and the training loop of `code/train_mean_teacher_2D.py`), keeping their
loop-based one-hot encoding and their exact arithmetic. If our adapted version ever drifts from
them, these tests fail with the numeric difference.

Run with:  python -m unittest discover -s tests -t . -v
'''

import unittest
from unittest import mock

import torch
import torch.nn as nn

from src.models.ssl import DiffRectLitWrapper, MeanTeacherLitWrapper, SupervisedLitWrapper
from src.third_party.diffrect import LatentDiffusion, cosine_beta_schedule
from src.third_party.ssl4mis import DiceLoss
from tests.helpers import DIFFRECT_KWARGS, OPTIMIZER_CFG, TINY_MODEL_CFG, make_batch


# --------------------------------------------------------------------------------------------
# Reference code, transcribed from SSL4MIS (code/utils/losses.py). Deliberately kept in its
# original shape -- loop-based one-hot, `.item()` bookkeeping and all -- so it is checkable
# against the upstream file line by line.
# --------------------------------------------------------------------------------------------
class ReferenceDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def reference_update_ema_variables(model, ema_model, alpha, global_step):
    '''From `train_mean_teacher_2D.py`, with the deprecated `add_(scalar, tensor)` call modernised.'''
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


N_CLASSES = 4


class TestDiceLossEquivalence(unittest.TestCase):
    '''Our adapted DiceLoss must equal the upstream one bit for bit.'''

    def test_matches_reference_on_random_inputs(self):
        ours, reference = DiceLoss(N_CLASSES), ReferenceDiceLoss(N_CLASSES)
        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            logits = torch.randn(3, N_CLASSES, 16, 16, generator=generator)
            target = torch.randint(0, N_CLASSES, (3, 16, 16), generator=generator)

            ours_value = ours(logits, target, softmax=True)
            # The reference indexes `target[:, i]`, so it expects the [B, 1, H, W] layout.
            reference_value = reference(logits, target.unsqueeze(1), softmax=True)

            self.assertAlmostEqual(ours_value.item(), reference_value.item(), places=6,
                                   msg=f"DiceLoss diverged from reference at seed {seed}")

    def test_matches_reference_on_degenerate_masks(self):
        '''All-background and single-class masks are where smoothing conventions usually differ.'''
        ours, reference = DiceLoss(N_CLASSES), ReferenceDiceLoss(N_CLASSES)
        logits = torch.randn(2, N_CLASSES, 8, 8)
        for name, target in [
            ("tudo fundo", torch.zeros(2, 8, 8, dtype=torch.long)),
            ("uma classe so", torch.full((2, 8, 8), 3, dtype=torch.long)),
        ]:
            with self.subTest(mask=name):
                self.assertAlmostEqual(
                    ours(logits, target, softmax=True).item(),
                    reference(logits, target.unsqueeze(1), softmax=True).item(),
                    places=6)


class TestSupervisedLossEquivalence(unittest.TestCase):
    '''`supervised_loss` must equal the reference's `0.5 * (loss_dice + loss_ce)`.'''

    def test_matches_reference_composition(self):
        module = SupervisedLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                      report_class_ids={1: "C2", 3: "C4"})
        reference_dice = ReferenceDiceLoss(N_CLASSES)
        ce_loss = torch.nn.CrossEntropyLoss()

        batch = make_batch(n_labeled=3, n_unlabeled=1)
        with torch.no_grad():
            logits = module.forward(batch["image"])
        labeled = module.labeled_mask(batch)
        logits_l, target_l = logits[labeled], batch["segmentation"][labeled]

        ours = module.supervised_loss(logits_l, target_l)

        # Exactly the reference's lines:
        #   loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
        #   loss_dice = dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
        #   supervised_loss = 0.5 * (loss_dice + loss_ce)
        reference_ce = ce_loss(logits_l, target_l.long())
        reference_dice_value = reference_dice(torch.softmax(logits_l, dim=1), target_l.unsqueeze(1))
        reference_value = 0.5 * (reference_dice_value + reference_ce)

        self.assertAlmostEqual(ours.item(), reference_value.item(), places=6)


class TestMeanTeacherEquivalence(unittest.TestCase):

    def test_ema_update_matches_reference(self):
        module = MeanTeacherLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                       report_class_ids={1: "C2", 3: "C4"}, ema_decay=0.99)

        # Two independent copies of the same starting point: ours and the reference's.
        reference_student = type(module.model)(**TINY_MODEL_CFG["params"])
        reference_teacher = type(module.model)(**TINY_MODEL_CFG["params"])
        reference_student.load_state_dict(module.model.state_dict())
        reference_teacher.load_state_dict(module.teacher.state_dict())

        for step in (0, 1, 7, 150, 10_000):
            with torch.no_grad():   # perturb the student so each step actually moves the teacher
                for p in module.model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
                reference_student.load_state_dict(module.model.state_dict())

            with mock.patch.object(type(module), "global_step",
                                   new_callable=mock.PropertyMock, return_value=step):
                module.update_teacher()
            reference_update_ema_variables(reference_student, reference_teacher, 0.99, step)

            for ours, reference in zip(module.teacher.parameters(), reference_teacher.parameters()):
                self.assertTrue(torch.allclose(ours, reference, atol=1e-7),
                                f"EMA diverged from reference at step {step}")

    def test_consistency_loss_matches_reference(self):
        '''Reference: `torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)`.'''
        module = MeanTeacherLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                       report_class_ids={1: "C2", 3: "C4"})
        batch = make_batch(n_labeled=2, n_unlabeled=2)
        is_labeled = module.labeled_mask(batch)

        torch.manual_seed(11)
        with torch.no_grad():
            logits = module.forward(batch["image"])
            ours = module.unsupervised_loss(batch, logits, is_labeled)

        # Same computation, written the reference's way, with the same noise draw.
        torch.manual_seed(11)
        with torch.no_grad():
            unlabeled = batch["image"][~is_labeled]
            noise = torch.clamp(torch.randn(unlabeled.shape) * 0.1, -0.2, 0.2)
            ema_output_soft = torch.softmax(module.teacher(unlabeled + noise), dim=1)
            outputs_soft = torch.softmax(logits, dim=1)
            reference = torch.mean((outputs_soft[~is_labeled] - ema_output_soft) ** 2)

        self.assertAlmostEqual(ours.item(), reference.item(), places=7)

    def test_teacher_sees_only_unlabeled_rows(self):
        '''
        The reference feeds the teacher `volume_batch[labeled_bs:]` -- unlabeled rows only. Feeding
        it the whole batch would still train, and still look reasonable, but would not be Mean
        Teacher as published.
        '''
        module = MeanTeacherLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                       report_class_ids={1: "C2", 3: "C4"})
        batch = make_batch(n_labeled=2, n_unlabeled=2)
        seen = {}
        original_teacher = module.teacher

        class SpyTeacher(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x):
                seen["batch_size"] = x.shape[0]
                return self.inner(x)
            def parameters(self, *a, **k): return self.inner.parameters(*a, **k)

        module.teacher = SpyTeacher(original_teacher)
        with torch.no_grad():
            logits = module.forward(batch["image"])
            module.unsupervised_loss(batch, logits, module.labeled_mask(batch))

        self.assertEqual(seen["batch_size"], 2,
                         "teacher was fed the whole batch, not just the unlabeled rows")


# ------------------------------------------------------------------------------------------------
# Reference code, transcribed from DiffRect (https://github.com/CUHK-AIM-Group/DiffRect, MIT):
# `train_diffrect_ACDC.py` for the training-loop pieces and
# `networks/guided_diffusion/gaussian_diffusion.py` for the diffusion schedule. Kept in their
# original shape -- including the parts that read like bugs -- so they stay checkable line by line
# against the upstream files.
# ------------------------------------------------------------------------------------------------

def reference_normalize(tensor):
    '''`train_diffrect_ACDC.py:155-160`. Note the divisor is `max_val`, not `max_val - min_val`.'''
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result


def reference_pseudo_label(outputs_soft, conf_thresh):
    '''`train_diffrect_ACDC.py:288-290`.'''
    pseudo_mask = (reference_normalize(outputs_soft) > conf_thresh).float()
    outputs_masked = outputs_soft * pseudo_mask
    return torch.argmax(outputs_masked.detach(), dim=1, keepdim=False)


def reference_get_comp_loss(weak, strong, bs, num_classes, patch_size):
    '''`train_diffrect_ACDC.py:124-153`, verbatim apart from the module-level names it closed over.'''
    import numpy as np
    from torch.distributions import Categorical

    ce_loss = nn.CrossEntropyLoss()
    il_output = torch.reshape(strong, (bs, num_classes, patch_size[0] * patch_size[1]))
    as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(patch_size[0] * patch_size[1]))
    as_weight = torch.mean(as_weight)
    comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
    comp_loss = as_weight * ce_loss(torch.add(torch.negative(strong), 1), comp_labels)
    return comp_loss, as_weight


def reference_betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    '''`guided_diffusion/gaussian_diffusion.py:57-62`.'''
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return betas


def reference_cosine_betas(num_diffusion_timesteps):
    '''`get_named_beta_schedule("cosine", T)`, `gaussian_diffusion.py:36-40`.'''
    import math
    return reference_betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


def reference_space_timesteps(num_timesteps, section_counts):
    '''`guided_diffusion/respace.py`, restricted to the single-section form DiffRect calls.'''
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class TestDiffRectMatchesReference(unittest.TestCase):
    '''
    Numeric parity for DiffRect's pieces. These matter more than usual because three of them look
    like defects and were kept deliberately: a future reader "fixing" any of them would silently
    change the method, and only these tests would notice.
    '''

    def build(self, **kwargs):
        torch.manual_seed(0)
        params = dict(DIFFRECT_KWARGS)
        params.update(kwargs)
        return DiffRectLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                  report_class_ids={1: "C2", 3: "C4"}, total_steps=30_000, **params)

    def test_pseudo_label_matches_reference(self):
        module = self.build(confidence_threshold=0.8)
        torch.manual_seed(7)
        probs = torch.softmax(torch.randn(3, N_CLASSES, 16, 16), dim=1)
        self.assertTrue(torch.equal(module.normalized_pseudo_label(probs),
                                    reference_pseudo_label(probs, 0.8)))

    def test_normalisation_divides_by_max_not_by_range(self):
        '''
        The reference's `normalize` computes `(x - min) / max`. That is almost certainly meant to be
        `(x - min) / (max - min)`, but it is not, and the threshold 0.8 is calibrated against what
        it actually computes. Pinned separately from the pseudo-label test so the intent is on the
        record rather than inferred from a passing equality.
        '''
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 4, 1, 1)
        module = self.build()
        minimum, maximum = probs.min(dim=1, keepdim=True)[0], probs.max(dim=1, keepdim=True)[0]
        by_max = (probs - minimum) / maximum
        by_range = (probs - minimum) / (maximum - minimum)
        self.assertFalse(torch.allclose(by_max, by_range))
        self.assertTrue(torch.allclose(reference_normalize(probs), by_max))
        # And the wrapper agrees with the reference at a threshold that separates the two.
        module.confidence_threshold = 0.8
        self.assertTrue(torch.equal(module.normalized_pseudo_label(probs),
                                    reference_pseudo_label(probs, 0.8)))

    def test_complementary_loss_matches_reference(self):
        torch.manual_seed(11)
        weak = torch.softmax(torch.randn(4, N_CLASSES, 8, 8), dim=1)
        strong = torch.softmax(torch.randn(4, N_CLASSES, 8, 8), dim=1)

        ours_loss, ours_weight = DiffRectLitWrapper.complementary_loss_term(strong, weak)
        ref_loss, ref_weight = reference_get_comp_loss(weak, strong, bs=4, num_classes=N_CLASSES,
                                                       patch_size=[8, 8])
        self.assertAlmostEqual(float(ours_weight), float(ref_weight), places=6)
        self.assertAlmostEqual(float(ours_loss), float(ref_loss), places=6)

    def test_complementary_weight_is_applied_twice(self):
        '''
        `get_comp_loss` returns a loss that already contains `as_weight`, and
        `train_diffrect_ACDC.py:304` multiplies by `as_weight` again. The term therefore enters the
        objective squared. Reproduced, and pinned, because "simplifying" it changes the objective.
        '''
        torch.manual_seed(13)
        weak = torch.softmax(torch.randn(2, N_CLASSES, 8, 8), dim=1)
        strong = torch.softmax(torch.randn(2, N_CLASSES, 8, 8), dim=1)

        module = self.build(semi_supervised_base="reference")
        comp_loss, as_weight = module.complementary_loss_term(strong, weak)
        unlabeled = torch.tensor([False, True])
        logits_strong = torch.randn(2, N_CLASSES, 8, 8)
        pl_weak = weak.argmax(dim=1)

        total = module._semi_supervised_loss(logits_strong, strong, weak, pl_weak, unlabeled)
        cedice = module._cedice(logits_strong[unlabeled], pl_weak[unlabeled])
        self.assertAlmostEqual(float(total - cedice), float(as_weight * comp_loss), places=5)

    def test_calibration_guidance_is_a_dice_loss_not_a_dice_score(self):
        '''
        Eq. 6 of the paper says `tau = Dice(y_s, y_w)`; `train_diffrect_ACDC.py:337` computes
        `dice_loss(...)`, i.e. `1 - Dice`. The code wins -- and the direction is load-bearing,
        because the guidance is fed in as the diffusion timestep, so a *worse* pseudo-label must
        map to a noisier step.
        '''
        module = self.build()
        labels = torch.randint(0, N_CLASSES, (2, 8, 8))

        identical = module.calibration_guidance(labels, labels)
        self.assertAlmostEqual(float(identical[0]), 0.0, places=5,
                               msg="a perfect pseudo-label must give guidance ~0, not ~1")

        torch.manual_seed(5)
        different = module.calibration_guidance(labels, torch.randint(0, N_CLASSES, (2, 8, 8)))
        self.assertGreater(float(different[0]), float(identical[0]))

    def test_calibration_guidance_is_broadcast_from_a_single_scalar(self):
        '''The reference computes one number over the labeled rows and applies it to the batch.'''
        module = self.build()
        labels = torch.randint(0, N_CLASSES, (3, 8, 8))
        guidance = module.calibration_guidance(labels, torch.randint(0, N_CLASSES, (3, 8, 8)))
        self.assertEqual(tuple(guidance.shape), (3,))
        self.assertTrue(torch.allclose(guidance, guidance[0].expand(3)))

    def test_cosine_beta_schedule_matches_guided_diffusion(self):
        for timesteps in (10, 50, 1000):
            with self.subTest(timesteps=timesteps):
                ours = cosine_beta_schedule(timesteps).tolist()
                theirs = reference_cosine_betas(timesteps)
                for a, b in zip(ours, theirs):
                    self.assertAlmostEqual(a, b, places=12)

    def test_respaced_timesteps_match_guided_diffusion(self):
        for timesteps, sample_steps in ((10, 2), (10, 10), (1000, 50)):
            with self.subTest(timesteps=timesteps, sample_steps=sample_steps):
                diffusion = LatentDiffusion(timesteps=timesteps, sample_steps=sample_steps)
                self.assertEqual(set(diffusion.sample_timesteps_index.tolist()),
                                 reference_space_timesteps(timesteps, [sample_steps]))

    def test_q_sample_matches_the_closed_form(self):
        '''`q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`, Eq. 1 of the paper.'''
        diffusion = LatentDiffusion(timesteps=10, sample_steps=2)
        torch.manual_seed(3)
        x_start = torch.randn(4, 8, 4, 4)
        noise = torch.randn_like(x_start)
        t = torch.tensor([0, 3, 6, 9])

        expected = torch.stack([
            diffusion.alphas_cumprod[step].sqrt() * x_start[i]
            + (1 - diffusion.alphas_cumprod[step]).sqrt() * noise[i]
            for i, step in enumerate(t.tolist())
        ])
        self.assertTrue(torch.allclose(diffusion.q_sample(x_start, t, noise=noise), expected, atol=1e-6))

    def test_sampling_clips_the_latent_like_the_reference(self):
        '''
        `ddim_sample_loop`'s `clip_denoised=True` default is never overridden by DiffRect, so the
        sampled *latent* is clamped to [-1, 1] -- a default meant for images, applied to a
        256-channel BatchNorm+LeakyReLU feature map. Kept because it is what produced the published
        numbers; pinned so removing it is a deliberate act.
        '''
        diffusion = LatentDiffusion(timesteps=10, sample_steps=2)
        constant = lambda x_t, t: torch.full_like(x_t, 5.0)
        clipped = diffusion.ddim_sample(constant, (1, 2, 4, 4), "cpu",
                                        generator=torch.Generator().manual_seed(0))
        unclipped = diffusion.ddim_sample(constant, (1, 2, 4, 4), "cpu",
                                          generator=torch.Generator().manual_seed(0),
                                          clip_denoised=False)
        self.assertAlmostEqual(float(clipped.abs().max()), 1.0, places=5)
        self.assertAlmostEqual(float(unclipped.abs().max()), 5.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------------------------
# The backbone. Every published 2-D ACDC number -- DiffRect's included -- is measured on SSL4MIS's
# U-Net, which is NOT `src/models/unet.py`. Reproducing a number while silently swapping a 1.81 M
# network for a 17.26 M one with no dropout is not a reproduction, so the vendored copy is pinned
# here against values transcribed from the upstream file.
# --------------------------------------------------------------------------------------------
class TestSSL4MISBackbone(unittest.TestCase):
    '''
    Transcribed from `code/networks/unet.py` (SSL4MIS, MIT, commit 06df6047...), whose `UNet` is
    built with `feature_chns=[16, 32, 64, 128, 256]` and `dropout=[0.05, 0.1, 0.2, 0.3, 0.5]`.
    '''

    # Measured on a faithful reconstruction of the upstream module at in_chns=1, class_num=4.
    REFERENCE_PARAMETER_COUNT = 1_813_764

    def test_parameter_count_matches_the_reference(self):
        from src.third_party.ssl4mis import SSL4MISUNet

        net = SSL4MISUNet(n_channels=1, n_classes=4)
        self.assertEqual(
            sum(p.numel() for p in net.parameters()), self.REFERENCE_PARAMETER_COUNT,
            "the vendored backbone drifted from SSL4MIS's U-Net; the ACDC numbers would no longer "
            "be measured on the architecture the published ones were",
        )

    def test_stage_widths_and_dropout_match_the_reference(self):
        from src.third_party.ssl4mis import SSL4MISUNet

        net = SSL4MISUNet(n_channels=1, n_classes=4)
        widths = [net.encoder.in_conv.conv_conv[0].out_channels]
        widths += [block.maxpool_conv[1].conv_conv[0].out_channels
                   for block in (net.encoder.down1, net.encoder.down2,
                                 net.encoder.down3, net.encoder.down4)]
        self.assertEqual(widths, [16, 32, 64, 128, 256])

        dropouts = [net.encoder.in_conv.conv_conv[3].p]
        dropouts += [block.maxpool_conv[1].conv_conv[3].p
                     for block in (net.encoder.down1, net.encoder.down2,
                                   net.encoder.down3, net.encoder.down4)]
        self.assertEqual(dropouts, [0.05, 0.1, 0.2, 0.3, 0.5])

    def test_decoder_upsamples_bilinearly_regardless_of_the_flag(self):
        '''
        Upstream reads `params['bilinear']` in `Decoder.__init__` and then never passes it to the
        up-blocks, so they always take `UpBlock`'s default of `True`. Transcribed as-is: honouring
        the flag instead would swap in transposed convolutions and move the model to 1.94 M
        parameters, i.e. a different network from the one the numbers come from.
        '''
        from src.third_party.ssl4mis import SSL4MISUNet

        net = SSL4MISUNet(n_channels=1, n_classes=4)
        for name in ("up1", "up2", "up3", "up4"):
            block = getattr(net.decoder, name)
            self.assertTrue(block.bilinear, f"decoder.{name} must upsample bilinearly")
            self.assertTrue(hasattr(block, "conv1x1"))

    def test_it_is_a_different_network_from_the_repo_unet(self):
        '''The gap that motivates running both on ACDC, stated so it cannot be forgotten.'''
        from src.models.unet import UNet
        from src.third_party.ssl4mis import SSL4MISUNet

        reference = sum(p.numel() for p in SSL4MISUNet(n_channels=1, n_classes=4).parameters())
        repo = sum(p.numel() for p in UNet(n_channels=1, n_classes=4, bilinear=True).parameters())
        self.assertGreater(repo / reference, 9.0,
                           "the repo U-Net is ~9.5x the reference's; if that ratio changed, the "
                           "ACDC backbone ablation needs rereading")

    def test_forward_contract(self):
        from src.third_party.ssl4mis import SSL4MISUNet

        net = SSL4MISUNet(n_channels=1, n_classes=4)
        # `LitWrapper` reads `.n_classes` off the model and feeds it `[B, C, H, W]` directly.
        self.assertEqual(net.n_classes, 4)
        self.assertEqual(net.n_channels, 1)
        self.assertEqual(tuple(net(torch.zeros(2, 1, 64, 64)).shape), (2, 4, 64, 64))
