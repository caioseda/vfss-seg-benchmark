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

from src.models.ssl import MeanTeacherLitWrapper, SupervisedLitWrapper
from src.third_party.ssl4mis import DiceLoss
from tests.helpers import OPTIMIZER_CFG, TINY_MODEL_CFG, make_batch


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
