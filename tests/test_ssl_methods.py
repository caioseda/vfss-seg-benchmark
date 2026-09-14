'''
Correctness tests for the semi-supervised wrappers in `src/models/ssl/`.

These are not smoke tests. A smoke test proves the code runs; every bug that matters here runs
perfectly and produces a plausible-looking loss curve:

  - supervising on the all-zeros placeholder mask of a hidden-label frame (the model learns
    "background everywhere" and the whole X1 comparison becomes meaningless);
  - a consistency term that is detached and contributes no gradient, so the "semi-supervised"
    method is silently just the supervised baseline;
  - a teacher that receives gradients, or is registered with the optimizer, so it is not an EMA;
  - pseudo-labels taken along the wrong axis.

Each test below pins one such invariant.

Run with:  python -m unittest discover -s tests -v
'''

import math
import unittest
from unittest import mock

import torch

from src.models.ssl import (
    DiffRectLitWrapper,
    FixMatchLitWrapper,
    MeanTeacherLitWrapper,
    SupervisedLitWrapper,
    sigmoid_rampup,
)
from src.models.ssl.diffrect_modules import (
    distinct_colors,
    nearest_color_labels,
    semantic_coloring,
)
from tests.helpers import (
    DIFFRECT_KWARGS,
    DIFFRECT_MIN_SIZE,
    OPTIMIZER_CFG,
    TINY_MODEL_CFG,
    make_batch,
)


def build(wrapper_cls, **kwargs):
    torch.manual_seed(0)
    # The consistency schedule is a fraction of the training budget, which a real run reads from the
    # trainer. These tests create no trainer, so the budget is stated explicitly -- SSL4MIS's 30000
    # iterations, so the resolved schedule equals the reference's.
    kwargs.setdefault("total_steps", 30_000)
    return wrapper_cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                       report_class_ids={1: "C2", 3: "C4"}, **kwargs)


def at_step(module, step: int):
    '''Pin `global_step`, which is otherwise driven by a Trainer that these tests do not create.'''
    return mock.patch.object(type(module), "global_step",
                             new_callable=mock.PropertyMock, return_value=step)


class TestSupervisionIsolation(unittest.TestCase):
    '''The invariant the whole X1 experiment rests on: hidden labels must never be supervised.'''

    def test_supervised_loss_ignores_unlabeled_rows(self):
        # A frame whose label is hidden gets an all-zeros mask from the dataset. If the loss looks
        # at it, the model is trained to predict background there. Poison those rows with garbage:
        # a correct implementation cannot notice.
        for wrapper_cls in (SupervisedLitWrapper, MeanTeacherLitWrapper, FixMatchLitWrapper,
                            DiffRectLitWrapper):
            with self.subTest(wrapper=wrapper_cls.__name__):
                extra = DIFFRECT_KWARGS if wrapper_cls is DiffRectLitWrapper else {}
                module = build(wrapper_cls, **extra)
                clean = make_batch(n_labeled=2, n_unlabeled=2, size=DIFFRECT_MIN_SIZE)

                poisoned = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in clean.items()}
                poisoned["metadata"] = clean["metadata"]
                poisoned["segmentation"][2:] = torch.randint(0, 4, poisoned["segmentation"][2:].shape)

                with at_step(module, 0), torch.no_grad():
                    logits = module.forward(clean["image"])
                    mask = module.labeled_mask(clean)
                    loss_clean = module.supervised_loss(logits[mask], clean["segmentation"][mask])
                    loss_poisoned = module.supervised_loss(logits[mask], poisoned["segmentation"][mask])

                self.assertEqual(loss_clean.item(), loss_poisoned.item(),
                                 "supervised loss changed when hidden-label rows were corrupted")

    def test_training_step_loss_unchanged_by_unlabeled_targets(self):
        '''End-to-end version of the above, through the real `training_step`.'''
        module = build(SupervisedLitWrapper)
        clean = make_batch()
        poisoned = dict(clean)
        poisoned["segmentation"] = clean["segmentation"].clone()
        poisoned["segmentation"][2:] = 2

        module.log = lambda *a, **k: None
        module.log_dict = lambda *a, **k: None

        with at_step(module, 0):
            torch.manual_seed(1); loss_a = module.training_step(clean, 0)
            torch.manual_seed(1); loss_b = module.training_step(poisoned, 0)

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)


class TestConsistencyTermIsReal(unittest.TestCase):
    '''A consistency term that produces no gradient makes the method a supervised baseline in disguise.'''

    def test_unsupervised_loss_produces_gradients(self):
        for wrapper_cls in (MeanTeacherLitWrapper, FixMatchLitWrapper):
            with self.subTest(wrapper=wrapper_cls.__name__):
                # threshold 0.0 so FixMatch keeps every pixel even with an untrained net
                kwargs = {"confidence_threshold": 0.0, "cutout_prob": 0.0} if wrapper_cls is FixMatchLitWrapper else {}
                module = build(wrapper_cls, **kwargs)
                module.log = lambda *a, **k: None
                batch = make_batch()

                with at_step(module, 0):
                    logits = module.forward(batch["image"])
                    is_labeled = module.labeled_mask(batch)
                    unsup = module.unsupervised_loss(batch, logits, is_labeled)

                self.assertTrue(unsup.requires_grad, "unsupervised loss is detached from the graph")
                module.zero_grad()
                unsup.backward()
                total = sum(p.grad.abs().sum().item() for p in module.model.parameters() if p.grad is not None)
                self.assertGreater(total, 0.0, "unsupervised loss produced zero gradient on the student")

    def test_zero_weight_reduces_to_supervised(self):
        '''
        Differential test: with `consistency_weight=0` every method must reproduce the supervised
        baseline *exactly*. Any difference means the SSL machinery is perturbing the supervised path
        (extra RNG draws, BatchNorm updates from extra forward passes, ...) and the X1 comparison
        between methods would not be attributable to the consistency term alone.
        '''
        batch = make_batch()
        losses = {}
        for name, wrapper_cls in [("supervised", SupervisedLitWrapper),
                                  ("meanteacher", MeanTeacherLitWrapper),
                                  ("fixmatch", FixMatchLitWrapper)]:
            module = build(wrapper_cls, consistency_weight=0.0)
            module.log = lambda *a, **k: None
            module.log_dict = lambda *a, **k: None
            with at_step(module, 5000):   # past warm-up, so only the weight zeroes the term
                torch.manual_seed(7)
                losses[name] = module.training_step(batch, 0).item()

        self.assertAlmostEqual(losses["meanteacher"], losses["supervised"], places=6)
        self.assertAlmostEqual(losses["fixmatch"], losses["supervised"], places=6)


class TestMeanTeacher(unittest.TestCase):

    def test_ema_update_math(self):
        module = build(MeanTeacherLitWrapper, ema_decay=0.99)
        # Make student and teacher differ by a known amount.
        with torch.no_grad():
            for p in module.model.parameters():
                p.fill_(1.0)
            for p in module.teacher.parameters():
                p.fill_(0.0)

        step = 10_000  # large enough that alpha == ema_decay, not the 1-1/(t+1) warm-up rule
        with at_step(module, step):
            module.update_teacher()

        expected = 0.99 * 0.0 + 0.01 * 1.0
        for p in module.teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.full_like(p, expected), atol=1e-6))

    def test_ema_alpha_uses_true_average_early(self):
        '''`alpha = min(1 - 1/(step+1), decay)`: at step 0 the teacher must copy the student outright.'''
        module = build(MeanTeacherLitWrapper, ema_decay=0.99)
        with torch.no_grad():
            for p in module.model.parameters():
                p.fill_(1.0)
            for p in module.teacher.parameters():
                p.fill_(0.0)
        with at_step(module, 0):
            module.update_teacher()
        for p in module.teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.ones_like(p)), "teacher did not copy student at step 0")

    def test_teacher_gets_no_gradients(self):
        module = build(MeanTeacherLitWrapper)
        batch = make_batch()
        out = module.teacher(batch["image"]).sum()
        self.assertFalse(out.requires_grad, "teacher output is attached to the autograd graph")
        for p in module.teacher.parameters():
            self.assertFalse(p.requires_grad)

    def test_teacher_excluded_from_optimizer(self):
        '''If the teacher were in the optimizer it would be trained by SGD, not by EMA.'''
        module = build(MeanTeacherLitWrapper)
        optimizer = module.configure_optimizers()
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for p in module.teacher.parameters():
            self.assertNotIn(id(p), optimized)
        for p in module.model.parameters():
            self.assertIn(id(p), optimized)

    def test_evaluation_uses_teacher(self):
        module = build(MeanTeacherLitWrapper, evaluate_with_teacher=True)
        with torch.no_grad():
            for p in module.model.parameters():
                p.fill_(0.5)
            for p in module.teacher.parameters():
                p.fill_(-0.5)
        x = make_batch()["image"]
        self.assertTrue(torch.allclose(module._eval_forward(x), module.teacher(x)))
        self.assertFalse(torch.allclose(module._eval_forward(x), module.forward(x)))


class TestFixMatch(unittest.TestCase):

    def test_impossible_threshold_yields_zero_loss(self):
        '''Above-1.0 threshold keeps no pixel, so the term must be exactly 0 -- not NaN from 0/0.'''
        module = build(FixMatchLitWrapper, confidence_threshold=1.01)
        module.log = lambda *a, **k: None
        batch = make_batch()
        with at_step(module, 0):
            logits = module.forward(batch["image"])
            unsup = module.unsupervised_loss(batch, logits, module.labeled_mask(batch))
        self.assertEqual(unsup.item(), 0.0)
        self.assertFalse(torch.isnan(unsup).any())

    def test_pseudo_labels_taken_over_class_axis(self):
        '''
        Pins the axis of the argmax. A model that always predicts class 2 must produce pseudo-labels
        that are all 2; an argmax over a spatial axis would produce indices in range [0, H) instead.
        '''
        module = build(FixMatchLitWrapper, confidence_threshold=0.0, cutout_prob=0.0)
        module.log = lambda *a, **k: None

        forced_class = 2
        with torch.no_grad():
            module.model.net[-1].weight.zero_()
            module.model.net[-1].bias.zero_()
            module.model.net[-1].bias[forced_class] = 10.0

        batch = make_batch()
        captured = {}
        real_ce = torch.nn.functional.cross_entropy

        def spy(logits, target, **kwargs):
            captured["target"] = target.clone()
            return real_ce(logits, target, **kwargs)

        with at_step(module, 0), mock.patch("torch.nn.functional.cross_entropy", spy):
            logits = module.forward(batch["image"])
            module.unsupervised_loss(batch, logits, module.labeled_mask(batch))

        pseudo = captured["target"]
        self.assertEqual(pseudo.shape, (2, 16, 16))            # [B_unlabeled, H, W]
        self.assertTrue((pseudo == forced_class).all(), f"pseudo-labels were {pseudo.unique().tolist()}")

    def test_cutout_region_is_excluded_from_loss(self):
        '''
        A cutout hole replaces the image content with flat grey: whatever the model predicts there is
        not evidence about the real frame, so those pixels must be dropped from the pseudo-label loss.
        Keeping them trains the model to agree with itself about erased content.
        '''
        batch = make_batch()
        coverage = {}
        for cutout_prob in (0.0, 1.0):
            module = build(FixMatchLitWrapper, confidence_threshold=0.0, cutout_prob=cutout_prob)
            seen = {}
            module.log = lambda name, value, **k: seen.__setitem__(name, float(value))
            with at_step(module, 0):
                torch.manual_seed(5)
                logits = module.forward(batch["image"])
                module.unsupervised_loss(batch, logits, module.labeled_mask(batch))
            coverage[cutout_prob] = seen["train/pseudo_label_coverage"]

        self.assertEqual(coverage[0.0], 1.0, "threshold 0 without cutout should keep every pixel")
        self.assertLess(coverage[1.0], 1.0,
                        "cutout holes were not excluded from the pseudo-label loss")

    def test_pseudo_labels_come_from_the_weak_view(self):
        '''
        FixMatch's whole premise is asymmetry: the pseudo-label is read off the *weak* view and the
        *strong* view is trained against it. Taking the label from the strong view instead removes
        the asymmetry and turns the method into self-training on noise -- while still running and
        still producing a falling loss. Pinned by call count: the strong view is built exactly once.
        '''
        module = build(FixMatchLitWrapper, confidence_threshold=0.0, cutout_prob=0.0)
        module.log = lambda *a, **k: None
        batch = make_batch()

        import src.models.ssl.fixmatch as fixmatch_module
        calls = {"strong": 0, "weak": 0}
        real_strong, real_weak = fixmatch_module.strong_augment, fixmatch_module.weak_augment

        def counting_strong(*a, **k):
            calls["strong"] += 1
            return real_strong(*a, **k)

        def counting_weak(*a, **k):
            calls["weak"] += 1
            return real_weak(*a, **k)

        with at_step(module, 0), \
             mock.patch.object(fixmatch_module, "strong_augment", counting_strong), \
             mock.patch.object(fixmatch_module, "weak_augment", counting_weak):
            logits = module.forward(batch["image"])
            module.unsupervised_loss(batch, logits, module.labeled_mask(batch))

        self.assertEqual(calls["weak"], 1, "the weak view must be built exactly once")
        self.assertEqual(calls["strong"], 1,
                         "the strong view must be built exactly once -- a second call means the "
                         "pseudo-label was taken from a strongly augmented view")

    def test_confidence_mask_respects_threshold(self):
        '''With a near-uniform model, a high threshold must keep strictly fewer pixels than a low one.'''
        batch = make_batch()
        coverages = {}
        for threshold in (0.0, 0.9):
            module = build(FixMatchLitWrapper, confidence_threshold=threshold, cutout_prob=0.0)
            seen = {}
            module.log = lambda name, value, **k: seen.__setitem__(name, float(value))
            with at_step(module, 0):
                torch.manual_seed(3)
                logits = module.forward(batch["image"])
                module.unsupervised_loss(batch, logits, module.labeled_mask(batch))
            coverages[threshold] = seen.get("train/pseudo_label_coverage", 0.0)

        self.assertEqual(coverages[0.0], 1.0, "threshold 0 should keep every pixel")
        self.assertLess(coverages[0.9], coverages[0.0])


class TestDiffRect(unittest.TestCase):
    '''
    DiffRect trains a second network and fuses three of the reference's optimizer steps into one
    backward. Both of those are places where the method can be wrong while training perfectly.
    '''

    def build_diffrect(self, **kwargs):
        params = dict(DIFFRECT_KWARGS)
        params.update(kwargs)
        return build(DiffRectLitWrapper, **params)

    def batch(self, **kwargs):
        kwargs.setdefault("size", DIFFRECT_MIN_SIZE)
        return make_batch(**kwargs)

    # -------------------------------------------------------------- supervision isolation

    def test_training_step_never_supervises_hidden_labels(self):
        '''
        The X1 invariant, through DiffRect's own `training_step`.

        `TestSupervisionIsolation` exercises `supervised_loss` directly, which every method shares
        and none can get wrong. What it cannot see is the *row selection* in a `training_step` --
        and DiffRect is the only method that writes its own. Dropping the `[is_labeled]` mask there
        trains the model to predict background on every hidden-label frame, converges beautifully,
        and makes the whole X1 comparison meaningless.

        Both the supervised term and the rectifier's supervised branch are checked, because the
        rectifier reads the ground truth too (as the target of the weak-to-ground-truth
        transportation, and as the calibration guidance).
        '''
        clean = self.batch(n_labeled=2, n_unlabeled=2)
        poisoned = dict(clean)
        poisoned["segmentation"] = clean["segmentation"].clone()
        torch.manual_seed(99)
        poisoned["segmentation"][2:] = torch.randint(0, 4, poisoned["segmentation"][2:].shape)

        parts = {}
        for name, batch in (("clean", clean), ("poisoned", poisoned)):
            module = self.build_diffrect(rectification_start_steps=0)
            _silence(module)
            torch.manual_seed(4242)
            with at_step(module, 5_000):
                parts[name] = module.compute_training_losses(batch)

        for term in ("sup_loss", "refine_sup", "cg_w2g"):
            with self.subTest(term=term):
                self.assertTrue(
                    torch.allclose(parts["clean"][term], parts["poisoned"][term]),
                    f"{term} changed when the hidden-label rows were corrupted, so DiffRect is "
                    f"supervising on the all-zeros placeholder mask",
                )

    # -------------------------------------------------------------- the optimizer

    def test_rectifier_is_trained(self):
        '''
        The rectifier must be in the optimizer. This is the exact inverse of
        `TestMeanTeacher.test_teacher_is_not_in_the_optimizer`: both modules are extra `nn.Module`
        attributes on the wrapper, but one is an EMA copy and the other is trained by gradient
        descent, and `LitWrapper.optimizer_param_groups` defaults to excluding both.

        A rectifier left out would be built, forwarded, saved in the checkpoint and never updated,
        with every logged loss still falling.
        '''
        module = self.build_diffrect()
        groups = module.optimizer_param_groups()
        self.assertEqual(len(groups), 2, "expected one param group per trained network")

        rectifier_ids = {id(p) for p in module.rectifier.parameters()}
        seg_ids = {id(p) for p in module.model.parameters()}
        registered = {id(p) for group in groups for p in group["params"]}

        self.assertTrue(rectifier_ids <= registered, "rectifier parameters are not in the optimizer")
        self.assertTrue(seg_ids <= registered, "segmentation parameters are not in the optimizer")
        self.assertEqual({id(p) for p in groups[1]["params"]}, rectifier_ids,
                         "the second param group is not exactly the rectifier")
        self.assertFalse(seg_ids & {id(p) for p in groups[1]["params"]},
                         "segmentation parameters leaked into the rectifier's param group")

    def test_rectifier_lr_override_applies_to_its_group_only(self):
        module = self.build_diffrect(rectifier_lr=0.5, rectifier_weight_decay=0.25)
        groups = module.optimizer_param_groups()
        self.assertEqual(groups[1]["lr"], 0.5)
        self.assertEqual(groups[1]["weight_decay"], 0.25)
        self.assertNotIn("lr", groups[0], "the segmentation group must inherit from optimizer_cfg")

    # -------------------------------------------------------------- the fused backward

    def test_loss_terms_are_gradient_disjoint(self):
        '''
        The licence for fusing the reference's three optimizer steps into one backward.

        The reference takes three steps per iteration; we sum the terms and take one. That is only
        equivalent if no term reaches across the two networks -- which holds because every tensor
        crossing between them (pseudo-labels in, rectified pseudo-label out) is detached. If a
        detach is ever dropped, this test fails and the fusion stops being sound.

        Using manual optimization instead is not an option: under
        `automatic_optimization = False` Lightning counts `global_step` in optimizer steps rather
        than batches and silently skips LR schedulers (verified against pytorch-lightning 2.6.1),
        which would give DiffRect a third of the other methods' images and a flat LR.
        '''
        batch = self.batch(n_labeled=2, n_unlabeled=2)
        expectation = {
            "sup_loss": ("model", "rectifier"),
            "unsup_loss": ("model", "rectifier"),
            "rect_loss": ("model", "rectifier"),
            "refine_loss": ("rectifier", "model"),
        }
        for term, (owner, foreign) in expectation.items():
            with self.subTest(term=term):
                module = self.build_diffrect()
                _silence(module)
                # Past the rectification warm-up, so `rect_loss` is a live term.
                with at_step(module, 5_000):
                    parts = module.compute_training_losses(batch)
                module.zero_grad()
                self.assertTrue(parts[term].requires_grad, f"{term} carries no gradient at all")
                parts[term].backward()

                self.assertGreater(_grad_magnitude(getattr(module, owner)), 0.0,
                                   f"{term} produced no gradient for {owner}")
                self.assertEqual(_grad_magnitude(getattr(module, foreign)), 0.0,
                                 f"{term} leaked gradient into {foreign}; the fused backward is "
                                 f"no longer equivalent to the reference's separate steps")

    def test_latent_loss_trains_the_denoiser(self):
        '''
        The latent MSE must reach `LatentDenoiseUNet`. If the *prediction* were detached instead of
        the target, the LFR would be a no-op whose loss curve still falls (the encoder would learn
        to make the target easy), which is exactly the failure this method is prone to.
        '''
        module = self.build_diffrect()
        _silence(module)
        with at_step(module, 5_000):
            parts = module.compute_training_losses(self.batch())
        module.zero_grad()
        (parts["latent_loss_w2g"] + parts["latent_loss_s2w"]).backward()
        self.assertGreater(_grad_magnitude(module.rectifier.denoiser), 0.0,
                           "the latent loss does not train the denoising U-Net")

    def test_latent_target_carries_no_gradient(self):
        '''
        The clean latent is a *target*: the denoiser must be pulled towards it, never it towards the
        denoiser. If the target latent stayed in the graph, the encoder could minimise the latent
        MSE by making the target trivial -- a collapse that trains, converges, and logs a falling
        loss curve the whole way down.

        Checked on the tensor itself rather than through gradients, because the encode of the
        target runs under `no_grad` *and* is detached, so removing either one alone leaves the
        other in place and no gradient-based probe can tell.
        '''
        module = self.build_diffrect()
        rectifier = module.rectifier
        seen = []
        original_encode = rectifier.encode

        def spy(image, colored_mask, cg):
            features, embedding = original_encode(image, colored_mask, cg)
            seen.append(features[-1])
            return features, embedding

        rectifier.encode = spy
        image = torch.randn(2, 3, DIFFRECT_MIN_SIZE, DIFFRECT_MIN_SIZE)
        source = torch.randint(0, 4, (2, DIFFRECT_MIN_SIZE, DIFFRECT_MIN_SIZE))
        target = torch.randint(0, 4, (2, DIFFRECT_MIN_SIZE, DIFFRECT_MIN_SIZE))
        rectifier.forward_train(image, rectifier.color(source), rectifier.color(target), torch.rand(2))

        self.assertEqual(len(seen), 2, "expected one encode of the input and one of the target")
        condition_latent, target_latent = seen
        self.assertTrue(condition_latent.requires_grad,
                        "the condition latent must stay in the graph -- it is how the rectifier learns")
        self.assertFalse(target_latent.requires_grad,
                         "the target latent is in the graph; the latent loss can move the target")

    # -------------------------------------------------------------- the rectification gate

    def test_rectification_is_off_during_warmup_and_on_after(self):
        module = self.build_diffrect(rectification_start_steps=1_000)
        _silence(module)
        batch = self.batch()

        with at_step(module, 999):
            before = module.compute_training_losses(batch)
        self.assertEqual(float(before["rect_loss"]), 0.0,
                         "the rectified pseudo-label supervised the model before warm-up ended")
        self.assertFalse(before["rectifying"])
        self.assertIsNone(before["pl_rectified"])

        with at_step(module, 1_000):
            after = module.compute_training_losses(batch)
        self.assertTrue(after["rectifying"])
        self.assertNotEqual(float(after["rect_loss"]), 0.0)
        self.assertIsNotNone(after["pl_rectified"])

    def test_rectification_supervises_only_unlabeled_rows(self):
        '''
        `L_Rect` replaces the pseudo-label on rows that have none. Applying it to labeled rows too
        would train the model to match a *generated* mask on frames whose real mask is right there,
        drowning the supervised signal at exactly the budgets where it is scarcest.
        '''
        module = self.build_diffrect(rectification_start_steps=0)
        _silence(module)
        with at_step(module, 5_000):
            parts = module.compute_training_losses(self.batch(n_labeled=2, n_unlabeled=2))

        unlabeled = parts["unlabeled"]
        expected = module.rectification_weight * module._cedice(
            parts["logits_weak"][unlabeled], parts["pl_rectified"][unlabeled]
        )
        self.assertAlmostEqual(float(parts["rect_loss"]), float(expected), places=5,
                               msg="the rectification loss was not taken over the unlabeled rows alone")

    def test_rectification_start_scales_with_the_budget(self):
        '''
        Same reason the consistency schedule is a fraction: the reference's `--refine_start 1000` is
        1/30 of its 30000 iterations, and copying the absolute number into an 8000-step run would
        gate 1/8 of it instead.
        '''
        reference = self.build_diffrect(total_steps=30_000)
        self.assertAlmostEqual(reference.rectification_start, 1_000.0, places=6)
        short = self.build_diffrect(total_steps=8_000)
        self.assertAlmostEqual(short.rectification_start, 8_000 * (1_000 / 30_000), places=6)

    # -------------------------------------------------------------- LCC

    def test_semantic_coloring_round_trips(self):
        '''The colouring must be injective, or two classes become the same input to the rectifier.'''
        colors = distinct_colors(4)
        labels = torch.randint(0, 4, (3, 16, 16))
        recovered = nearest_color_labels(semantic_coloring(labels, colors), colors)
        self.assertTrue(torch.equal(recovered, labels))

    def test_semantic_coloring_matches_the_reference_palette(self):
        '''Black plus the primaries, as hardcoded in `train_diffrect_ACDC.py:230-236`.'''
        self.assertEqual(
            [[int(v) for v in c] for c in distinct_colors(4).tolist()],
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        )

    def test_coloring_shares_the_image_range(self):
        '''
        The coloured mask is concatenated onto the image, which this repository normalises to
        [-1, 1]. A [0, 1] mask would hand the encoder two differently-scaled halves.
        '''
        colored = semantic_coloring(torch.tensor([[[0, 1], [2, 3]]]), distinct_colors(4))
        self.assertAlmostEqual(float(colored.min()), -1.0, places=5)
        self.assertAlmostEqual(float(colored.max()), 1.0, places=5)

    # -------------------------------------------------------------- pseudo-labels

    def test_pseudo_label_falls_back_to_background_not_to_nothing(self):
        '''
        DiffRect's threshold is not FixMatch's. A pixel where no class clears it is zeroed and then
        argmax-ed, so it comes back as class 0 -- background -- and is still trained on. Reading
        this as "uncertain pixels are dropped" would be a different method.
        '''
        module = self.build_diffrect(confidence_threshold=0.8)
        # A near-uniform distribution: min/max ~ 1, so the normalised top class ~ 0 < 0.8.
        probs = torch.full((1, 4, 4, 4), 0.25)
        probs[:, 2] = 0.2501
        probs = probs / probs.sum(dim=1, keepdim=True)
        pseudo = module.normalized_pseudo_label(probs)
        self.assertTrue(torch.all(pseudo == 0),
                        "an all-uncertain map should collapse to background, not to its argmax")

    def test_pseudo_label_keeps_a_confident_pixel(self):
        module = self.build_diffrect(confidence_threshold=0.8)
        probs = torch.tensor([0.01, 0.02, 0.95, 0.02]).view(1, 4, 1, 1).expand(1, 4, 4, 4).contiguous()
        self.assertTrue(torch.all(module.normalized_pseudo_label(probs) == 2))

    def test_pseudo_labels_come_from_the_weak_view(self):
        '''
        The whole weak/strong hierarchy rests on the weak view being the better one. Taking the
        pseudo-label from the strong view instead inverts the direction the rectifier learns -- and
        since both views are the same shape, the swap runs and trains.

        The two runs below are seeded identically, so the weak view drawn inside
        `compute_training_losses` is bit-identical to the one reconstructed here.
        '''
        from src.data.augmentations import weak_augment_multi

        batch = self.batch()

        # A randomly initialised net is never confident, so at the real threshold *every* pixel of
        # *both* views collapses to background and the comparison below would pass no matter which
        # view it read. Threshold 0 keeps the raw argmax, which does differ between the views.
        module = self.build_diffrect(confidence_threshold=0.0)
        torch.manual_seed(1234)
        with torch.no_grad():
            weak_images, _ = weak_augment_multi(batch["image"], [batch["segmentation"]])
            weak_probs = torch.softmax(module.forward(weak_images), dim=1)
            expected = module.normalized_pseudo_label(weak_probs)

        module = self.build_diffrect(confidence_threshold=0.0)
        _silence(module)
        torch.manual_seed(1234)
        with at_step(module, 5_000):
            parts = module.compute_training_losses(batch)

        self.assertNotEqual(int(parts["pl_strong"].sub(parts["pl_weak"]).abs().sum()), 0,
                            "the two views produced identical pseudo-labels, so this test would "
                            "pass regardless of which one was read")
        self.assertTrue(torch.equal(parts["pl_weak"], expected),
                        "the pseudo-label did not come from the weakly-augmented view")

    # -------------------------------------------------------------- configuration

    def test_unknown_semi_supervised_base_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build_diffrect(semi_supervised_base="fixmatch")

    def test_canonical_base_drops_uncertain_pixels(self):
        '''The `canonical` flavour is the repository's FixMatch objective, threshold and all.'''
        module = self.build_diffrect(semi_supervised_base="canonical", confidence_threshold=1.01)
        _silence(module)
        with at_step(module, 5_000):
            parts = module.compute_training_losses(self.batch())
        self.assertEqual(float(parts["unsup_loss"]), 0.0,
                         "an unreachable threshold must leave no pixels to train on")

    def test_label_only_rectifier_ignores_the_image(self):
        '''
        `condition_on_image=False` is the paper's description of LFR; True (the default) is what the
        reference code does. The flag exists so the discrepancy is a one-line ablation.
        '''
        module = self.build_diffrect(condition_on_image=False)
        self.assertIsNone(module.rectifier.image_encoder)
        _silence(module)
        with at_step(module, 5_000):
            parts = module.compute_training_losses(self.batch())
        self.assertTrue(torch.isfinite(parts["refine_loss"]))

    def test_sampling_is_reproducible_under_a_seeded_generator(self):
        module = self.build_diffrect()
        rectifier = module.rectifier
        rectifier.eval()
        image = torch.randn(1, 3, DIFFRECT_MIN_SIZE, DIFFRECT_MIN_SIZE)
        labels = torch.randint(0, 4, (1, DIFFRECT_MIN_SIZE, DIFFRECT_MIN_SIZE))
        cg = torch.rand(1)
        with torch.no_grad():
            first = rectifier.forward_sample(image, rectifier.color(labels), cg,
                                             generator=torch.Generator().manual_seed(3))
            second = rectifier.forward_sample(image, rectifier.color(labels), cg,
                                              generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(first, second))


def _silence(module) -> None:
    '''Suppress Lightning logging, which needs a trainer these tests do not create.'''
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None


def _grad_magnitude(module) -> float:
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class TestConsistencySchedule(unittest.TestCase):

    def test_warmup_holds_weight_at_zero(self):
        module = build(SupervisedLitWrapper, consistency_weight=0.1, consistency_warmup_steps=1000)
        for step in (0, 500, 999):
            with at_step(module, step):
                self.assertEqual(module.current_consistency_weight(), 0.0)
        with at_step(module, 1000):
            self.assertGreater(module.current_consistency_weight(), 0.0)

    def test_warmup_and_rampup_scale_with_the_budget(self):
        '''
        The regression this class exists for.

        SSL4MIS hard-codes `iter_num // 150` against `--consistency_rampup 200.0` because its budget
        is 30000 steps (150 * 200). Carried literally into an 8000-step run, the ramp reaches only
        53/200 of its length and the consistency term peaks at ~0.7% of its configured weight -- the
        semi-supervised methods become the supervised baseline by arithmetic, and X1 measures
        nothing. The schedule must therefore be a fraction of whatever budget the run has.
        '''
        for budget in (8_000, 30_000):
            module = build(SupervisedLitWrapper, consistency_weight=0.1, total_steps=budget)
            with self.subTest(budget=budget):
                self.assertAlmostEqual(module.warmup_steps, budget / 30.0, places=6)
                with at_step(module, budget):
                    self.assertAlmostEqual(module.current_consistency_weight(), 0.1, places=6,
                                           msg="weight must reach its maximum at the end of training")
                with at_step(module, budget // 2):
                    # exp(-5 * 0.25) at half the budget, whatever the budget is.
                    self.assertAlmostEqual(module.current_consistency_weight(),
                                           0.1 * math.exp(-1.25), places=6)

    def test_budget_is_read_from_the_trainer(self):
        '''A real run states its budget in `trainer.params.max_steps`, not on the module.'''
        module = SupervisedLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                                      report_class_ids={1: "C2", 3: "C4"}, consistency_weight=0.1)
        with self.assertRaises(RuntimeError):
            _ = module.rampup_steps   # no trainer, no explicit budget: refuse to guess

        module._trainer = mock.Mock(max_steps=8_000)
        self.assertEqual(module.total_training_steps, 8_000)
        with at_step(module, 8_000):
            self.assertAlmostEqual(module.current_consistency_weight(), 0.1, places=6)

    def test_ramp_is_monotonic_and_capped(self):
        module = build(SupervisedLitWrapper, consistency_weight=0.1,
                       consistency_warmup_steps=0, consistency_rampup_steps=30_000)
        weights = []
        for step in range(0, 60_000, 1500):
            with at_step(module, step):
                weights.append(module.current_consistency_weight())
        self.assertEqual(weights, sorted(weights), "consistency weight is not monotonic")
        self.assertLessEqual(max(weights), 0.1 + 1e-9, "weight exceeded its configured maximum")
        self.assertAlmostEqual(max(weights), 0.1, places=6, msg="weight never reached its maximum")
        # A constant weight is also monotonic and also capped -- so the ramp must be shown to *ramp*,
        # otherwise dropping `sigmoid_rampup` entirely would go unnoticed.
        self.assertLess(weights[0], 0.5 * weights[-1],
                        f"weight does not ramp: starts at {weights[0]:.4f}, ends at {weights[-1]:.4f}")

    def test_sigmoid_rampup_matches_paper_formula(self):
        '''exp(-5(1-t)^2), from arXiv:1610.02242 -- reimplemented rather than copied (CC BY-NC).'''
        import math
        self.assertAlmostEqual(sigmoid_rampup(0, 200), math.exp(-5.0), places=9)
        self.assertAlmostEqual(sigmoid_rampup(100, 200), math.exp(-5.0 * 0.25), places=9)
        self.assertAlmostEqual(sigmoid_rampup(200, 200), 1.0, places=9)
        self.assertAlmostEqual(sigmoid_rampup(999, 200), 1.0, places=9, msg="ramp must clamp past its length")
        self.assertEqual(sigmoid_rampup(5, 0), 1.0, "zero-length ramp must be a no-op")


if __name__ == "__main__":
    unittest.main(verbosity=2)
