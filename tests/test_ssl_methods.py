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
    FixMatchLitWrapper,
    MeanTeacherLitWrapper,
    SupervisedLitWrapper,
    sigmoid_rampup,
)
from tests.helpers import OPTIMIZER_CFG, TINY_MODEL_CFG, make_batch


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
        for wrapper_cls in (SupervisedLitWrapper, MeanTeacherLitWrapper, FixMatchLitWrapper):
            with self.subTest(wrapper=wrapper_cls.__name__):
                module = build(wrapper_cls)
                clean = make_batch(n_labeled=2, n_unlabeled=2)

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
