'''
Functional tests: the training machinery actually optimises.

Unit tests pin the algebra; equivalence tests pin fidelity to the reference. Neither would catch a
wrong loss sign, a mis-wired optimizer, or a learning rate that makes training a no-op -- all of
which produce a loss curve that merely looks unremarkable. The check for that is whether each
method can *overfit a handful of images*: if it cannot drive the loss down on 4 frames it will
certainly not learn anything on 690.

Slower than the other suites (a few seconds); still CPU-only.

Run with:  python -m unittest tests.test_learning -v
'''

import unittest
from unittest import mock

import torch

from src.models.ssl import (
    DiffRectLitWrapper,
    FixMatchLitWrapper,
    MeanTeacherLitWrapper,
    SupervisedLitWrapper,
)
from tests.helpers import DIFFRECT_KWARGS, DIFFRECT_MIN_SIZE, TINY_MODEL_CFG, make_batch

# DiffRect trains a second, much larger network per step and its rectifier needs >= 64px inputs,
# so it gets a shorter loop on a smaller batch. It is included here at all because its
# `training_step` returns a real loss -- which is only true because of the fused-backward design
# (manual optimization would return None and drop out of this suite entirely).
DIFFRECT_STEPS = 120

# Adam, not the SGD of `tests.helpers.OPTIMIZER_CFG`: with SGD(lr=0.1) this net needs ~1000 steps to
# memorise the batch, which would make the suite needlessly slow. The point is to prove the training
# loop optimises, not to benchmark the optimiser.
OVERFIT_OPTIMIZER_CFG = {"target": "torch.optim.Adam", "params": {"lr": 0.01}}
STEPS = 300


def at_step(module, step: int):
    '''Pin `global_step`, which is otherwise driven by a Trainer that these tests do not create.'''
    return mock.patch.object(type(module), "global_step",
                             new_callable=mock.PropertyMock, return_value=step)


def overfit(wrapper_cls, batch, steps: int = STEPS, **kwargs):
    '''Run a plain optimisation loop over one fixed batch; return (first_loss, last_loss, final_dice).'''
    torch.manual_seed(0)
    kwargs.setdefault("consistency_warmup_steps", 0)   # exercise the SSL path, not just warm-up
    kwargs.setdefault("consistency_rampup_steps", STEPS)  # no trainer here: state the budget outright
    module = wrapper_cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
                         report_class_ids={1: "C2", 3: "C4"}, **kwargs)
    module.log = lambda *a, **k: None
    module.log_dict = lambda *a, **k: None
    optimizer = module.configure_optimizers()

    losses = []
    for step in range(steps):
        with at_step(module, step):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()
            if hasattr(module, "update_teacher"):
                module.update_teacher()
        losses.append(loss.item())

    is_labeled = module.labeled_mask(batch)
    with torch.no_grad():
        forward = getattr(module, "_eval_forward", module.forward)
        logits = forward(batch["image"])
        dice = module.compute_metrics(logits[is_labeled], batch["segmentation"][is_labeled],
                                      stage="train")["dice_score"].item()
    return losses[0], losses[-1], dice


class TestCanOverfit(unittest.TestCase):
    '''Each method must be able to memorise a tiny labelled set.'''

    def test_all_methods_overfit_a_small_batch(self):
        batch = make_batch(n_labeled=3, n_unlabeled=1, size=16)
        for name, wrapper_cls in [("supervised", SupervisedLitWrapper),
                                  ("meanteacher", MeanTeacherLitWrapper),
                                  ("fixmatch", FixMatchLitWrapper)]:
            with self.subTest(method=name):
                first, last, dice = overfit(wrapper_cls, batch)
                self.assertLess(last, first * 0.2,
                                f"{name}: loss barely moved ({first:.4f} -> {last:.4f})")
                self.assertGreater(dice, 0.9,
                                   f"{name}: could not memorise 3 frames (Dice {dice:.3f})")

    def test_diffrect_leaves_the_segmentation_path_untouched(self):
        '''
        The invariant the fused-backward design rests on, stated as an *equality* rather than a
        threshold: with the unsupervised machinery off, DiffRect must train the segmentation network
        **bit-identically** to `SupervisedLitWrapper`.

        DiffRect adds a second network, four loss terms and a diffusion sampler to a single fused
        backward. If any of that leaked gradient into the segmentation network -- an undetached
        pseudo-label, a rectifier tensor still attached to the segmentation graph, the two param
        groups crossed -- the two runs would diverge. They do not, and that is checkable exactly,
        with no tolerance to tune.

        Why this replaced a "reaches at least 0.4x the supervised Dice" check: on this fixture that
        criterion measures the fixture, not the method. `TinyNet` is two convolutions (5x5 receptive
        field) and cannot memorise 64x64 inputs at any step budget this suite can afford, so the
        supervised baseline itself only reaches ~0.07 Dice. Against that floor, *every* consistency
        method degrades -- Mean Teacher reaches exactly 0.0 here, and FixMatch only ties the baseline
        because its 0.95 threshold keeps its unsupervised term at exactly zero throughout. A
        pseudo-label term trained against noise is *supposed* to hurt when the pseudo-labels are
        noise; that is not a defect to assert against.
        '''
        batch = make_batch(n_labeled=3, n_unlabeled=1, size=DIFFRECT_MIN_SIZE)
        common = dict(steps=DIFFRECT_STEPS, consistency_rampup_steps=DIFFRECT_STEPS,
                      consistency_weight=0.0)

        _, _, baseline_dice = overfit(SupervisedLitWrapper, batch, **common)
        _, _, dice = overfit(
            DiffRectLitWrapper, batch, total_steps=DIFFRECT_STEPS, supervise_weak_view=False,
            # Both halves of the unsupervised path off: consistency weight 0 above, and the
            # rectified pseudo-label never switched on. What is left is the supervised term alone.
            rectification_start_steps=DIFFRECT_STEPS + 1, **common, **DIFFRECT_KWARGS,
        )
        self.assertGreater(baseline_dice, 0.0, "the supervised baseline itself learned nothing; "
                                               "this fixture cannot support the comparison")
        self.assertAlmostEqual(
            dice, baseline_dice, places=6,
            msg=f"diffrect reached Dice {dice:.6f} where the same net trained supervised on the "
                f"same batch reached {baseline_dice:.6f}. With the consistency weight at 0 and the "
                f"rectification feedback off these must be identical -- a difference means the "
                f"rectifier or one of the extra loss terms is leaking gradient into the "
                f"segmentation network.",
        )

    def test_diffrect_rectifier_actually_trains(self):
        '''
        The other half: the rectifier is disjoint from the segmentation network, so *nothing in the
        segmentation metrics can tell us whether it trains at all*. A rectifier whose latent loss
        never moves would leave every test above green while making DiffRect an expensive FixMatch.

        Pinned on the latent MSE, which is the LFR's own objective (`L_Lat-U` / `L_Lat-L`).
        '''
        batch = make_batch(n_labeled=3, n_unlabeled=1, size=DIFFRECT_MIN_SIZE)
        torch.manual_seed(0)
        module = DiffRectLitWrapper(
            model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
            report_class_ids={1: "C2", 3: "C4"}, consistency_warmup_steps=0,
            consistency_rampup_steps=DIFFRECT_STEPS, total_steps=DIFFRECT_STEPS,
            **DIFFRECT_KWARGS,
        )
        latent_losses = []
        module.log = lambda *a, **k: None
        module.log_dict = lambda logs, *a, **k: latent_losses.append(
            float(logs["train/latent_loss"]))
        optimizer = module.configure_optimizers()
        for step in range(DIFFRECT_STEPS):
            with at_step(module, step):
                optimizer.zero_grad()
                loss = module.training_step(batch, 0)
                loss.backward()
                optimizer.step()

        self.assertTrue(latent_losses, "the module never logged train/latent_loss")
        first = sum(latent_losses[:10]) / len(latent_losses[:10])
        last = sum(latent_losses[-10:]) / len(latent_losses[-10:])
        self.assertLess(last, 0.9 * first,
                        f"the rectifier's latent loss barely moved ({first:.4f} -> {last:.4f}); "
                        f"the LFR is not learning and DiffRect degenerates to FixMatch with "
                        f"extra cost")

    def test_loss_is_finite_throughout(self):
        '''NaN/inf from the Dice smoothing or an empty confidence mask would poison training silently.'''
        for name, wrapper_cls in [("meanteacher", MeanTeacherLitWrapper),
                                  ("fixmatch", FixMatchLitWrapper),
                                  ("diffrect", DiffRectLitWrapper)]:
            with self.subTest(method=name):
                is_diffrect = wrapper_cls is DiffRectLitWrapper
                batch = make_batch(n_labeled=2, n_unlabeled=2,
                                   size=DIFFRECT_MIN_SIZE if is_diffrect else 16)
                torch.manual_seed(0)
                module = wrapper_cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
                                     report_class_ids={1: "C2", 3: "C4"},
                                     consistency_warmup_steps=0, consistency_rampup_steps=40,
                                     **({**DIFFRECT_KWARGS, "total_steps": 40} if is_diffrect else {}))
                module.log = lambda *a, **k: None
                module.log_dict = lambda *a, **k: None
                optimizer = module.configure_optimizers()
                for step in range(40):
                    with mock.patch.object(type(module), "global_step",
                                           new_callable=mock.PropertyMock, return_value=step):
                        optimizer.zero_grad()
                        loss = module.training_step(batch, 0)
                        self.assertTrue(torch.isfinite(loss), f"{name}: non-finite loss at step {step}")
                        loss.backward()
                        optimizer.step()


class TestTeacherTracksStudent(unittest.TestCase):
    '''The EMA teacher must lag the student, then close the gap as the student converges.'''

    def test_teacher_gap_shrinks_as_student_converges(self):
        '''
        The invariant is the *shape* of the trajectory, not any single value. The teacher-student gap
        legitimately peaks early -- that is what an EMA does while the student is moving fastest --
        and must then shrink as the student settles. A teacher that is diverging (wrong EMA sign,
        teacher being optimised, buffers not synced) shows the opposite: a gap that keeps growing.
        '''
        batch = make_batch(n_labeled=3, n_unlabeled=1, size=16)
        torch.manual_seed(0)
        module = MeanTeacherLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
                                       report_class_ids={1: "C2", 3: "C4"}, ema_decay=0.9,
                                       consistency_warmup_steps=0, consistency_rampup_steps=STEPS)
        module.log = lambda *a, **k: None
        module.log_dict = lambda *a, **k: None
        optimizer = module.configure_optimizers()

        def gap():
            return sum((t - s).abs().sum().item()
                       for t, s in zip(module.teacher.parameters(), module.model.parameters()))

        gaps = []
        for step in range(STEPS):
            with mock.patch.object(type(module), "global_step",
                                   new_callable=mock.PropertyMock, return_value=step):
                optimizer.zero_grad()
                module.training_step(batch, 0).backward()
                optimizer.step()
                module.update_teacher()
            gaps.append(gap())

        peak = max(gaps)
        final = gaps[-1]
        self.assertGreater(peak, 0.0, "teacher never lagged the student -- is the EMA a no-op?")
        self.assertLess(final, 0.3 * peak,
                        f"teacher is not closing the gap (peak {peak:.2f}, final {final:.2f})")

    def test_teacher_starts_synced_and_then_lags(self):
        '''At step 0 the alpha rule copies the student outright; afterwards the teacher must lag.'''
        batch = make_batch(n_labeled=3, n_unlabeled=1, size=16)
        torch.manual_seed(0)
        module = MeanTeacherLitWrapper(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
                                       report_class_ids={1: "C2", 3: "C4"}, ema_decay=0.9,
                                       consistency_warmup_steps=0, consistency_rampup_steps=STEPS)
        module.log = lambda *a, **k: None
        module.log_dict = lambda *a, **k: None
        optimizer = module.configure_optimizers()

        def gap():
            return sum((t - s).abs().sum().item()
                       for t, s in zip(module.teacher.parameters(), module.model.parameters()))

        with mock.patch.object(type(module), "global_step",
                               new_callable=mock.PropertyMock, return_value=0):
            optimizer.zero_grad()
            module.training_step(batch, 0).backward()
            optimizer.step()
            module.update_teacher()
        self.assertAlmostEqual(gap(), 0.0, places=4, msg="teacher did not copy student at step 0")

        for step in range(1, 20):
            with mock.patch.object(type(module), "global_step",
                                   new_callable=mock.PropertyMock, return_value=step):
                optimizer.zero_grad()
                module.training_step(batch, 0).backward()
                optimizer.step()
                module.update_teacher()
        self.assertGreater(gap(), 0.0, "teacher tracks the student exactly -- EMA is not averaging")


if __name__ == "__main__":
    unittest.main(verbosity=2)
