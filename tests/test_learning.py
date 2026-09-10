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

from src.models.ssl import FixMatchLitWrapper, MeanTeacherLitWrapper, SupervisedLitWrapper
from tests.helpers import TINY_MODEL_CFG, make_batch

# Adam, not the SGD of `tests.helpers.OPTIMIZER_CFG`: with SGD(lr=0.1) this net needs ~1000 steps to
# memorise the batch, which would make the suite needlessly slow. The point is to prove the training
# loop optimises, not to benchmark the optimiser.
OVERFIT_OPTIMIZER_CFG = {"target": "torch.optim.Adam", "params": {"lr": 0.01}}
STEPS = 300


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
        with mock.patch.object(type(module), "global_step",
                               new_callable=mock.PropertyMock, return_value=step):
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

    def test_loss_is_finite_throughout(self):
        '''NaN/inf from the Dice smoothing or an empty confidence mask would poison training silently.'''
        batch = make_batch(n_labeled=2, n_unlabeled=2, size=16)
        for name, wrapper_cls in [("meanteacher", MeanTeacherLitWrapper),
                                  ("fixmatch", FixMatchLitWrapper)]:
            with self.subTest(method=name):
                torch.manual_seed(0)
                module = wrapper_cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OVERFIT_OPTIMIZER_CFG,
                                     report_class_ids={1: "C2", 3: "C4"},
                                     consistency_warmup_steps=0, consistency_rampup_steps=40)
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
