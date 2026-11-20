'''
Shared scaffolding for the semi-supervised baselines of experiment X1.

Two pools of frames reach training as "unlabeled" (see `src/data/vfss_frame_dataset.py`):

  1. annotated frames whose label the current regime hides (`label_fraction < 1.0`), which keep an
     all-zeros placeholder mask, and
  2. frames from the **unlabeled pool** -- video frames that were never annotated at all, which are
     the reason semi-supervision is worth trying on this dataset in the first place.

Supervising on the placeholder would teach the model "everything is background", so every subclass
here selects the supervised rows via `batch['metadata']['is_labeled']` -- never via mask content.

Loss composition follows the 2D semi-supervised baselines of HiLab-git/SSL4MIS
(https://github.com/HiLab-git/SSL4MIS, MIT, commit 06df6047a59aba9988ced8331998b5957ecb356b):

    supervised_loss = 0.5 * (cross_entropy + dice)
    loss            = supervised_loss + consistency_weight(step) * consistency_loss

so that the supervised baseline and the semi-supervised methods differ **only** in the consistency
term -- which is what makes the X1 comparison a clean read on supervision.
'''

import math
from collections import Counter

import torch
from torch import Tensor
from torchmetrics.functional.segmentation import dice_score

from ..base import LitWrapper
from ...data.augmentations import weak_augment_multi
from ...third_party.ssl4mis import DiceLoss

from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------------
# Why the schedule is expressed as a *fraction of the compute budget* and not in absolute steps.
#
# The SSL4MIS 2D scripts run for `max_iterations = 30000` and evaluate the ramp as
# `get_current_consistency_weight(iter_num // 150)` against `--consistency_rampup 200.0`.
# Those two constants are not independent: 150 * 200 == 30000. The divisor exists precisely so that
# the ramp reaches its maximum at the *last* iteration of training. Likewise the warm-up gate
# (`if iter_num < 1000`) covers the first 1/30 of the run.
#
# Copying `150` and `200` into a run with a different budget silently rescales the schedule: with
# max_steps=8000 the ramp only reaches 8000//150 = 53 of its 200 units, i.e. exp(-5*(1-53/200)^2)
# = 0.067 -- the consistency term would peak at 6.7% of its configured weight and the semi-supervised
# methods would be indistinguishable from the supervised baseline for reasons of arithmetic rather
# than of data. Anchoring both to the budget keeps the *shape* of the reference schedule under any
# `max_steps`.
# --------------------------------------------------------------------------------------------
REFERENCE_TOTAL_STEPS = 30_000
REFERENCE_RAMPUP_STEPS = 30_000    # 150 (divisor) * 200 (`--consistency_rampup`)
REFERENCE_WARMUP_STEPS = 1_000     # `if iter_num < 1000: consistency_loss = 0.0`

DEFAULT_RAMPUP_FRACTION = REFERENCE_RAMPUP_STEPS / REFERENCE_TOTAL_STEPS   # 1.0
DEFAULT_WARMUP_FRACTION = REFERENCE_WARMUP_STEPS / REFERENCE_TOTAL_STEPS   # 0.0333...

STUDENT_AUGMENT_MODES = ("none", "weak")


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    '''
    Exponential ramp-up `exp(-5 * (1 - t)^2)`, from Laine & Aila, "Temporal Ensembling for
    Semi-Supervised Learning" (arXiv:1610.02242), as used by every Mean Teacher descendant.

    Implemented from the paper rather than copied from SSL4MIS's `code/utils/ramps.py`: that file
    carries a CC BY-NC 4.0 header (Curious AI) even though the repository as a whole is MIT, and
    vendoring it would pull a NonCommercial clause into this repo.
    '''
    if rampup_length <= 0:
        return 1.0
    current = float(min(max(current, 0.0), rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


class SemiSupervisedLitWrapper(LitWrapper):
    '''
    Base LightningModule for the X1 methods. Subclasses implement `unsupervised_loss`.

    Validation and test are inherited untouched from `LitWrapper`: the label regime only ever hides
    labels in the `train` split (`label_regime_splits=('train',)`) and the unlabeled pool is only
    ever built from train-split videos, so val/test are always fully labelled and are evaluated
    identically for every method and every budget.
    '''

    # Whether the shared `training_step` (which is where `student_augment` is applied) is the one
    # this method trains with. Subclasses that override the step, or that already augment on their
    # own (FixMatch), set it False so that the flag cannot be silently ignored.
    supports_student_augment = True
    # Which rows `student_augment` reaches: `all` (the shared step, and Mean Teacher, whose teacher then
    # sees the same geometry) or `labeled` (FixMatch: its unlabeled rows already get their own
    # weak/strong views, and augmenting them here would stack two weak draws).
    student_augment_rows = "all"
    # Which rows go through the shared step's student forward (the one the supervised term uses):
    # `all` (every row -- the unsupervised term of Mean Teacher & co. reads the unlabeled logits from it)
    # or `labeled` (FixMatch, whose unsupervised term makes its own weak/strong forwards and never reads
    # the unlabeled logits of this one).
    #
    # `labeled` is not an optimisation, it is a correctness requirement whenever the two halves of the
    # batch are distinguishable (`student_augment_rows = "labeled"`): with 8 augmented labeled rows and
    # 8 RAW unlabeled rows that carry no loss in one train-mode BatchNorm batch, the network learns to
    # use the raw subgroup as a sink for the batch statistics, and in eval every raw INCA image falls in
    # that regime -- the E2 FixMatch predicted "all foreground" on INCA (val Dice ~0.05) while training
    # Dice was 0.90 (diagnosis 2026-11-17; pinned by tests/test_e2_aug.py).
    main_forward_rows = "all"

    def __init__(
        self,
        model_cfg,
        optimizer_cfg,
        lr_scheduler_cfg=None,
        consistency_weight: float = 0.1,
        consistency_rampup_fraction: float = DEFAULT_RAMPUP_FRACTION,
        consistency_warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
        consistency_rampup_steps: Optional[float] = None,
        consistency_warmup_steps: Optional[float] = None,
        total_steps: Optional[int] = None,
        log_pseudo_label_quality: bool = True,
        student_augment: str = "none",
        **lit_wrapper_kwargs,
    ):
        '''
        Args:
            consistency_weight: maximum weight of the unsupervised term (SSL4MIS `--consistency`).
            consistency_rampup_fraction: fraction of the training budget over which the weight ramps
                from ~0 to `consistency_weight`. 1.0 reproduces the reference schedule (full weight
                only at the last step); lower values turn the term on earlier.
            consistency_warmup_fraction: fraction of the budget during which the term is held at
                exactly 0, so the model is not asked to be consistent before it predicts anything.
            consistency_rampup_steps / consistency_warmup_steps: absolute overrides, in optimizer
                steps. When given they win over the fractions -- useful in tests, and for a run whose
                budget is not known ahead of time.
            total_steps: absolute override for the training budget. Normally left None and read from
                the trainer (`max_steps`, else `estimated_stepping_batches`).
            log_pseudo_label_quality: log the quality of the unsupervised signal against the ground
                truth that the label regime hid (never used by the loss -- see
                `log_unlabeled_diagnostics`). Requires `expose_hidden_targets=True` on the dataset.
            student_augment: `none` (default, the historical behaviour: no augmentation at all) or
                `weak` -- `weak_augment_multi` (flip + small affine) applied before the student
                forward to the rows picked by `student_augment_rows` (the whole batch, or only the
                labeled rows for FixMatch), with the masks warped by the same draw. Because the batch
                itself is replaced, every downstream consumer (the supervised term, and Mean
                Teacher's teacher forward on the unlabeled rows) sees the same geometry, so student
                and teacher outputs stay pixel-aligned with no inverse warp. The temporal MT also
                carries the draw into the neighbours, the optical flow and M_noc.
        '''
        if student_augment not in STUDENT_AUGMENT_MODES:
            raise ValueError(f"student_augment deve ser um de {STUDENT_AUGMENT_MODES}, recebeu '{student_augment}'")
        if student_augment != "none" and not self.supports_student_augment:
            raise ValueError(
                f"{type(self).__name__} nao aplica `student_augment` (o passo de treino e proprio ou o "
                f"metodo ja tem augmentation); use student_augment='none'."
            )
        super().__init__(model_cfg, optimizer_cfg, lr_scheduler_cfg, **lit_wrapper_kwargs)
        self.save_hyperparameters({
            "consistency_weight": consistency_weight,
            "consistency_rampup_fraction": consistency_rampup_fraction,
            "consistency_warmup_fraction": consistency_warmup_fraction,
            "consistency_rampup_steps": consistency_rampup_steps,
            "consistency_warmup_steps": consistency_warmup_steps,
            "total_steps": total_steps,
            "log_pseudo_label_quality": log_pseudo_label_quality,
            "student_augment": student_augment,
        })
        self.student_augment = student_augment
        self.consistency_weight = consistency_weight
        self.consistency_rampup_fraction = consistency_rampup_fraction
        self.consistency_warmup_fraction = consistency_warmup_fraction
        self._rampup_steps_override = consistency_rampup_steps
        self._warmup_steps_override = consistency_warmup_steps
        self._total_steps_override = total_steps
        self._total_steps_cache: Optional[int] = None
        self.log_pseudo_label_quality = log_pseudo_label_quality
        self.dice_loss = DiceLoss(self.n_classes)
        self.forward_counts: Counter = Counter()

    # ------------------------------------------------------------------ schedule

    @property
    def total_training_steps(self) -> int:
        '''
        The compute budget, in optimizer steps, that the consistency schedule is anchored to.

        Read once from the trainer and cached: `max_steps` when the run is step-budgeted (which is
        what the X1 configs do, so that every label budget sees the same number of images), otherwise
        Lightning's `estimated_stepping_batches`, which folds in `max_epochs`, the dataloader length
        and `accumulate_grad_batches`.
        '''
        if self._total_steps_override is not None:
            return int(self._total_steps_override)
        if self._total_steps_cache is not None:
            return self._total_steps_cache

        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            if getattr(trainer, "max_steps", -1) and trainer.max_steps > 0:
                self._total_steps_cache = int(trainer.max_steps)
                return self._total_steps_cache
            estimated = float(trainer.estimated_stepping_batches)
            if math.isfinite(estimated) and estimated > 0:
                self._total_steps_cache = int(estimated)
                return self._total_steps_cache

        raise RuntimeError(
            "The consistency schedule is defined as a fraction of the training budget, but the "
            "budget could not be determined: no trainer is attached and `total_steps` was not "
            "given. Set `trainer.params.max_steps` in the config, or pass `total_steps=` / the "
            "absolute `consistency_rampup_steps=` and `consistency_warmup_steps=` overrides."
        )

    def budget_fraction_steps(self, fraction: float, override: Optional[float] = None) -> float:
        '''
        Resolve a schedule point expressed as a fraction of the compute budget into absolute steps.

        The one idiom every schedule in this hierarchy uses -- consistency warm-up, consistency
        ramp-up, and DiffRect's rectification start. An absolute `override` always wins, which is
        what lets tests pin a schedule without a trainer. See the note on `REFERENCE_TOTAL_STEPS`
        for why fractions rather than the reference's absolute constants.
        '''
        if override is not None:
            return float(override)
        return fraction * self.total_training_steps

    @property
    def rampup_steps(self) -> float:
        return self.budget_fraction_steps(self.consistency_rampup_fraction, self._rampup_steps_override)

    @property
    def warmup_steps(self) -> float:
        return self.budget_fraction_steps(self.consistency_warmup_fraction, self._warmup_steps_override)

    def current_consistency_weight(self) -> float:
        '''Ramped weight of the unsupervised term at the current global step (0 during warm-up).'''
        if self.global_step < self.warmup_steps:
            return 0.0
        return self.consistency_weight * sigmoid_rampup(self.global_step, self.rampup_steps)

    # ------------------------------------------------------------------ batch plumbing

    @staticmethod
    def labeled_mask(batch: Dict) -> Tensor:
        '''Boolean `[B]` tensor flagging the rows whose ground truth is revealed in this regime.'''
        is_labeled = batch["metadata"]["is_labeled"]
        if not torch.is_tensor(is_labeled):
            is_labeled = torch.as_tensor(is_labeled)
        return is_labeled.bool()

    @staticmethod
    def hidden_targets(batch: Dict, rows: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        '''
        Ground truth of the selected rows *that the label regime hid*, for diagnostics only.

        Present only when the dataset was built with `expose_hidden_targets=True`, and only for rows
        that have an annotation at all -- frames from the unlabeled pool were never annotated, and
        their `has_hidden_target` flag is False.

        Returns `(targets [N,H,W], has_target [N] bool)`, or `(None, None)` when unavailable.
        '''
        if "hidden_segmentation" not in batch:
            return None, None
        targets = batch["hidden_segmentation"][rows]
        flags = batch["metadata"].get("has_hidden_target")
        if flags is None:
            has_target = torch.ones(targets.shape[0], dtype=torch.bool, device=targets.device)
        else:
            if not torch.is_tensor(flags):
                flags = torch.as_tensor(flags)
            has_target = flags.bool().to(targets.device)[rows]
        if not has_target.any():
            return None, None
        return targets, has_target

    # ------------------------------------------------------------------ losses

    def supervised_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        '''`0.5 * (CE + Dice)`, the composition used by the SSL4MIS 2D baselines.'''
        target = target.long()
        loss_ce = torch.nn.functional.cross_entropy(logits, target)
        loss_dice = self.dice_loss(logits, target, softmax=True)
        return 0.5 * (loss_ce + loss_dice)

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        '''
        The method-specific consistency/pseudo-label term.

        Args:
            batch: the raw batch (subclasses that need a second forward pass re-read `batch['image']`).
            logits: the shared step's student forward, already computed for the supervised term:
                `self.forward(batch['image'])` with `main_forward_rows = "all"`, or only the labeled
                rows (None if there are none) with `main_forward_rows = "labeled"`.
            is_labeled: `[B]` bool mask.
        '''
        raise NotImplementedError("Subclasses of SemiSupervisedLitWrapper must implement unsupervised_loss.")

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def log_unlabeled_diagnostics(
        self,
        predictions: Tensor,
        targets: Optional[Tensor],
        has_target: Optional[Tensor],
        keep: Optional[Tensor] = None,
        prefix: str = "train/pseudo",
    ) -> None:
        '''
        Quality of the signal the method is training the unlabeled rows on, measured against the
        ground truth the label regime hid.

        This is the single most informative diagnostic in semi-supervised training and it is
        **strictly observational**: it runs under `no_grad`, reads a tensor the loss never sees, and
        exists because the labels here are *masked*, not deleted. Coverage rising while accuracy
        falls is confirmation bias happening in real time -- the failure mode that makes a
        semi-supervised method quietly worse than its supervised baseline.

        Args:
            predictions: `[N,H,W]` class indices the unsupervised term is training towards, already
                aligned with `targets` (a method that augments geometrically must pass the targets
                through the same transform).
            targets / has_target: from `hidden_targets`. No-op when either is None.
            keep: optional `[N,H,W]` bool mask of the pixels that actually carry loss (FixMatch's
                confidence mask). Accuracy is reported over those pixels only.
        '''
        if not self.log_pseudo_label_quality or targets is None or has_target is None:
            return

        predictions = predictions[has_target]
        targets = targets[has_target].long()
        if predictions.numel() == 0:
            return

        correct = predictions == targets
        logs = {
            f"{prefix}_accuracy_all": correct.float().mean(),
            f"{prefix}_rows_with_gt": has_target.sum().float(),
        }

        if keep is not None:
            keep = keep[has_target]
            n_kept = keep.sum()
            logs[f"{prefix}_coverage_on_gt"] = keep.float().mean()
            logs[f"{prefix}_accuracy"] = ((correct & keep).sum() / n_kept) if n_kept > 0 else torch.zeros((), device=predictions.device)
        else:
            logs[f"{prefix}_accuracy"] = logs[f"{prefix}_accuracy_all"]

        per_class = dice_score(
            predictions, targets, num_classes=self.n_classes,
            include_background=False, average="none", input_format="index",
        )
        class_means = []
        for class_id, class_name in self.report_class_ids.items():
            value = per_class[:, class_id - 1].nanmean()
            logs[f"{prefix}_dice/{class_name}"] = value
            class_means.append(value)
        logs[f"{prefix}_dice"] = torch.stack(class_means).nanmean()

        self.log_dict(logs, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------ cost accounting

    def count_forward(self, network: str, n_images: int) -> None:
        '''
        Record that `network` ("student", "student_nograd", "teacher", ...) processed `n_images` in this
        training step. Read by `src.callbacks.ResourceMonitor` to report images per step per network,
        which is what explains the wall-clock differences between methods (the temporal variants run
        the teacher on 2-3x the unlabeled rows). Pure bookkeeping: never touches the loss.
        '''
        self.forward_counts[network] += int(n_images)

    def on_train_batch_start(self, batch, batch_idx):
        self.forward_counts = Counter()

    # ------------------------------------------------------------------ training

    def log_training(
        self,
        loss: Tensor,
        sup_loss: Tensor,
        unsup_loss: Tensor,
        weight: float,
        is_labeled: Tensor,
        logits: Tensor,
        targets: Tensor,
        extra: Optional[Dict[str, Tensor]] = None,
        logits_rows: str = "all",
    ) -> None:
        '''
        The training-time log block, shared by every subclass.

        Factored out of `training_step` so that a method which cannot use the shared step --
        `DiffRectLitWrapper`, which trains a second network -- still emits *exactly* the same keys.
        The X1/X1B notebooks read these keys off the CSV logger; a method that spelled them
        differently would silently drop out of the comparison plots.

        Args:
            extra: additional `{key: scalar}` logged with the same on_epoch reduction. Keys are
                used verbatim, so they must already carry the `train/` prefix.
            logits_rows: `all` -- `logits` is aligned with the whole batch (`is_labeled`); `labeled` --
                `logits` holds only the labeled rows, in order (or is None when there are none), as
                produced by `main_forward_rows = "labeled"`. `targets` is always the whole batch.
        '''
        log_dict = {
            "train/loss": loss,
            "train/sup_loss": sup_loss,
            "train/unsup_loss": unsup_loss,
            "train/unsup_loss_weighted": weight * unsup_loss,
            "train/consistency_weight": torch.tensor(weight, device=loss.device),
            "train/labeled_in_batch": is_labeled.sum().float(),
            "train/unlabeled_in_batch": (~is_labeled).sum().float(),
        }
        if extra:
            log_dict.update(extra)
        if logits_rows not in ("all", "labeled"):
            raise ValueError(f"logits_rows deve ser 'all' ou 'labeled', recebeu '{logits_rows}'")
        if is_labeled.any() and logits is not None:
            labeled_logits = logits if logits_rows == "labeled" else logits[is_labeled]
            for name, value in self.compute_metrics(labeled_logits, targets[is_labeled], stage="train").items():
                log_dict[f"train/{name}"] = value

        self.log("step", float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # The schedule is the one curve that has to be readable per step: it is what decides whether
        # the run is a semi-supervised run at all (see the note on REFERENCE_TOTAL_STEPS).
        self.log("train/consistency_weight_step", weight, prog_bar=False, logger=True,
                 on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def augment_batch(self, batch: Dict) -> Dict:
        '''
        Apply `student_augment` to a training batch. Returns a shallow copy whose per-row tensors
        (`image`, `segmentation`, `hidden_segmentation` and whatever `_weak_augment_rows` adds) are
        replaced, on the rows chosen by `student_augment_rows`, by one shared geometric draw per row;
        everything else (metadata, ...) is passed through untouched.
        '''
        if self.student_augment == "none":
            return batch
        images = batch["image"]
        if self.student_augment_rows == "labeled":
            rows = self.labeled_mask(batch).to(images.device)
            if not rows.any():
                return batch
        else:
            rows = None

        def take(tensor):
            return tensor if rows is None else tensor[rows]

        subset = {key: take(batch[key]) for key in self._augmented_keys(batch)}
        augmented_rows = self._weak_augment_rows(subset)

        augmented = dict(batch)
        for key, value in augmented_rows.items():
            value = value.to(batch[key].dtype)
            if rows is None:
                augmented[key] = value
            else:
                full = batch[key].clone()
                full[rows] = value
                augmented[key] = full
        return augmented

    def _augmented_keys(self, batch: Dict) -> List[str]:
        '''Per-row keys the geometry has to reach, so that nothing downstream sees a stale view.'''
        return [key for key in ("image", "segmentation", "hidden_segmentation") if key in batch]

    def _weak_augment_rows(self, rows: Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''One weak draw per row, applied to the image and to every mask of `rows`.'''
        mask_keys = [key for key in ("segmentation", "hidden_segmentation") if key in rows]
        image, masks = weak_augment_multi(rows["image"], [rows[key] for key in mask_keys])
        return {"image": image, **dict(zip(mask_keys, masks))}

    def training_step(self, batch, batch_idx):
        batch = self.augment_batch(batch)
        images, targets = batch["image"], batch["segmentation"]
        is_labeled = self.labeled_mask(batch).to(images.device)

        if self.main_forward_rows == "labeled":
            # `logits` covers ONLY the labeled rows (see `main_forward_rows`).
            if is_labeled.any():
                logits = self.forward(images[is_labeled])
                self.count_forward("student", logits.shape[0])
                sup_loss = self.supervised_loss(logits, targets[is_labeled])
            else:
                # No labeled row: no student forward at all here (forwarding the raw unlabeled rows would
                # bring the BN sink back). A zero tied to the parameters keeps the loss differentiable.
                logits = None
                sup_loss = sum(param.sum() for param in self.model.parameters()) * 0.0
        else:
            logits = self.forward(images)
            self.count_forward("student", images.shape[0])
            if is_labeled.any():
                sup_loss = self.supervised_loss(logits[is_labeled], targets[is_labeled])
            else:
                # Can only happen if a batch is drawn entirely from the unlabeled stream.
                sup_loss = logits.sum() * 0.0

        weight = self.current_consistency_weight()
        if weight > 0.0 and (~is_labeled).any():
            unsup_loss = self.unsupervised_loss(batch, logits, is_labeled)
        else:
            # No unlabeled rows in the batch, or still in warm-up: the method degenerates to its
            # supervised baseline for this step.
            unsup_loss = sup_loss * 0.0

        loss = sup_loss + weight * unsup_loss
        self.log_training(loss, sup_loss, unsup_loss, weight, is_labeled, logits, targets,
                          logits_rows=self.main_forward_rows)
        return loss
