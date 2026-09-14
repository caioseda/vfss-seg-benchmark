'''
DiffRect -- Latent Diffusion Label Rectification for Semi-supervised Medical Image Segmentation
(Liu, Li & Yuan, MICCAI 2024, arXiv:2407.09918).
Reference implementation: https://github.com/CUHK-AIM-Group/DiffRect (MIT), `train_diffrect_ACDC.py`.

DiffRect trains **two** networks. `self.model` is the ordinary segmentation U-Net that is all that
survives to inference. `self.rectifier` (see `diffrect_modules.RectificationNet`) is a label
rectifier: it compresses a pseudo-label into a latent, runs a small conditional latent diffusion in
that space, and decodes a *corrected* pseudo-label, which then supervises the segmentation network.
The rectifier is discarded at test time, so DiffRect costs nothing extra at inference.

The three losses of Eq. 13, in this file's names:

    L_Seg   = sup_loss + w * unsup_loss          the FixMatch-style baseline on the seg net
    L_Lat   = refine_loss                        the rectifier's own supervision + its latent MSE
    L_Rect  = rect_loss                          rectified pseudo-label -> seg net (after warm-up)

Two structural deviations from the reference, both deliberate:

**1. One optimizer with two param groups, one fused backward** (the reference takes three separate
optimizer steps per iteration). Not a stylistic choice -- Lightning's manual optimization would
break this experiment in two silent ways, both verified against the installed pytorch-lightning
2.6.1:

  - `loops/training_epoch_loop.py:114-117` makes `global_step` count *optimizer* steps rather than
    batches when `automatic_optimization` is False. With three steps per iteration, the config's
    `max_steps: 8000` would stop after ~2,667 batches, so DiffRect would see a third of the images
    the other X1 methods see -- and the third step is gated on the rectification warm-up, so the
    steps-to-batches ratio is not even constant.
  - `loops/training_epoch_loop.py:475` skips LR schedulers entirely under manual optimization,
    without warning. DiffRect would train at a flat LR while the other methods anneal.

The fusion is sound because the reference's three backwards are already gradient-disjoint: every
tensor crossing between the two networks (pseudo-labels in, rectified pseudo-label out) is detached
on both sides. `tests/test_ssl_methods.py` pins that disjointness, which is what makes the
equivalence checkable rather than merely asserted. The one real difference is *update ordering*:
the reference's rectification feedback sees a seg net already updated by the step before it.

**2. The rectification feedback reuses the weak-view logits** instead of re-running the segmentation
network. The reference re-forwards only because it had already freed the graph with its first
backward; with a fused backward there is nothing to recover.
'''

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import SemiSupervisedLitWrapper
from .diffrect_modules import RectificationNet
from ...data.augmentations import strong_augment, weak_augment_multi

from typing import Dict, Optional, Tuple

SEMI_SUPERVISED_BASES = ("reference", "canonical")


class DiffRectLitWrapper(SemiSupervisedLitWrapper):

    def __init__(
        self,
        *args,
        confidence_threshold: float = 0.8,
        semi_supervised_base: str = "reference",
        supervise_weak_view: bool = True,
        complementary_loss: bool = True,
        cutout_prob: float = 0.0,
        latent_channels: int = 256,
        rectifier_timesteps: int = 10,
        rectifier_sample_steps: int = 2,
        rectifier_beta_schedule: str = "cosine",
        rectifier_lr: Optional[float] = None,
        rectifier_weight_decay: Optional[float] = None,
        condition_on_image: bool = True,
        clip_denoised: bool = True,
        rectification_start_fraction: float = 1000 / 30000,
        rectification_start_steps: Optional[int] = None,
        rectification_weight: float = 1.0,
        refine_loss_weight: float = 1.0,
        image_channels: Optional[int] = None,
        **kwargs,
    ):
        '''
        Args:
            confidence_threshold: the reference's `--conf_thresh 0.8`, applied to a *min-max
                normalised* softmax rather than to the raw max probability -- see
                `normalized_pseudo_label`, which is not FixMatch's threshold and does not behave
                like it.
            semi_supervised_base: which semi-supervised objective the segmentation network is
                trained with underneath the rectification.

                - `"reference"`: what DiffRect actually published -- SSL4MIS-flavoured FixMatch with
                  the min-max pseudo-label and the entropy-weighted complementary loss. Required for
                  the ACDC numbers to reproduce.
                - `"canonical"`: this repository's `FixMatchLitWrapper` objective (max-softmax >=
                  tau, uncertain pixels dropped). Makes X1 a clean read on the *rectification module
                  alone*, since DiffRect and FixMatch then differ in exactly one thing.
            supervise_weak_view: take the supervised loss on the weakly-augmented view (reference)
                rather than the raw image (what the other three X1 methods do).
            cutout_prob: 0.0 by default because the reference's strong augmentation is photometric
                only (blur, contrast, sharpness, brightness) with no cutout.
            latent_channels / rectifier_*: see `RectificationNet`. Defaults reproduce the reference.
            rectifier_lr / rectifier_weight_decay: overrides for the rectifier's param group. None
                inherits from `optimizer_cfg`, which is what the reference does (both its optimizers
                are SGD 0.01 / 0.9 / 1e-4).
            rectification_start_fraction: fraction of the budget before the rectified pseudo-label
                starts supervising the segmentation network (the reference's `--refine_start 1000`
                out of 30000 iterations). Expressed as a fraction for the same reason the
                consistency schedule is -- see the note on `REFERENCE_TOTAL_STEPS` in `base.py`.
            rectification_weight / refine_loss_weight: scale on `L_Rect` and on the rectifier's own
                objective. 1.0 is the reference.
            image_channels: input channels of the rectifier. Defaults to the segmentation model's
                `n_channels`, which is 3 for VFSS and 1 for ACDC.
        '''
        super().__init__(*args, **kwargs)
        if semi_supervised_base not in SEMI_SUPERVISED_BASES:
            raise ValueError(
                f"semi_supervised_base must be one of {SEMI_SUPERVISED_BASES}; got {semi_supervised_base!r}."
            )
        self.save_hyperparameters({
            "confidence_threshold": confidence_threshold,
            "semi_supervised_base": semi_supervised_base,
            "supervise_weak_view": supervise_weak_view,
            "complementary_loss": complementary_loss,
            "cutout_prob": cutout_prob,
            "latent_channels": latent_channels,
            "rectifier_timesteps": rectifier_timesteps,
            "rectifier_sample_steps": rectifier_sample_steps,
            "rectifier_beta_schedule": rectifier_beta_schedule,
            "rectifier_lr": rectifier_lr,
            "rectifier_weight_decay": rectifier_weight_decay,
            "condition_on_image": condition_on_image,
            "clip_denoised": clip_denoised,
            "rectification_start_fraction": rectification_start_fraction,
            "rectification_start_steps": rectification_start_steps,
            "rectification_weight": rectification_weight,
            "refine_loss_weight": refine_loss_weight,
            "image_channels": image_channels,
        })
        self.confidence_threshold = confidence_threshold
        self.semi_supervised_base = semi_supervised_base
        self.supervise_weak_view = supervise_weak_view
        self.complementary_loss = complementary_loss
        self.cutout_prob = cutout_prob
        self.rectification_start_fraction = rectification_start_fraction
        self._rectification_start_override = rectification_start_steps
        self.rectification_weight = rectification_weight
        self.refine_loss_weight = refine_loss_weight
        self.rectifier_lr = rectifier_lr
        self.rectifier_weight_decay = rectifier_weight_decay

        if image_channels is None:
            image_channels = getattr(self.model, "n_channels", None)
            if image_channels is None:
                raise ValueError(
                    "image_channels could not be inferred: the segmentation model exposes no "
                    "`n_channels`. Pass `image_channels=` explicitly."
                )
        self.rectifier = RectificationNet(
            image_channels=image_channels,
            n_classes=self.n_classes,
            latent_channels=latent_channels,
            timesteps=rectifier_timesteps,
            sample_steps=rectifier_sample_steps,
            beta_schedule=rectifier_beta_schedule,
            condition_on_image=condition_on_image,
            clip_denoised=clip_denoised,
        )

    # ------------------------------------------------------------------ optimizer

    def optimizer_param_groups(self):
        '''
        Two param groups: the segmentation network and the rectifier.

        The rectifier is a *trained* second network, unlike Mean Teacher's EMA copy -- so unlike
        that one it must be in the optimizer. Overriding this is the whole reason
        `LitWrapper.optimizer_param_groups` exists: the base returns `self.model.parameters()` only,
        and a rectifier left out of it would be built, forwarded, checkpointed and never updated,
        while every loss curve still looked healthy.
        '''
        rectifier_group: Dict = {"params": list(self.rectifier.parameters())}
        if self.rectifier_lr is not None:
            rectifier_group["lr"] = self.rectifier_lr
        if self.rectifier_weight_decay is not None:
            rectifier_group["weight_decay"] = self.rectifier_weight_decay
        return [{"params": list(self.model.parameters())}, rectifier_group]

    @property
    def rectification_start(self) -> float:
        '''Step at which the rectified pseudo-label starts supervising the segmentation network.'''
        return self.budget_fraction_steps(
            self.rectification_start_fraction, self._rectification_start_override
        )

    # ------------------------------------------------------------------ reference loss pieces

    def normalized_pseudo_label(self, probs: Tensor) -> Tensor:
        '''
        DiffRect's pseudo-label rule, transcribed from `train_diffrect_ACDC.py:155-160,288-290`.

        Two properties that make this **not** interchangeable with FixMatch's threshold, and that
        are easy to lose in a "cleanup":

        1. The normalisation is `(p - min) / max` over the class axis -- *not* `(p - min) /
           (max - min)`. The reference's `normalize()` divides by `max_val`, so the transformed
           top class equals `1 - min/max` and the threshold `> 0.8` really asks "is the least
           likely class under a fifth of the most likely one".
        2. Pixels where **no** class clears the threshold are zeroed out and then `argmax`-ed, so
           they come back as class 0 -- *background* -- and are still trained on at full weight.
           FixMatch would drop them. Confidently-wrong background supervision on uncertain pixels is
           a load-bearing part of what this method does, not an oversight to fix.

        Args:
            probs: `[B, C, H, W]` softmax probabilities.

        Returns:
            `[B, H, W]` class indices.
        '''
        minimum = probs.min(dim=1, keepdim=True)[0]
        maximum = probs.max(dim=1, keepdim=True)[0]
        normalized = (probs - minimum) / maximum
        mask = (normalized > self.confidence_threshold).float()
        return torch.argmax((probs * mask).detach(), dim=1)

    @staticmethod
    def complementary_loss_term(strong_probs: Tensor, weak_probs: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        The "complementary loss" and its adaptive weight, transcribed from `get_comp_loss`
        (`train_diffrect_ACDC.py:124-153`). Negative learning: the class the *weak* view finds
        least likely is trained as the answer for `1 - strong_probs`.

        Transcribed rather than tidied, including two things that read like bugs and are kept
        because they are what produced the published numbers:

        - `Categorical(probs=...)` is built on a `[B, C, H*W]` reshape, and `Categorical` treats the
          **last** axis as the categories. So the entropy is over *pixels*, not classes, and the
          normaliser `log(H*W)` matches that. `as_weight` therefore measures how spatially peaked
          each class map is, not how confident the classifier is.
        - the returned `comp_loss` already contains `as_weight`, and the caller multiplies by
          `as_weight` a second time, so the term enters the objective as `as_weight**2 * CE(...)`.
          Reproduced in `unsupervised_loss` for the same reason.

        Returns `(comp_loss, as_weight)`.
        '''
        batch_size, n_classes = strong_probs.shape[:2]
        n_pixels = strong_probs.shape[2] * strong_probs.shape[3]
        flat = strong_probs.reshape(batch_size, n_classes, n_pixels)

        entropy = torch.distributions.Categorical(probs=flat).entropy()
        as_weight = torch.mean(1.0 - entropy / torch.log(torch.tensor(float(n_pixels))))

        comp_labels = torch.argmin(weak_probs.detach(), dim=1)
        comp_loss = as_weight * F.cross_entropy(1.0 - strong_probs, comp_labels)
        return comp_loss, as_weight

    def calibration_guidance(self, pseudo: Tensor, target: Tensor) -> Tensor:
        '''
        The paper's calibration guidance tau: how far the pseudo-label is from the better label.

        **This is the Dice *loss*, not the Dice *score*.** Eq. 6 of the paper reads
        `tau = Dice(y_s, y_w)`, but `train_diffrect_ACDC.py:337` computes `dice_loss(...)`, i.e.
        `1 - Dice`, so a large guidance means a *bad* pseudo-label. The direction matters because
        the guidance is fed in as the rectifier's timestep: a bad pseudo-label lands at the noisy
        end of the schedule, which is the behaviour that makes sense. The code is followed and the
        disagreement is pinned in `tests/test_reference_equivalence.py`.

        Returns a scalar broadcast to `[B]`, matching the reference, which computes one number from
        the labeled rows and applies it to the whole batch.
        '''
        guidance = self.dice_loss(pseudo.unsqueeze(1), target.unsqueeze(1), oh_input=True)
        return guidance.expand(pseudo.shape[0]) if guidance.dim() else guidance.repeat(pseudo.shape[0])

    def _cedice(self, logits: Tensor, target: Tensor) -> Tensor:
        '''`0.5 * (CE + Dice)`, the repository's composition, on hard integer targets.'''
        target = target.long()
        return 0.5 * (F.cross_entropy(logits, target) + self.dice_loss(logits, target, softmax=True))

    # ------------------------------------------------------------------ training

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        '''
        Unused by DiffRect, which overrides `training_step` outright: the rectifier needs both views
        and its own losses, none of which fit the single-forward contract this hook assumes. Kept so
        the class still satisfies `SemiSupervisedLitWrapper`'s interface, and raising rather than
        silently returning zero so a future refactor that routes through the base `training_step`
        fails loudly instead of quietly training the supervised baseline.
        '''
        raise NotImplementedError(
            "DiffRect overrides training_step; unsupervised_loss is not part of its path."
        )

    def compute_training_losses(self, batch: Dict) -> Dict[str, Tensor]:
        '''
        Every term of Eq. 13, separately, before they are summed.

        Split out of `training_step` so the fused backward is *testable*: the claim that one
        backward over the sum equals the reference's three separate backwards rests entirely on the
        terms being gradient-disjoint, and that can only be checked by backpropagating them one at a
        time. See `tests/test_ssl_methods.py::TestDiffRectGradientDisjointness`.

        Returns a dict with `sup_loss`, `unsup_loss`, `refine_loss`, `rect_loss`, the scalar
        `weight`, and the tensors the caller needs for logging and diagnostics.
        '''
        images, targets = batch["image"], batch["segmentation"]
        is_labeled = self.labeled_mask(batch).to(images.device)
        unlabeled = ~is_labeled

        all_rows = torch.arange(images.shape[0], device=images.device)
        hidden, has_hidden = self.hidden_targets(batch, all_rows)

        # One geometric draw for the image and every mask riding along with it: the supervised
        # target must stay aligned with the weak view it is scored against, and so must the hidden
        # diagnostic target, or the pseudo-label quality curves measure nothing.
        weak_images, (weak_targets, weak_hidden) = weak_augment_multi(images, [targets, hidden])
        strong_images, _ = strong_augment(weak_images, cutout_prob=self.cutout_prob)

        logits_weak = self.forward(weak_images)
        logits_strong = self.forward(strong_images)
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)

        # ---------------------------------------------------------------- L_Seg
        supervised_view = logits_weak if self.supervise_weak_view else self.forward(images)
        supervised_targets = weak_targets if self.supervise_weak_view else targets
        if is_labeled.any():
            sup_loss = self.supervised_loss(supervised_view[is_labeled], supervised_targets[is_labeled])
        else:
            sup_loss = logits_weak.sum() * 0.0

        pl_weak = self.normalized_pseudo_label(probs_weak).detach()
        pl_strong = self.normalized_pseudo_label(probs_strong).detach()

        weight = self.current_consistency_weight()
        unsup_loss = self._semi_supervised_loss(logits_strong, probs_strong, probs_weak, pl_weak, unlabeled)

        # ---------------------------------------------------------------- L_Lat (the rectifier)
        # Everything the rectifier reads from the segmentation network is already detached: the two
        # objectives share a batch, not a graph.
        colored_weak = self.rectifier.color(pl_weak)
        colored_strong = self.rectifier.color(pl_strong)
        # Unlabeled rows have no ground truth, so their "better" label is the weak pseudo-label --
        # for them the weak-to-ground-truth transportation degenerates to identity, exactly as in
        # the reference, which splices the two together before colouring.
        better_labels = torch.where(is_labeled[:, None, None], weak_targets.long(), pl_weak)
        colored_better = self.rectifier.color(better_labels)

        cg_w2g = self.calibration_guidance(pl_weak[is_labeled], weak_targets[is_labeled].long()) \
            if is_labeled.any() else torch.zeros(1, device=images.device)
        cg_w2g = cg_w2g[0].expand(images.shape[0])

        # (b1) weak pseudo-label -> ground truth
        ref_logits, latent_loss_w2g = self.rectifier.forward_train(
            weak_images, colored_weak, colored_better, cg_w2g
        )
        refine_sup = self._cedice(ref_logits[is_labeled], weak_targets[is_labeled]) \
            if is_labeled.any() else ref_logits.sum() * 0.0
        refine_sup = refine_sup + latent_loss_w2g

        ref_probs = torch.softmax(ref_logits, dim=1)
        pl_rectified_train = self.normalized_pseudo_label(ref_probs).detach()

        # (b2) strong pseudo-label -> weak pseudo-label
        cg_s2w = self.calibration_guidance(pl_strong[unlabeled], pl_rectified_train[unlabeled]) \
            if unlabeled.any() else torch.zeros(1, device=images.device)
        cg_s2w = cg_s2w[0].expand(images.shape[0])
        ref_logits_strong, latent_loss_s2w = self.rectifier.forward_train(
            strong_images, colored_strong, colored_weak, cg_s2w
        )
        refine_unsup = self._refine_unsupervised_loss(
            ref_logits_strong, ref_probs, pl_rectified_train, unlabeled
        ) + latent_loss_s2w

        refine_loss = refine_sup + weight * refine_unsup

        # ---------------------------------------------------------------- L_Rect
        rect_loss = logits_weak.sum() * 0.0
        rectifying = self.global_step >= self.rectification_start and unlabeled.any()
        pl_rectified = None
        if rectifying:
            with torch.no_grad():
                sampled_logits = self.rectifier.forward_sample(weak_images, colored_weak, cg_w2g)
                pl_rectified = self.normalized_pseudo_label(torch.softmax(sampled_logits, dim=1))
            rect_loss = self.rectification_weight * self._cedice(
                logits_weak[unlabeled], pl_rectified[unlabeled]
            )

        return {
            "logits_weak": logits_weak,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "refine_loss": refine_loss,
            "rect_loss": rect_loss,
            "weight": weight,
            "is_labeled": is_labeled,
            "unlabeled": unlabeled,
            "supervised_view": supervised_view,
            "supervised_targets": supervised_targets,
            "pl_weak": pl_weak,
            "pl_strong": pl_strong,
            "pl_rectified": pl_rectified,
            "weak_hidden": weak_hidden,
            "rectifying": rectifying,
            "refine_sup": refine_sup,
            "refine_unsup": refine_unsup,
            "latent_loss_w2g": latent_loss_w2g,
            "latent_loss_s2w": latent_loss_s2w,
            "cg_w2g": cg_w2g,
            "cg_s2w": cg_s2w,
        }

    def training_step(self, batch, batch_idx):
        parts = self.compute_training_losses(batch)
        sup_loss, unsup_loss = parts["sup_loss"], parts["unsup_loss"]
        refine_loss, rect_loss = parts["refine_loss"], parts["rect_loss"]
        weight = parts["weight"]
        is_labeled, unlabeled = parts["is_labeled"], parts["unlabeled"]
        latent_loss_w2g, latent_loss_s2w = parts["latent_loss_w2g"], parts["latent_loss_s2w"]
        refine_sup, refine_unsup = parts["refine_sup"], parts["refine_unsup"]
        cg_w2g, cg_s2w = parts["cg_w2g"], parts["cg_s2w"]
        pl_weak, pl_rectified, weak_hidden = parts["pl_weak"], parts["pl_rectified"], parts["weak_hidden"]
        device = sup_loss.device

        loss = sup_loss + weight * unsup_loss + self.refine_loss_weight * refine_loss + rect_loss

        # ---------------------------------------------------------------- logging
        extra = {
            "train/refine_loss": refine_loss.detach(),
            "train/refine_sup_loss": refine_sup.detach(),
            "train/refine_unsup_loss": refine_unsup.detach(),
            "train/latent_loss": (latent_loss_w2g + latent_loss_s2w).detach() / 2.0,
            "train/latent_loss_w2g": latent_loss_w2g.detach(),
            "train/latent_loss_s2w": latent_loss_s2w.detach(),
            "train/rect_loss": rect_loss.detach(),
            "train/rectification_active": torch.tensor(float(parts["rectifying"]), device=device),
            "train/calibration_guidance_w2g": cg_w2g[0].detach(),
            "train/calibration_guidance_s2w": cg_s2w[0].detach(),
        }
        self.log_training(loss, sup_loss, unsup_loss, weight, is_labeled,
                          parts["supervised_view"], parts["supervised_targets"], extra=extra)

        # The one diagnostic the reference does not have, and the reason to trust (or not) that the
        # rectifier is doing anything at all: the same pseudo-labels before and after rectification,
        # both scored against the ground truth the label regime hid. Rectified accuracy that does
        # not exceed pseudo accuracy means the second network is costing 3x the compute for nothing.
        unlabeled_hidden, unlabeled_has_hidden = self.hidden_targets(batch, unlabeled)
        if unlabeled_hidden is not None:
            warped_hidden = weak_hidden[unlabeled] if weak_hidden is not None else None
            self.log_unlabeled_diagnostics(pl_weak[unlabeled], warped_hidden, unlabeled_has_hidden,
                                           prefix="train/pseudo")
            if pl_rectified is not None:
                self.log_unlabeled_diagnostics(pl_rectified[unlabeled], warped_hidden,
                                               unlabeled_has_hidden, prefix="train/rectified")

        return loss

    # ------------------------------------------------------------------ semi-supervised variants

    def _semi_supervised_loss(self, logits_strong: Tensor, probs_strong: Tensor,
                              probs_weak: Tensor, pl_weak: Tensor, unlabeled: Tensor) -> Tensor:
        '''DiffRect's `L_Semi^Seg`, in whichever of the two flavours is configured.'''
        if not unlabeled.any():
            return logits_strong.sum() * 0.0

        if self.semi_supervised_base == "canonical":
            keep = probs_weak.max(dim=1)[0] >= self.confidence_threshold
            keep = keep & unlabeled[:, None, None]
            if not keep.any():
                return logits_strong.sum() * 0.0
            pixelwise = F.cross_entropy(logits_strong, pl_weak, reduction="none")
            return (pixelwise * keep).sum() / keep.sum()

        loss = self._cedice(logits_strong[unlabeled], pl_weak[unlabeled])
        if self.complementary_loss:
            # Computed over the *whole* batch, as in the reference -- the complementary term is the
            # one part of DiffRect's objective that also touches labeled rows.
            comp_loss, as_weight = self.complementary_loss_term(probs_strong, probs_weak)
            loss = loss + as_weight * comp_loss
        return loss

    def _refine_unsupervised_loss(self, ref_logits_strong: Tensor, ref_probs: Tensor,
                                  pl_rectified: Tensor, unlabeled: Tensor) -> Tensor:
        '''The rectifier's own strong-to-weak consistency term, mirroring `_semi_supervised_loss`.'''
        if not unlabeled.any():
            return ref_logits_strong.sum() * 0.0
        loss = self._cedice(ref_logits_strong[unlabeled], pl_rectified[unlabeled])
        if self.complementary_loss and self.semi_supervised_base == "reference":
            comp_loss, as_weight = self.complementary_loss_term(
                torch.softmax(ref_logits_strong, dim=1), ref_probs
            )
            loss = loss + as_weight * comp_loss
        return loss
