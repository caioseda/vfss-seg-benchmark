'''
FixMatch (Sohn et al., arXiv:2001.07685) adapted to dense prediction.

**Deliberate divergence from SSL4MIS.** `code/train_fixmatch_standard_augs.py` in
HiLab-git/SSL4MIS is not vanilla FixMatch: it adds a "complementary loss" weighted by prediction
entropy (`get_comp_loss`), and its sibling `train_fixmatch_cta.py` uses CTAugment. X1 wants the
*simplest* semi-supervised baseline -- the floor any temporal contribution has to clear -- so this
implements canonical FixMatch instead:

    weak view  -> pseudo-label = argmax, kept where max softmax >= threshold
    strong view -> cross-entropy against that pseudo-label, on the kept pixels only

The strong view is built by applying **photometric-only** augmentation on top of the weak view
(see `src/data/augmentations.py`), so the pseudo-label stays pixel-aligned with the strong view and
no inverse warp is needed. Cutout holes are excluded from the loss: they carry no evidence.
'''

import torch
from torch import Tensor

from .base import SemiSupervisedLitWrapper
from ...data.augmentations import strong_augment, weak_augment

from typing import Dict


class FixMatchLitWrapper(SemiSupervisedLitWrapper):

    def __init__(self, *args, confidence_threshold: float = 0.95,
                 cutout_prob: float = 0.5, **kwargs):
        '''
        Args:
            confidence_threshold: minimum max-softmax for a pixel's pseudo-label to be trained on
                (FixMatch's tau; 0.95 in the paper).
            cutout_prob: probability of a cutout hole in the strong view.
        '''
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({
            "confidence_threshold": confidence_threshold,
            "cutout_prob": cutout_prob,
        })
        self.confidence_threshold = confidence_threshold
        self.cutout_prob = cutout_prob

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        unlabeled = ~is_labeled
        unlabeled_images = batch["image"][unlabeled]

        # The hidden ground truth (diagnostics only) rides along through `weak_augment`, which draws
        # its geometric parameters once and applies them to image and mask alike -- otherwise the
        # pseudo-label would be scored against an unwarped mask and the numbers would be meaningless.
        hidden, has_hidden = self.hidden_targets(batch, unlabeled)
        weak_images, hidden = weak_augment(unlabeled_images, hidden)

        with torch.no_grad():
            weak_probs = torch.softmax(self.forward(weak_images), dim=1)
            max_probs, pseudo_labels = weak_probs.max(dim=1)

        strong_images, valid = strong_augment(weak_images, cutout_prob=self.cutout_prob)
        strong_logits = self.forward(strong_images)

        keep = (max_probs >= self.confidence_threshold) & valid
        self.log("train/pseudo_label_coverage", keep.float().mean(),
                 on_step=False, on_epoch=True, logger=True)
        self.log_unlabeled_diagnostics(pseudo_labels, hidden, has_hidden, keep=keep,
                                       prefix="train/pseudo")

        if not keep.any():
            # Nothing confident yet -- returning a real zero (rather than skipping) keeps the graph
            # connected and the logged value meaningful.
            return strong_logits.sum() * 0.0

        pixelwise_ce = torch.nn.functional.cross_entropy(
            strong_logits, pseudo_labels, reduction="none"
        )
        return (pixelwise_ce * keep).sum() / keep.sum()
