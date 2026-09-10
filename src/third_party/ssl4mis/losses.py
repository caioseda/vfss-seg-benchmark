'''
Adapted from the official SSL4MIS implementation:
    https://github.com/HiLab-git/SSL4MIS  --  code/utils/losses.py
    commit 06df6047a59aba9988ced8331998b5957ecb356b
    License: MIT

`DiceLoss` is the loss used by every 2D semi-supervised baseline in that repository
(`train_mean_teacher_2D.py`, `train_fixmatch_standard_augs.py`, ...), always combined with
cross-entropy as `supervised_loss = 0.5 * (loss_dice + loss_ce)`.

Deviations from the original, all mechanical:
  - accepts targets in index format `[B, H, W]` (what `VFSSWindowImageDataset` yields) as well as
    the original `[B, 1, H, W]`;
  - one-hot encoding via `torch.nn.functional.one_hot` instead of a Python loop over classes;
  - no hard-coded `.cuda()` anywhere in this module, so it follows the module's device.
The loss arithmetic (squared denominators, `smooth = 1e-5`, unweighted mean over all classes
*including* background) is unchanged.
'''

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from typing import Optional, Sequence


class DiceLoss(nn.Module):
    '''Soft multiclass Dice loss, averaged over classes (background included).'''

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, target: Tensor) -> Tensor:
        '''Encode an index-format target `[B, H, W]` (or `[B, 1, H, W]`) as `[B, C, H, W]` floats.'''
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        return F.one_hot(target.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()

    @staticmethod
    def _dice_loss(score: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(
        self,
        inputs: Tensor,
        target: Tensor,
        weight: Optional[Sequence[float]] = None,
        softmax: bool = False,
    ) -> Tensor:
        '''
        Args:
            inputs: `[B, C, H, W]` -- logits when `softmax=True`, probabilities otherwise.
            target: `[B, H, W]` or `[B, 1, H, W]` class indices.
            weight: optional per-class weights (defaults to uniform).
            softmax: apply softmax over the class dimension of `inputs` first.
        '''
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        if inputs.size() != target.size():
            raise ValueError(f"predict & target shape do not match: {inputs.size()} vs {target.size()}")

        loss = 0.0
        for i in range(self.n_classes):
            loss = loss + self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes
