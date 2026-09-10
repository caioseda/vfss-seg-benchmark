'''Supervised baseline at a given annotation budget -- the reference every X1 method must beat.'''

import torch
from torch import Tensor

from .base import SemiSupervisedLitWrapper

from typing import Dict


class SupervisedLitWrapper(SemiSupervisedLitWrapper):
    '''
    Trains on the revealed labels only, while still *seeing* the same batches as the
    semi-supervised methods (the unlabeled rows go through the forward pass, they just carry no
    loss). Keeping the batch composition identical is what isolates the effect of supervision:
    between this and, say, FixMatch, the only thing that changes is the consistency term.
    '''

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        return logits.sum() * 0.0
