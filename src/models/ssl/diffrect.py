'''
DiffRect -- Latent Diffusion Label Rectification for Semi-supervised Medical Image Segmentation
(Bai et al., MICCAI 2024). Reference implementation: https://github.com/CUHK-AIM-Group/DiffRect (MIT).

**Phase 2 slot, not yet implemented.** It is registered here so the X1 config, the runner and the
notebook grid already carry a DiffRect cell; filling it in requires porting the Label Context
Calibration module and the latent diffusion rectification network from the reference repo.

That repo shares SSL4MIS's layout (`dataloaders/`, `networks/`, `utils/`, `val_2D.py`) because it is
built on that codebase -- which is why the rest of `src/models/ssl/` follows SSL4MIS conventions
(two-stream batches, `0.5 * (CE + Dice)` supervised loss, sigmoid-ramped consistency): the port
should drop in without reshaping the surrounding pipeline.
'''

from torch import Tensor

from .base import SemiSupervisedLitWrapper

from typing import Dict


class DiffRectLitWrapper(SemiSupervisedLitWrapper):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DiffRect is a phase-2 slot and is not implemented yet. Port the LCC + latent diffusion "
            "rectification modules from https://github.com/CUHK-AIM-Group/DiffRect (MIT), then "
            "implement `unsupervised_loss` here. Until then, run X1 with "
            "METHODS = ['supervised', 'fixmatch', 'meanteacher']."
        )

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        raise NotImplementedError
