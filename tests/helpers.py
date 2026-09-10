'''Test fixtures: a tiny segmentation net and batch builders, so the SSL tests stay fast on CPU.'''

import torch
from torch import nn


class TinyNet(nn.Module):
    '''Minimal segmentation net (input size == output size) standing in for UNet in unit tests.'''

    def __init__(self, n_channels: int = 3, n_classes: int = 4, hidden: int = 8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_classes, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


TINY_MODEL_CFG = {
    "target": "tests.helpers.TinyNet",
    "params": {"n_channels": 3, "n_classes": 4, "hidden": 8},
}

OPTIMIZER_CFG = {"target": "torch.optim.SGD", "params": {"lr": 0.1}}


def make_batch(n_labeled: int = 2, n_unlabeled: int = 2, size: int = 16, seed: int = 0):
    '''
    Build a batch shaped like `VFSSWindowImageDataset` output, with the labeled rows first
    (the layout `TwoStreamBatchSampler` produces).

    Unlabeled rows carry an all-zeros `segmentation`, exactly as the dataset does for a frame whose
    label is hidden -- so any code that supervises on them is training on a lie.
    '''
    generator = torch.Generator().manual_seed(seed)
    batch_size = n_labeled + n_unlabeled

    images = torch.randn(batch_size, 3, size, size, generator=generator)
    targets = torch.zeros(batch_size, size, size, dtype=torch.long)
    # Give the labeled rows a non-trivial mask (classes 1 and 3, as in multiclass_c2_c4).
    targets[:n_labeled, 2:8, 2:8] = 1
    targets[:n_labeled, 9:14, 9:14] = 3

    is_labeled = torch.zeros(batch_size, dtype=torch.bool)
    is_labeled[:n_labeled] = True

    return {
        "image": images,
        "segmentation": targets,
        "metadata": {"is_labeled": is_labeled},
    }
