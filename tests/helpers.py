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


class TinyBNNet(nn.Module):
    '''`TinyNet` with a BatchNorm, for the tests that are about BN running statistics.'''

    def __init__(self, n_channels: int = 3, n_classes: int = 2, hidden: int = 8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, n_classes, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class TinyFeatureNet(nn.Module):
    '''
    `TinyBNNet` com o contrato `forward_features` do `SMPModel` (logits, local, global), para o DTC-TRUS.
    O mapa local fica a 1/2 da resolução (como a DeepLabV3+ fica a 1/4) e o global a 1/4, para exercitar a
    reamostragem de máscaras e fluxo.
    '''

    def __init__(self, n_channels: int = 3, n_classes: int = 2, hidden: int = 8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder = nn.Sequential(nn.Conv2d(n_channels, hidden, 3, stride=2, padding=1),
                                     nn.BatchNorm2d(hidden), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(2 * hidden, hidden, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(hidden, n_classes, 1), nn.Upsample(scale_factor=2, mode="bilinear"))

    def forward_features(self, x):
        skip = self.encoder(x)
        bottom = self.bottleneck(skip)
        up = nn.functional.interpolate(bottom, size=skip.shape[-2:], mode="nearest")
        local = self.decoder(torch.cat([skip, up], dim=1))
        return self.head(local), local, bottom

    def forward(self, x):
        return self.forward_features(x)[0]


TINY_FEATURE_MODEL_CFG = {
    "target": "tests.helpers.TinyFeatureNet",
    "params": {"n_channels": 3, "n_classes": 2, "hidden": 8},
}


TINY_BN_MODEL_CFG = {
    "target": "tests.helpers.TinyBNNet",
    "params": {"n_channels": 3, "n_classes": 2, "hidden": 8},
}


TINY_MODEL_CFG = {
    "target": "tests.helpers.TinyNet",
    "params": {"n_channels": 3, "n_classes": 4, "hidden": 8},
}

OPTIMIZER_CFG = {"target": "torch.optim.SGD", "params": {"lr": 0.1}}


# DiffRect's rectifier downsamples 16x and its latent denoiser a further 4x, so anything below
# 64x64 collapses to a zero-sized feature map. Tests that build one must use at least this.
DIFFRECT_MIN_SIZE = 64

# A rectifier small enough for CPU tests. `latent_channels=32` instead of the reference's 256, and
# 4 diffusion steps instead of 10 -- the shapes and the code paths are identical, only the widths
# shrink.
DIFFRECT_KWARGS = {
    "latent_channels": 32,
    "rectifier_timesteps": 4,
    "rectifier_sample_steps": 2,
}


def make_batch(n_labeled: int = 2, n_unlabeled: int = 2, size: int = 16, seed: int = 0,
               expose_hidden: bool = False):
    '''
    Build a batch shaped like `VFSSWindowImageDataset` output, with the labeled rows first
    (the layout `TwoStreamBatchSampler` produces).

    Unlabeled rows carry an all-zeros `segmentation`, exactly as the dataset does for a frame whose
    label is hidden -- so any code that supervises on them is training on a lie.

    Args:
        expose_hidden: add `hidden_segmentation` and `metadata['has_hidden_target']`, as the dataset
            does under `expose_hidden_targets=True`. The hidden mask of an unlabeled row is the real
            one, so a diagnostic scored against it is meaningful; only the last row is marked as
            having no hidden target at all, standing in for a frame from the never-annotated pool.
    '''
    generator = torch.Generator().manual_seed(seed)
    batch_size = n_labeled + n_unlabeled

    images = torch.randn(batch_size, 3, size, size, generator=generator)
    targets = torch.zeros(batch_size, size, size, dtype=torch.long)
    # Give the labeled rows a non-trivial mask (classes 1 and 3, as in multiclass_c2_c4).
    # The boxes scale with `size`, and are written so that `size=16` reproduces the original fixed
    # coordinates exactly. They have to scale: at fixed coordinates a 64x64 batch (which DiffRect
    # needs, since its rectifier downsamples 16x) would put under 1% of its pixels in the
    # foreground, and *every* method fails to overfit it -- a fixture artefact that reads exactly
    # like a broken method.
    targets[:n_labeled, size // 8:size // 2, size // 8:size // 2] = 1
    targets[:n_labeled, 9 * size // 16:14 * size // 16, 9 * size // 16:14 * size // 16] = 3

    is_labeled = torch.zeros(batch_size, dtype=torch.bool)
    is_labeled[:n_labeled] = True

    batch = {
        "image": images,
        "segmentation": targets,
        "metadata": {"is_labeled": is_labeled},
    }

    if expose_hidden:
        hidden = torch.zeros(batch_size, size, size, dtype=torch.long)
        hidden[:, 3 * size // 16:9 * size // 16, 3 * size // 16:9 * size // 16] = 1
        hidden[:, 10 * size // 16:15 * size // 16, 10 * size // 16:15 * size // 16] = 3
        has_hidden = torch.ones(batch_size, dtype=torch.bool)
        if n_unlabeled > 0:
            has_hidden[-1] = False
        batch["hidden_segmentation"] = hidden
        batch["metadata"]["has_hidden_target"] = has_hidden

    return batch


class ToyPatientFrames(torch.utils.data.Dataset):
    '''
    Frames of a few fake patients, with the `metadata` keys the VFSS datasets carry (`paciente_id`,
    `video_id`, `video_frame`, `is_labeled`) -- enough to drive `Experiment`, `PatientValidation` and
    `evaluate_per_class` on CPU in a unit test.
    '''

    def __init__(self, n_patients: int = 3, frames_per_patient: int = 4, size: int = 16, seed: int = 0, **_):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n_patients * frames_per_patient, 3, size, size, generator=generator) * 2 - 1
        self.masks = (self.images[:, 0] > 0).long()
        self.patients = [p for p in range(n_patients) for _ in range(frames_per_patient)]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int):
        patient = self.patients[index]
        return {
            "image": self.images[index],
            "segmentation": self.masks[index],
            "metadata": {"paciente_id": patient, "video_id": patient, "video_frame": f"v{patient}_f{index}",
                         "is_labeled": True, "has_target": True},
        }
