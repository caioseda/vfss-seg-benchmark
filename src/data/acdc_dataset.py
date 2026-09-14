'''
ACDC (Automated Cardiac Diagnosis Challenge) for the semi-supervised reproduction.

**This is a benchmark harness, not a second first-class dataset.** It exists so the SSL methods in
`src/models/ssl/` can be checked against published numbers before being trusted on VFSS, where no
published number exists. Nothing in the VFSS pipeline was changed to accommodate it: instead these
classes present the interface `DataModuleFromConfig` and `TwoStreamBatchSampler` already require.

What that interface actually is (all of it discovered by reading, not by documentation):

  - `dataset.video_frame_df` -- a `pandas.DataFrame` in the dataset's own index order, read by
    `src/data/samplers.py::labeled_unlabeled_indices` and `frame_index_metadata`. It must carry
    `is_labeled` in numpy `bool` dtype (an `object` column silently produces `-2`/`-1` under `~`,
    both truthy, and *both* index lists come out wrong without raising), plus integer `video_id`
    and `frame_id`. Those two are required **even under `unlabeled_policy: global`**, because
    `DataModuleFromConfig.setup` injects `frame_metadata` whenever the sampler class declares the
    parameter, which `TwoStreamBatchSampler` does unconditionally.
  - `__getitem__` -> `{'image', 'segmentation', 'metadata': {'is_labeled': ...}}`, and under
    `expose_hidden_targets` also `hidden_segmentation` + `metadata['has_hidden_target']`.
  - A row whose label is hidden carries an **all-zeros placeholder** in `segmentation`. Every SSL
    wrapper selects supervised rows via `metadata['is_labeled']` and never by mask content; the
    placeholder is what makes a violation of that rule show up as a broken model rather than as a
    crash.

For ACDC the video/frame analogy is exact: `video_id` is the patient number and `frame_id` the slice
index within the volume, so the video-conditioned sampling policies of experiment X1B are
meaningful here too (`same_video` = "another slice of the same heart").

Splits and label budgets
------------------------
Both come from SSL4MIS (`data/ACDC/*.list`) rather than from this repository's seeded splitter, so
the comparison is against the published setup rather than against a same-dataset-different-split
rerun. `scripts/prepare_acdc.py` downloads them. The labeled subset is the **first N rows of
`train_slices.list` in file order**, because SSL4MIS selects it with `range(0, labeled_slice_num)`;
drawing it at random instead would be a different experiment wearing the same name.
'''

import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import Dict, List, Optional, Sequence

# `patients_to_slices` from `train_diffrect_ACDC.py:70-82`: how many *slices* the reference reveals
# for a given number of labeled *patients*. 1% of ACDC is 1 patient, 5% is 7, 10% is 14.
ACDC_PATIENTS_TO_SLICES = {1: 32, 3: 68, 7: 136, 14: 256, 21: 396, 28: 512, 35: 664, 140: 1312}

# The label budgets of the paper's Table 1, as (label ratio -> labeled patients).
ACDC_LABEL_RATIOS = {"1%": 1, "5%": 7, "10%": 14, "100%": 140}

ACDC_NUM_CLASSES = 4  # background, RV cavity, myocardium, LV cavity
ACDC_CLASS_IDS = {1: "RV", 2: "Myo", 3: "LV"}


def _read_list(root: pathlib.Path, name: str) -> List[str]:
    path = root / "lists" / f"{name}.list"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_acdc.py --root {root}` first."
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _patient_number(case: str) -> int:
    '''`patient012_frame01` -> 12. Used as `video_id`.'''
    return int(case.split("_")[0].replace("patient", ""))


def resize_slice(array: np.ndarray, size: int, order: int) -> np.ndarray:
    '''
    Resample a 2-D slice to `size x size`, the way `val_2D.test_single_volume` does.

    The reference uses `scipy.ndimage.zoom` with `order=0` for *both* image and label. Keeping
    `order` explicit so the image can use bilinear where that is wanted, while labels never can.
    '''
    from scipy.ndimage import zoom

    height, width = array.shape
    if (height, width) == (size, size):
        return array
    return zoom(array, (size / height, size / width), order=order)


def random_rot_flip(image: np.ndarray, label: np.ndarray, rng) -> "tuple":
    '''Transcribed from SSL4MIS / DiffRect `dataloaders/dataset.py::random_rot_flip`.'''
    k = rng.integers(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = rng.integers(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray, rng) -> "tuple":
    '''Transcribed from SSL4MIS / DiffRect `dataloaders/dataset.py::random_rotate`.'''
    from scipy.ndimage import rotate

    angle = int(rng.integers(-20, 20))
    return (rotate(image, angle, order=0, reshape=False),
            rotate(label, angle, order=0, reshape=False))


def random_generator(image: np.ndarray, label: np.ndarray, rng) -> "tuple":
    '''
    The training-time augmentation every 2-D ACDC baseline in this literature uses, transcribed
    from `RandomGenerator` in SSL4MIS / DiffRect: with probability 1/2 a random 90-degree rotation
    plus a random flip, otherwise (with probability 1/4) a random rotation in [-20, 20] degrees.

    **This is not optional.** Trained without it, the supervised baseline at the 10% budget reaches
    train Dice 0.987 against volumetric validation Dice 0.70 *and falling* by 17k iterations -- it
    memorises 256 slices. The published supervised baselines at that budget are around 0.85. The
    omission does not raise anything and does not look wrong in a loss curve; it just puts every
    number in the reproduction 15 points low, and asymmetrically, since FixMatch and DiffRect
    augment inside their wrappers while the supervised and Mean Teacher baselines do not.

    Applied at the volume's native in-plane resolution and *before* the resize to the network's
    input size, matching the reference's order.
    '''
    draw = rng.random()
    if draw > 0.5:
        return random_rot_flip(image, label, rng)
    if rng.random() > 0.5:
        return random_rotate(image, label, rng)
    return image, label


class ACDCSliceDataset(Dataset):
    '''
    ACDC as independent 2-D slices -- the unit the 2-D baselines train and are evaluated on.

    Args:
        root: prepared dataset root (`volumes/`, `lists/`), see `scripts/prepare_acdc.py`.
        split: `train` | `val` | `test`.
        labeled_patients: how many patients' worth of slices keep their label, resolved through
            `ACDC_PATIENTS_TO_SLICES`. `None` (the default) labels everything, which is the
            fully-supervised upper bound. Only meaningful for `split='train'`.
        labeled_slices: absolute override for the above, for a budget not in the reference's table.
        size: square resolution fed to the network (256 in every published ACDC 2-D result).
        expose_hidden_targets: keep the hidden ground truth in `hidden_segmentation` for the
            pseudo-label quality diagnostics. Free on ACDC -- unlike VFSS, every frame really is
            annotated -- and it is the diagnostic that tells you whether a method works *before*
            the final metric does.
        image_channels: repeat the single MRI channel this many times. ACDC is 1-channel; the
            configs set it to match the backbone.
    '''

    def __init__(
        self,
        root,
        split: str = "train",
        labeled_patients: Optional[int] = None,
        labeled_slices: Optional[int] = None,
        size: int = 256,
        expose_hidden_targets: bool = False,
        image_channels: int = 1,
        return_metadata: bool = True,
        augment: Optional[bool] = None,
        seed: int = 1337,
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self.size = size
        self.expose_hidden_targets = expose_hidden_targets
        self.image_channels = image_channels
        self.return_metadata = return_metadata
        # Augment the training split and nothing else: val/test must be deterministic, and the
        # published numbers are measured on unaugmented volumes.
        self.augment = (split == "train") if augment is None else augment
        self._seed = seed
        self._rng = None          # built lazily, per worker -- see `_augmentation_rng`
        self._rng_worker = None

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        cases = _read_list(self.root, split)
        self._volumes: Dict[str, Dict[str, np.ndarray]] = {}
        rows = []
        for case in cases:
            path = self.root / "volumes" / f"{case}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; rerun scripts/prepare_acdc.py")
            with np.load(path) as data:
                # ACDC is small enough (~500 MB total) to hold in RAM, which also keeps the
                # DataLoader workers from each reopening 200 files.
                self._volumes[case] = {"image": data["image"], "label": data["label"]}
            for slice_index in range(self._volumes[case]["image"].shape[0]):
                rows.append((case, _patient_number(case), slice_index))

        frame = pd.DataFrame(rows, columns=["case", "video_id", "frame_id"])

        if split == "train":
            n_labeled = self._resolve_labeled_count(labeled_patients, labeled_slices, len(frame))
        else:
            n_labeled = len(frame)

        # SSL4MIS reveals `range(0, labeled_slice_num)` of `train_slices.list`, i.e. the first N in
        # file order (which is patient order). Not a random draw: the budgets are nested and
        # reproducible precisely because the order is fixed.
        is_labeled = np.zeros(len(frame), dtype=bool)
        is_labeled[:n_labeled] = True
        frame["is_labeled"] = is_labeled
        frame["has_target"] = np.ones(len(frame), dtype=bool)

        self.video_frame_df = frame
        self.n_labeled = n_labeled

    @staticmethod
    def _resolve_labeled_count(labeled_patients, labeled_slices, total: int) -> int:
        if labeled_slices is not None and labeled_patients is not None:
            raise ValueError("pass labeled_patients or labeled_slices, not both")
        if labeled_slices is not None:
            return int(labeled_slices)
        if labeled_patients is None:
            return total
        if labeled_patients not in ACDC_PATIENTS_TO_SLICES:
            raise ValueError(
                f"labeled_patients={labeled_patients} is not one of the reference's budgets "
                f"{sorted(ACDC_PATIENTS_TO_SLICES)}. Pass labeled_slices= to use another."
            )
        return ACDC_PATIENTS_TO_SLICES[labeled_patients]

    def _augmentation_rng(self) -> "np.random.Generator":
        '''
        Per-worker RNG. A single `Generator` built in `__init__` would be *forked* into every
        DataLoader worker with identical state, so `num_workers=4` would draw the same rotation and
        flip for four different slices in lockstep -- less augmentation than it looks like, and no
        error. The ACDC configs pin `num_workers: 0`, but that is a config choice, not a guarantee.
        '''
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        worker = -1 if info is None else info.id
        if self._rng is None or self._rng_worker != worker:
            self._rng = np.random.default_rng([self._seed, worker + 1])
            self._rng_worker = worker
        return self._rng

    def __len__(self) -> int:
        return len(self.video_frame_df)

    def __getitem__(self, index: int) -> Dict:
        row = self.video_frame_df.iloc[index]
        volume = self._volumes[row["case"]]
        slice_index = int(row["frame_id"])

        image = volume["image"][slice_index]
        label = volume["label"][slice_index]
        if self.augment:
            # Before the resize, as the reference does.
            image, label = random_generator(image, label, self._augmentation_rng())
        image = resize_slice(image, self.size, order=0)
        label = resize_slice(label, self.size, order=0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        # The SSL wrappers and `src/data/augmentations.py` assume images in [-1, 1]; the stored
        # volumes are already min-max normalised to [0, 1] by the reference's preprocessing.
        image_tensor = image_tensor * 2.0 - 1.0
        if self.image_channels > 1:
            image_tensor = image_tensor.repeat(self.image_channels, 1, 1)

        label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long()
        is_labeled = bool(row["is_labeled"])

        sample = {
            "image": image_tensor,
            # A hidden-label row gets the all-zeros placeholder, exactly as the VFSS dataset does.
            "segmentation": label_tensor if is_labeled else torch.zeros_like(label_tensor),
        }
        if self.return_metadata:
            sample["metadata"] = {
                "case": row["case"],
                "video_id": int(row["video_id"]),
                "frame_id": slice_index,
                "video_frame": f"{row['case']}_slice_{slice_index}",
                "paciente_id": int(row["video_id"]),
                "is_labeled": is_labeled,
            }
        if self.expose_hidden_targets:
            sample["hidden_segmentation"] = label_tensor
            sample.setdefault("metadata", {})["has_hidden_target"] = True
        return sample


class ACDCVolumeDataset(Dataset):
    '''
    ACDC as whole volumes, for the volumetric evaluation the published numbers use.

    Kept separate from `ACDCSliceDataset` on purpose: `LitWrapper.shared_step` feeds
    `batch['image']` straight to the network, so a volume-shaped batch reaching the ordinary
    validation/test path would hand a 5-D tensor to a 2-D U-Net. Volumes are consumed only by
    `src/evaluation_volume.py` and by the `VolumetricValidation` callback, which slice them
    themselves.
    '''

    def __init__(self, root, split: str = "val", image_channels: int = 1):
        self.root = pathlib.Path(root)
        self.split = split
        self.image_channels = image_channels
        self.cases = _read_list(self.root, split)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict:
        case = self.cases[index]
        with np.load(self.root / "volumes" / f"{case}.npz") as data:
            image, label = data["image"], data["label"]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(1)
        # The same [0, 1] -> [-1, 1] mapping `ACDCSliceDataset.__getitem__` applies. It has to be
        # the same: the network is trained on slices from that class and validated on volumes from
        # this one, and feeding evaluation a different input range than training would degrade
        # every volumetric metric without raising anything.
        image_tensor = image_tensor * 2.0 - 1.0
        if self.image_channels > 1:
            image_tensor = image_tensor.repeat(1, self.image_channels, 1, 1)

        return {
            # `[D, C, H, W]` at native in-plane resolution: `predict_volume` resizes each slice and
            # maps the prediction back, so the metrics are computed in the volume's own geometry,
            # as the reference does.
            "image": image_tensor,
            "segmentation": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "metadata": {"case": case, "video_id": _patient_number(case)},
        }


class ACDCTrain(ACDCSliceDataset):
    def __init__(self, **kwargs):
        kwargs.pop("split", None)
        super().__init__(split="train", **kwargs)


class ACDCVal(ACDCSliceDataset):
    def __init__(self, **kwargs):
        kwargs.pop("split", None)
        super().__init__(split="val", **kwargs)


class ACDCTest(ACDCSliceDataset):
    def __init__(self, **kwargs):
        kwargs.pop("split", None)
        super().__init__(split="test", **kwargs)


class ACDCValVolumes(ACDCVolumeDataset):
    def __init__(self, **kwargs):
        kwargs.pop("split", None)
        super().__init__(split="val", **kwargs)


class ACDCTestVolumes(ACDCVolumeDataset):
    def __init__(self, **kwargs):
        kwargs.pop("split", None)
        super().__init__(split="test", **kwargs)
