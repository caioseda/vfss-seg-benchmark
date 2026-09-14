'''
Volume-level evaluation, for the ACDC reproduction.

`src/evaluation.py` scores one *slice* at a time, which is the right unit for VFSS: a VFSS frame is
an independent observation and the reported number is a per-frame average. The ACDC literature
reports something else -- per-*volume* metrics over a reassembled 3-D prediction -- and the two are
not interchangeable:

  - Dice over a stack of D slices averages D per-slice scores. A slice where the right ventricle
    simply is not present scores 0 (or NaN) and drags the mean down, while the volumetric Dice it is
    being compared against never sees that slice as a separate observation. On ACDC's basal/apical
    slices this is worth several points.
  - HD95 and ASD are *distances between surfaces*. A per-slice surface is a contour; the volumetric
    surface is a shell. The published millimetre figures are the shell.

So reproducing 82.40 / 71.96 / 10.04 / 2.90 requires evaluating the way those numbers were produced.
`predict_volume` transcribes the reference's `val_2D.test_single_volume`
(github.com/CUHK-AIM-Group/DiffRect, MIT; itself from HiLab-git/SSL4MIS): each slice is resampled to
the network's input size with nearest-neighbour interpolation, forwarded, argmaxed, and resampled
back to the volume's native in-plane size before the stack is scored.

One deliberate divergence from the reference, stated so the numbers are read correctly: the
reference computes `dc`/`jc`/`hd95` with `medpy`, which is GPL-3.0. This repository already declines
to vendor a CC BY-NC ramp-up (see `src/models/ssl/base.py`), so the surface metrics here are
reimplemented on `scipy.ndimage` (BSD) in `src/metrics.py`. Dice and Jaccard are exact set
operations and should agree to floating point; HD95 and ASD are implementation-comparable rather
than bit-identical, because they depend on how the surface voxels are extracted.
'''

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .metrics import volumetric_asd, volumetric_hd95, volumetric_overlap

from typing import Dict, Optional, Sequence

# `dice` and `hd95` keep the names `src/evaluation.py` uses, so `aggregate_per_class` reduces this
# table unchanged. `jaccard` and `asd` are the names the ACDC tables use.
VOLUME_METRIC_NAMES = ("dice", "jaccard", "hd95", "asd")
VOLUME_METRIC_DIRECTION = {"dice": "max", "jaccard": "max", "hd95": "min", "asd": "min"}


def _resample(array: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    '''Nearest-neighbour resample of a 2-D array, matching the reference's `zoom(..., order=0)`.'''
    from scipy.ndimage import zoom

    factors = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    if factors == (1.0, 1.0):
        return array
    return zoom(array, factors, order=order)


@torch.no_grad()
def predict_volume(
    module,
    image: Tensor,
    size: int = 256,
    forward_fn=None,
    batch_size: int = 16,
) -> Tensor:
    '''
    Segment a whole volume slice by slice and reassemble it at the volume's native resolution.

    Args:
        module: a `LitWrapper` (or subclass), already on the target device and in eval mode.
        image: `[D, C, H, W]` for one volume, as `ACDCVolumeDataset` yields it (no batch dimension).
        size: the network's input resolution. Slices are resampled to `size x size` and the
            prediction is resampled back to `H x W` -- the reference's protocol.
        forward_fn: override for the forward pass (Mean Teacher evaluates with the teacher;
            pass `module._eval_forward`).
        batch_size: slices per forward. The reference runs one slice at a time; batching is a pure
            speed-up and changes nothing, because the network has no cross-slice state.

    Returns:
        `[D, H, W]` class indices at the volume's native in-plane size.
    '''
    forward = forward_fn if forward_fn is not None else module.forward
    device = next(module.parameters()).device

    depth, _, height, width = image.shape
    image_np = image.detach().cpu().numpy()

    resized = np.stack([
        np.stack([_resample(image_np[d, c], (size, size), order=0)
                  for c in range(image_np.shape[1])])
        for d in range(depth)
    ])
    inputs = torch.from_numpy(np.ascontiguousarray(resized)).float().to(device)

    predictions = np.empty((depth, height, width), dtype=np.int64)
    for start in range(0, depth, batch_size):
        logits = forward(inputs[start:start + batch_size])
        chunk = logits.argmax(dim=1).cpu().numpy()
        for offset, plane in enumerate(chunk):
            predictions[start + offset] = _resample(plane, (height, width), order=0)

    return torch.from_numpy(predictions)


@torch.no_grad()
def evaluate_volumes(
    module,
    dataset,
    num_classes: int,
    class_ids: Dict[int, str],
    device: Optional[torch.device] = None,
    forward_fn=None,
    size: int = 256,
    surface_metrics: bool = True,
) -> pd.DataFrame:
    '''
    Score every volume in `dataset`, one tidy row per (case, reported class).

    Mirrors the contract of `src/evaluation.py::evaluate_per_class` so that
    `aggregate_per_class(..., group_columns=[...])` reduces the result unchanged.

    Args:
        dataset: an `ACDCVolumeDataset`, yielding `{'image': [D,C,H,W], 'segmentation': [D,H,W],
            'metadata': {'case', 'video_id'}}`.
        class_ids: `{class_index: display_name}`, e.g. `{1: 'RV', 2: 'Myo', 3: 'LV'}`.
        surface_metrics: compute HD95/ASD. They are the expensive half (a Euclidean distance
            transform per class per volume), so validation-time callers turn them off and only the
            final reporting pass turns them on.

    Returns:
        Columns `case, video_id, class, dice, jaccard` (+ `hd95, asd` when requested).
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    module = module.to(device)
    module.eval()

    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = predict_volume(module, sample["image"], size=size, forward_fn=forward_fn)
        target = sample["segmentation"].long()

        dice, jaccard = volumetric_overlap(prediction, target, num_classes, include_background=False)
        metrics = {"dice": dice, "jaccard": jaccard}
        if surface_metrics:
            metrics["hd95"] = volumetric_hd95(prediction, target, num_classes, include_background=False)
            metrics["asd"] = volumetric_asd(prediction, target, num_classes, include_background=False)

        metadata = sample.get("metadata", {})
        for class_id, class_name in class_ids.items():
            row = {
                "case": metadata.get("case"),
                "video_id": _scalar(metadata.get("video_id")),
                "class": class_name,
            }
            for name, values in metrics.items():
                row[name] = float(values[class_id - 1].item())
            rows.append(row)

    return pd.DataFrame(rows)


def _scalar(value):
    return value.item() if torch.is_tensor(value) else value


def aggregate_volumes(per_volume: pd.DataFrame, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    '''
    Reduce the per-volume table with `nanmean`, so a class that is undefined in some volumes drops
    out instead of poisoning the mean (see `src/evaluation.py::aggregate_per_class`, same reasoning).
    '''
    metric_columns = [name for name in VOLUME_METRIC_NAMES if name in per_volume.columns]

    def _nanmean(values) -> float:
        array = values.to_numpy(dtype=float)
        if np.all(np.isnan(array)):
            return float("nan")
        return float(np.nanmean(array))

    keys = list(group_columns) + ["class"]
    aggregated = per_volume.groupby(keys, dropna=False)[metric_columns].agg(_nanmean).reset_index()
    counts = per_volume.groupby(keys, dropna=False).size().rename("n_volumes").reset_index()
    return aggregated.merge(counts, on=keys)
