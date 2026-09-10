'''
Per-class test-set evaluation.

Why this exists instead of just reading `trainer.test()`'s numbers: Lightning aggregates a logged
metric as the mean over *batches*. ASSD/HD95 are NaN whenever a class is present in only one of
prediction/target (see `src/metrics.py`) -- plausible at the 10% budget, where the model may simply
predict nothing for a class -- and a single all-NaN batch poisons the whole epoch average. Here the
per-sample values are accumulated over the entire test set and reduced with `nanmean` exactly once.

It also returns the per-sample table, so results can be grouped by patient or video afterwards.
'''

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchmetrics.functional.segmentation import dice_score, mean_iou

from .metrics import absent_class_to_nan, average_symmetric_surface_distance, hausdorff_distance_95

from typing import Dict, Optional, Sequence

METRIC_NAMES = ("dice", "iou", "assd", "hd95")

# Lower is better for the surface metrics; higher is better for the overlap metrics.
METRIC_DIRECTION = {"dice": "max", "iou": "max", "assd": "min", "hd95": "min"}


def _per_sample_metrics(preds: Tensor, target: Tensor, num_classes: int) -> Dict[str, Tensor]:
    '''All four metrics as `[B, C]` tensors, `C` excluding background (class index `c` -> column `c-1`).'''
    return {
        "dice": dice_score(
            preds, target, num_classes=num_classes,
            include_background=False, average="none", input_format="index",
        ),
        "iou": absent_class_to_nan(mean_iou(
            preds, target, num_classes=num_classes,
            include_background=False, per_class=True, input_format="index",
        )),
        "assd": average_symmetric_surface_distance(
            preds, target, num_classes=num_classes, include_background=False
        ),
        "hd95": hausdorff_distance_95(
            preds, target, num_classes=num_classes, include_background=False
        ),
    }


@torch.no_grad()
def evaluate_per_class(
    module,
    dataloader,
    num_classes: int,
    class_ids: Dict[int, str],
    device: Optional[torch.device] = None,
    forward_fn=None,
) -> pd.DataFrame:
    '''
    Run `module` over `dataloader` and return one tidy row per (sample, reported class).

    Args:
        module: a `LitWrapper` (or subclass). Put in eval mode and moved to `device`.
        dataloader: the test dataloader; its dataset must yield the usual
            `{'image', 'segmentation', 'metadata'}` dict.
        num_classes: the model's class count, including background.
        class_ids: `{class_index: display_name}`, e.g. `{1: 'C2', 3: 'C4'}`.
        device: defaults to CUDA when available. **Strongly prefer a GPU**: ASSD/HD95 measure
            ~170 ms per 8x256x256 batch on an RTX 6000 Ada versus ~18 s on CPU (100x).
        forward_fn: optional override for the forward pass (Mean Teacher evaluates with the teacher;
            passing `module._eval_forward` reproduces that here).

    Returns:
        A DataFrame with columns `video_frame, video_id, paciente_id, class, dice, iou, assd, hd95`.
        Aggregate with `aggregate_per_class`.
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    module = module.to(device)
    module.eval()
    forward = forward_fn if forward_fn is not None else module.forward

    rows = []
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["segmentation"].to(device).long()
        preds = forward(images).argmax(dim=1)

        metrics = _per_sample_metrics(preds, targets, num_classes)
        metadata = batch["metadata"]
        batch_size = images.shape[0]

        for i in range(batch_size):
            base = {
                "video_frame": _meta_item(metadata, "video_frame", i),
                "video_id": _meta_item(metadata, "video_id", i),
                "paciente_id": _meta_item(metadata, "paciente_id", i),
            }
            for class_id, class_name in class_ids.items():
                row = dict(base, **{"class": class_name})
                for metric_name, values in metrics.items():
                    row[metric_name] = float(values[i, class_id - 1].item())
                rows.append(row)

    return pd.DataFrame(rows)


def _meta_item(metadata: Dict, key: str, index: int):
    '''Read one element out of a collated metadata field, which may be a tensor or a list.'''
    if key not in metadata:
        return None
    value = metadata[key][index]
    return value.item() if torch.is_tensor(value) else value


def aggregate_per_class(per_sample: pd.DataFrame, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    '''
    Reduce the per-sample table to one row per class (optionally within `group_columns`), using
    `nanmean` so that undefined surface distances drop out instead of propagating.
    '''
    def _nanmean(values) -> float:
        array = values.to_numpy(dtype=float)
        # An all-NaN group is a real outcome, not an error: it means the class was never predicted
        # *and* never absent-in-both across the whole set, so the surface distance is undefined
        # everywhere. Report NaN quietly instead of letting numpy warn on every group.
        if np.all(np.isnan(array)):
            return float("nan")
        return float(np.nanmean(array))

    keys = list(group_columns) + ["class"]
    aggregated = (
        per_sample
        .groupby(keys, dropna=False)[list(METRIC_NAMES)]
        .agg(_nanmean)
        .reset_index()
    )
    counts = per_sample.groupby(keys, dropna=False).size().rename("n_samples").reset_index()
    return aggregated.merge(counts, on=keys)
