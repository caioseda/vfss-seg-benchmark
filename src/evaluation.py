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


def aggregate_per_patient(per_sample: pd.DataFrame, group_columns: Sequence[str] = (),
                          patient_column: str = "paciente_id") -> pd.DataFrame:
    '''
    The article protocol's aggregation: average the frames of each patient first, then the patients.

    `aggregate_per_class` averages frames directly, so a patient with 60 annotated frames weighs 15x
    one with 4 -- and in INCA 15 of 114 patients hold most of the frames. Here every patient counts
    once. Within a patient the reduction is the same `nanmean` as `aggregate_per_class`, so a frame
    where the class is absent from both prediction and target still drops out of Dice/IoU.

    Returns one row per `group_columns + class` with the four metrics, `n_patients` and `n_samples`.
    '''
    keys = list(group_columns) + ["class"]
    per_patient = aggregate_per_class(per_sample, group_columns=list(group_columns) + [patient_column])
    aggregated = (
        per_patient
        .groupby(keys, dropna=False)[list(METRIC_NAMES)]
        .mean()   # pandas' mean skips NaN: a patient whose class is undefined everywhere drops out
        .reset_index()
    )
    counts = (
        per_patient.groupby(keys, dropna=False)
        .agg(n_patients=(patient_column, "nunique"), n_samples=("n_samples", "sum"))
        .reset_index()
    )
    return aggregated.merge(counts, on=keys)


def summarize_over_folds(per_fold: pd.DataFrame, group_columns: Sequence[str],
                         fold_column: str = "fold",
                         metrics: Sequence[str] = METRIC_NAMES) -> pd.DataFrame:
    '''
    Mean and sample standard deviation (ddof=1) across folds, one row per `group_columns + class`.

    The spread measures sensitivity to the partition. The folds share training patients, so they are
    not independent replicates and the std is not a standard error.
    '''
    keys = list(group_columns) + ["class"]
    grouped = per_fold.groupby(keys, dropna=False)
    summary = grouped[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = grouped[fold_column].nunique()
    return summary.reset_index()


def paired_fold_differences(per_fold: pd.DataFrame, by: str, reference,
                            fold_column: str = "fold", group_columns: Sequence[str] = (),
                            metrics: Sequence[str] = METRIC_NAMES) -> pd.DataFrame:
    '''
    Per-fold difference of every level of `by` against `reference`, matched on the same fold, class
    and `group_columns` (e.g. seed). Positive means higher than the reference -- better for Dice/IoU,
    worse for ASSD/HD95 (see `METRIC_DIRECTION`).

    Returns the per-fold differences; reduce them with `summarize_over_folds(diffs, [by, ...])`.
    '''
    keys = [fold_column, "class"] + list(group_columns)
    reference_rows = per_fold[per_fold[by] == reference]
    if reference_rows.empty:
        raise ValueError(f"No rows with {by}={reference!r} to use as the reference.")
    merged = per_fold[per_fold[by] != reference].merge(
        reference_rows[keys + list(metrics)], on=keys, suffixes=("", "_reference"), how="inner",
    )
    for metric in metrics:
        merged[metric] = merged[metric] - merged[f"{metric}_reference"]
    merged["reference"] = reference
    return merged[[by, "reference"] + keys + list(metrics)]
