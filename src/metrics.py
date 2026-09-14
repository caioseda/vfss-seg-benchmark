from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torchmetrics.functional.segmentation.utils import edge_surface_distance

DistanceMetric = str  # "euclidean" | "chessboard" | "taxicab"


def absent_class_to_nan(values: Tensor) -> Tensor:
    '''
    Normalise torchmetrics' "class absent from both prediction and target" sentinel to NaN.

    The metric families in use disagree on this degenerate case: `dice_score(average="none")`
    returns NaN, `average_symmetric_surface_distance`/`hausdorff_distance_95` below return 0.0
    (perfect agreement that the structure is not there), but `mean_iou(per_class=True)` returns
    **-1.0**. Averaged in as-is that sentinel produces a *negative* IoU -- and with
    `target_variant='multiclass_c2_c4'` class 2 (C3) is absent from every mask, so it would be hit
    on every single sample.

    Mapping it to NaN makes every metric agree on one convention: aggregate with `.nanmean()`.
    '''
    return torch.where(values < 0, torch.full_like(values, float("nan")), values)


def _class_masks(x: Tensor, num_classes: int, include_background: bool) -> Tensor:
    '''Convert an index-format segmentation map `[B, H, W]` (class-index values) into one-hot
    boolean masks `[B, C, H, W]`, optionally dropping the background class (index 0).'''
    one_hot = torch.nn.functional.one_hot(x.long(), num_classes=num_classes).permute(0, 3, 1, 2).bool()
    if not include_background:
        one_hot = one_hot[:, 1:]
    return one_hot


def _pairwise_surface_distances(
    preds_masks: Tensor,
    target_masks: Tensor,
    distance_metric: DistanceMetric,
    spacing: Optional[Sequence[float]],
    reduce: str,
) -> Tensor:
    '''
    Shared implementation for `average_symmetric_surface_distance`/`hausdorff_distance_95`: loops
    over (batch, class) pairs and reduces each pair's symmetric surface distances with `reduce`
    ("mean" for ASSD, "hd95" for the 95th-percentile Hausdorff distance).

    Degenerate cases per (batch, class) pair:
    - Class absent from both prediction and target: distance is defined as 0.0 (perfect agreement
      that the structure is not present).
    - Class present in only one of prediction/target: distance is undefined and reported as NaN.
      Aggregate the returned tensor with `.nanmean()` (not `.mean()`) so a single such pair does not
      turn an entire batch/epoch average into NaN.

    Returns a `[B, C]` tensor (`C` excludes the background class when it was excluded from the masks).
    '''
    batch_size, num_included_classes = preds_masks.shape[:2]
    out = torch.full((batch_size, num_included_classes), float("nan"), device=preds_masks.device)

    for b in range(batch_size):
        for c in range(num_included_classes):
            pred_mask = preds_masks[b, c]
            target_mask = target_masks[b, c]

            if not pred_mask.any() and not target_mask.any():
                out[b, c] = 0.0
                continue

            dist_pred_to_target, dist_target_to_pred = edge_surface_distance(
                pred_mask, target_mask, distance_metric=distance_metric, spacing=spacing, symmetric=True
            )
            all_distances = torch.cat([dist_pred_to_target, dist_target_to_pred])

            if all_distances.numel() == 0 or torch.isinf(all_distances).any():
                # Class present in only one of pred/target -> undefined, left as NaN.
                continue

            out[b, c] = all_distances.mean() if reduce == "mean" else torch.quantile(all_distances, 0.95)

    return out


def average_symmetric_surface_distance(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    distance_metric: DistanceMetric = "euclidean",
    spacing: Optional[Sequence[float]] = None,
) -> Tensor:
    '''
    Average Symmetric Surface Distance (ASSD) for semantic segmentation.

    Args:
        preds: predicted segmentation map, index format `[B, H, W]` (class-index values, e.g. `logits.argmax(dim=1)`).
        target: target segmentation map, index format `[B, H, W]`.
        num_classes: total number of classes (including background).
        include_background: whether to include the background class (index 0) in the per-class output.
        distance_metric: one of "euclidean", "chessboard", "taxicab" (see `torchmetrics.functional.segmentation`).
        spacing: pixel spacing along each spatial dimension; defaults to isotropic unit spacing.

    Returns:
        A `[B, C]` tensor of ASSD per batch element and class (`C = num_classes - 1` if
        `include_background=False`). See `_pairwise_surface_distances` for the convention used for
        classes absent from prediction and/or target. Aggregate with `.nanmean()`.
    '''
    preds_masks = _class_masks(preds, num_classes, include_background)
    target_masks = _class_masks(target, num_classes, include_background)
    return _pairwise_surface_distances(preds_masks, target_masks, distance_metric, spacing, reduce="mean")


def hausdorff_distance_95(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    distance_metric: DistanceMetric = "euclidean",
    spacing: Optional[Sequence[float]] = None,
) -> Tensor:
    '''
    95th-percentile Hausdorff Distance (HD95) for semantic segmentation. Same arguments, return
    shape and degenerate-case convention as `average_symmetric_surface_distance` -- see there.
    '''
    preds_masks = _class_masks(preds, num_classes, include_background)
    target_masks = _class_masks(target, num_classes, include_background)
    return _pairwise_surface_distances(preds_masks, target_masks, distance_metric, spacing, reduce="hd95")


# ------------------------------------------------------------------------------------------------
# Volumetric (N-d) surface distances.
#
# The 2-D helpers above cannot be reused for whole volumes: torchmetrics'
# `edge_surface_distance` is rank-2 only (it raises
# `ValueError: Expected argument 'x' to be of rank 2 but got rank '3'`), and `_class_masks`
# hardcodes the 2-D `permute(0, 3, 1, 2)`.
#
# This matters for the ACDC reproduction, where the published HD95/ASD are computed over the
# reconstructed 3-D volume, not per slice -- a per-slice average is a different, systematically
# smaller number.
#
# Implemented on `scipy.ndimage` (BSD) rather than `medpy` (which is where the reference gets
# `metric.binary.hd95`): medpy is GPL-3.0, and this repository already reimplements rather than
# vendors when a licence would be inherited (see the note on `sigmoid_rampup` in
# `src/models/ssl/base.py`). The consequence is that HD95/ASD here are *implementation-comparable*
# but not bit-identical to published medpy numbers; Dice and Jaccard are unaffected.
# ------------------------------------------------------------------------------------------------

def _surface_distances(pred_mask: "np.ndarray", target_mask: "np.ndarray",
                       spacing: Optional[Sequence[float]] = None) -> "np.ndarray":
    '''
    Symmetric surface distances between two boolean masks of identical N-d shape.

    The surface of a mask is its set of voxels having at least one background 6/26-neighbour, i.e.
    `mask ^ binary_erosion(mask)`. Distances are read off the Euclidean distance transform of the
    *other* mask's complement, which is exactly medpy's `__surface_distances` construction.

    Returns the concatenation of pred->target and target->pred distances (empty if either surface
    is empty).
    '''
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    footprint = generate_binary_structure(pred_mask.ndim, 1)
    pred_surface = pred_mask ^ binary_erosion(pred_mask, structure=footprint, border_value=0)
    target_surface = target_mask ^ binary_erosion(target_mask, structure=footprint, border_value=0)

    if not pred_surface.any() or not target_surface.any():
        return np.empty(0, dtype=float)

    to_target = distance_transform_edt(~target_surface, sampling=spacing)
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([to_target[pred_surface], to_pred[target_surface]])


def volumetric_surface_distances(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    spacing: Optional[Sequence[float]] = None,
    reduce: str = "mean",
) -> Tensor:
    '''
    Per-class surface distance over a whole N-d volume.

    Args:
        preds / target: index-format label maps of identical shape, e.g. `[D, H, W]`. Unlike the
            2-D helpers there is **no batch dimension**: one call, one volume.
        reduce: "mean" for ASD, "hd95" for the 95th percentile.

    Returns:
        A `[C]` tensor. Degenerate cases follow the same convention as
        `_pairwise_surface_distances`: absent from both -> 0.0, present in only one -> NaN.
        Aggregate with `.nanmean()`.
    '''
    import numpy as np

    preds_np = preds.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    class_ids = list(range(num_classes) if include_background else range(1, num_classes))

    out = torch.full((len(class_ids),), float("nan"), dtype=torch.float32)
    for column, class_id in enumerate(class_ids):
        pred_mask = preds_np == class_id
        target_mask = target_np == class_id

        if not pred_mask.any() and not target_mask.any():
            out[column] = 0.0
            continue
        if not pred_mask.any() or not target_mask.any():
            continue  # undefined, stays NaN

        distances = _surface_distances(pred_mask, target_mask, spacing=spacing)
        if distances.size == 0:
            continue
        out[column] = float(distances.mean() if reduce == "mean" else np.percentile(distances, 95))

    return out


def volumetric_asd(preds: Tensor, target: Tensor, num_classes: int,
                   include_background: bool = False,
                   spacing: Optional[Sequence[float]] = None) -> Tensor:
    '''Average Symmetric Surface Distance over a whole volume. `[C]`; see `volumetric_surface_distances`.'''
    return volumetric_surface_distances(preds, target, num_classes, include_background, spacing, reduce="mean")


def volumetric_hd95(preds: Tensor, target: Tensor, num_classes: int,
                    include_background: bool = False,
                    spacing: Optional[Sequence[float]] = None) -> Tensor:
    '''95th-percentile Hausdorff Distance over a whole volume. `[C]`; see `volumetric_surface_distances`.'''
    return volumetric_surface_distances(preds, target, num_classes, include_background, spacing, reduce="hd95")


def volumetric_overlap(preds: Tensor, target: Tensor, num_classes: int,
                       include_background: bool = False) -> "Tuple[Tensor, Tensor]":
    '''
    Per-class Dice and Jaccard over a whole volume, as `([C], [C])`.

    Computed here rather than via torchmetrics because the published ACDC numbers are volumetric:
    `dice_score` on a `[D, ...]` stack would average D per-slice scores, which is a different
    quantity (empty slices score 0 or NaN and drag the mean).

    A class absent from both prediction and target scores NaN, not 1.0 -- "we agree it is not
    there" is not evidence of segmentation quality and must not inflate the average.
    '''
    class_ids = list(range(num_classes) if include_background else range(1, num_classes))
    dice = torch.full((len(class_ids),), float("nan"), dtype=torch.float32)
    jaccard = torch.full((len(class_ids),), float("nan"), dtype=torch.float32)

    for column, class_id in enumerate(class_ids):
        pred_mask = preds == class_id
        target_mask = target == class_id
        pred_sum = int(pred_mask.sum())
        target_sum = int(target_mask.sum())
        if pred_sum == 0 and target_sum == 0:
            continue
        intersection = float((pred_mask & target_mask).sum())
        union = float((pred_mask | target_mask).sum())
        dice[column] = 2.0 * intersection / (pred_sum + target_sum)
        jaccard[column] = intersection / union

    return dice, jaccard
