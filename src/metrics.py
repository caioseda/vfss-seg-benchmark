from typing import Optional, Sequence

import torch
from torch import Tensor
from torchmetrics.functional.segmentation.utils import edge_surface_distance

DistanceMetric = str  # "euclidean" | "chessboard" | "taxicab"


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
