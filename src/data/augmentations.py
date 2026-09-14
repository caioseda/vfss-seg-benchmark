'''
Batch-level augmentations for semi-supervised training.

These operate on already-collated, already-normalised batches (on whatever device they arrive on)
rather than inside the `Dataset`, for two reasons:

  - FixMatch needs *two aligned views* of the same tensor (a weak one that produces the pseudo-label
    and a strong one that is trained against it). Producing both inside `__getitem__` would mean
    loading and preprocessing each frame twice.
  - `VFSSFrameDatasetBase` applies `image_transform` and `target_transform` independently
    (see `src/data/vfss_frame_dataset.py`), so any *random* geometric op wired in there would
    desynchronise the image from its mask. Here the geometric parameters are drawn once and applied
    to both.

Images reach these functions in **[-1, 1]** (`VFSSFrameDatasetBase._preprocess_image` does a
per-image min-max to [0, 1] and then rescales), so every photometric op converts to [0, 1], applies
itself, and converts back.
'''

import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import InterpolationMode

from typing import List, Optional, Sequence, Tuple

# Weak-augmentation geometry, matching the mild policy used by 2D semi-supervised segmentation
# baselines (random flip + small affine); strong enough to decorrelate the two views, small enough
# that the pseudo-label stays trustworthy.
WEAK_FLIP_PROB = 0.5
WEAK_MAX_ROTATION_DEG = 10.0
WEAK_MAX_TRANSLATE = 0.05
WEAK_SCALE_RANGE = (0.9, 1.1)


def _rand(generator: Optional[torch.Generator], device: torch.device) -> float:
    return float(torch.rand((), generator=generator, device=device))


def _uniform(low: float, high: float, generator: Optional[torch.Generator], device: torch.device) -> float:
    return low + (high - low) * _rand(generator, device)


def _to_unit(image: Tensor) -> Tensor:
    '''[-1, 1] -> [0, 1].'''
    return (image + 1.0) / 2.0


def _to_signed(image: Tensor) -> Tensor:
    '''[0, 1] -> [-1, 1].'''
    return image * 2.0 - 1.0


def weak_augment_multi(
    image: Tensor,
    masks: Sequence[Optional[Tensor]],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, List[Optional[Tensor]]]:
    '''
    `weak_augment` for **several masks under one draw**.

    DiffRect needs the supervised target *and* the hidden diagnostic target warped by the same
    geometry as the image; calling `weak_augment` twice would draw two different transforms and
    silently misalign the pseudo-label quality curves from the loss.

    The random draws (flip, angle, dx, dy, scale) are consumed in a fixed order that does not
    depend on how many masks are passed, so for a given seeded `generator` this produces exactly
    the same image as `weak_augment` with zero or one mask -- pinned by
    `tests/test_ssl_methods.py::TestWeakAugmentMulti`.

    Args:
        image: `[B, C, H, W]` in [-1, 1].
        masks: sequence of optional `[B, H, W]` class-index tensors. `None` entries pass through
            as `None`, so a caller can hand over an absent diagnostic target without branching.

    Returns:
        `(augmented_image, [augmented_mask, ...])`, positionally aligned with `masks`.
    '''
    device = image.device
    images: List[Tensor] = []
    warped: List[List[Tensor]] = [[] for _ in masks]

    for i in range(image.shape[0]):
        img_i = image[i]
        # Masks are transformed as [1, H, W] so torchvision treats them as an image, then squeezed back.
        masks_i = [m[i].unsqueeze(0).float() if m is not None else None for m in masks]

        if _rand(generator, device) < WEAK_FLIP_PROB:
            img_i = TF.hflip(img_i)
            masks_i = [TF.hflip(m) if m is not None else None for m in masks_i]

        angle = _uniform(-WEAK_MAX_ROTATION_DEG, WEAK_MAX_ROTATION_DEG, generator, device)
        max_dx = WEAK_MAX_TRANSLATE * img_i.shape[-1]
        max_dy = WEAK_MAX_TRANSLATE * img_i.shape[-2]
        translate = [
            int(round(_uniform(-max_dx, max_dx, generator, device))),
            int(round(_uniform(-max_dy, max_dy, generator, device))),
        ]
        scale = _uniform(*WEAK_SCALE_RANGE, generator=generator, device=device)

        img_i = TF.affine(
            img_i, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=[-1.0],
        )
        for slot, mask_i in enumerate(masks_i):
            if mask_i is None:
                continue
            warped[slot].append(TF.affine(
                mask_i, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST, fill=[0.0],
            ).squeeze(0).long())

        images.append(img_i)

    out_image = torch.stack(images, dim=0)
    out_masks = [
        torch.stack(rows, dim=0) if mask is not None else None
        for mask, rows in zip(masks, warped)
    ]
    return out_image, out_masks


def weak_augment(
    image: Tensor,
    mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    '''
    Geometric weak augmentation: random horizontal flip + small affine, drawn **once per batch
    element** and applied identically to the image (bilinear) and the mask (nearest).

    Args:
        image: `[B, C, H, W]` in [-1, 1].
        mask: optional `[B, H, W]` class indices, transformed with the same parameters.

    Returns:
        `(augmented_image, augmented_mask)`; the mask is None if none was given.
    '''
    out_image, out_masks = weak_augment_multi(image, [mask], generator=generator)
    return out_image, out_masks[0]


def strong_augment(
    image: Tensor,
    generator: Optional[torch.Generator] = None,
    cutout_prob: float = 0.5,
    cutout_max_fraction: float = 0.3,
) -> Tuple[Tensor, Tensor]:
    '''
    Photometric strong augmentation -- deliberately **no geometry**, so that a pseudo-label taken
    from the weak view stays pixel-aligned with this view and needs no inverse warp.

    Applies, per batch element: brightness, contrast and gamma jitter, gaussian blur, gaussian
    noise, and optionally a cutout square.

    Args:
        image: `[B, C, H, W]` in [-1, 1] (typically the *output* of `weak_augment`).

    Returns:
        `(augmented_image, valid_mask)` where `valid_mask` is `[B, H, W]` bool and is False inside
        cutout holes -- those pixels carry no evidence and must be dropped from the loss.
    '''
    device = image.device
    batch_size, _, height, width = image.shape

    out = _to_unit(image.clone())
    valid = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)

    for i in range(batch_size):
        img_i = out[i]

        img_i = TF.adjust_brightness(img_i, _uniform(0.6, 1.4, generator, device))
        img_i = TF.adjust_contrast(img_i, _uniform(0.6, 1.4, generator, device))
        img_i = TF.adjust_gamma(img_i.clamp(0, 1), _uniform(0.7, 1.4, generator, device))

        if _rand(generator, device) < 0.5:
            sigma = _uniform(0.1, 1.5, generator, device)
            img_i = TF.gaussian_blur(img_i, kernel_size=[5, 5], sigma=[sigma, sigma])

        noise_std = _uniform(0.0, 0.05, generator, device)
        img_i = img_i + torch.randn(img_i.shape, generator=generator, device=device) * noise_std

        out[i] = img_i.clamp(0.0, 1.0)

        if _rand(generator, device) < cutout_prob:
            hole_h = int(height * _uniform(0.1, cutout_max_fraction, generator, device))
            hole_w = int(width * _uniform(0.1, cutout_max_fraction, generator, device))
            top = int(_uniform(0, max(height - hole_h, 1), generator, device))
            left = int(_uniform(0, max(width - hole_w, 1), generator, device))
            out[i, :, top:top + hole_h, left:left + hole_w] = 0.5  # mid-grey in [0, 1]
            valid[i, top:top + hole_h, left:left + hole_w] = False

    return _to_signed(out), valid


def gaussian_noise(image: Tensor, std: float = 0.1, clamp: float = 0.2,
                   generator: Optional[torch.Generator] = None) -> Tensor:
    '''
    The input perturbation used by Mean Teacher: `clamp(randn * std, -clamp, +clamp)` added to the
    image. Matches `train_mean_teacher_2D.py` in HiLab-git/SSL4MIS (std 0.1, clamped at 0.2).
    '''
    noise = torch.clamp(torch.randn(image.shape, generator=generator, device=image.device) * std, -clamp, clamp)
    return image + noise
