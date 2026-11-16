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



# ------------------------------------------------------------------ weak geometry, shared pieces

WeakParams = Tuple[bool, float, List[int], float]   # (flip, angle em graus, [dx, dy] em pixels, escala)


def _draw_weak_params(height: int, width: int, generator: Optional[torch.Generator],
                      device: torch.device) -> WeakParams:
    '''Um sorteio da geometria fraca. A ORDEM dos sorteios (flip, ângulo, dx, dy, escala) é fixa.'''
    flip = _rand(generator, device) < WEAK_FLIP_PROB
    angle = _uniform(-WEAK_MAX_ROTATION_DEG, WEAK_MAX_ROTATION_DEG, generator, device)
    max_dx = WEAK_MAX_TRANSLATE * width
    max_dy = WEAK_MAX_TRANSLATE * height
    translate = [
        int(round(_uniform(-max_dx, max_dx, generator, device))),
        int(round(_uniform(-max_dy, max_dy, generator, device))),
    ]
    scale = _uniform(*WEAK_SCALE_RANGE, generator=generator, device=device)
    return flip, angle, translate, scale


def _apply_weak_geometry(x: Tensor, params: WeakParams, interpolation: InterpolationMode,
                         fill: float) -> Tensor:
    '''Aplica flip e depois afim a `x` `[..., H, W]` (todas as dimensões iniciais com a mesma geometria).'''
    flip, angle, translate, scale = params
    if flip:
        x = TF.hflip(x)
    return TF.affine(x, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                     interpolation=interpolation, fill=[fill])


def _warp_mask(mask: Tensor, params: WeakParams) -> Tensor:
    '''`[H, W]` de índices de classe; nearest, fundo fora do quadro.'''
    warped = _apply_weak_geometry(mask.unsqueeze(0).float(), params, InterpolationMode.NEAREST, fill=0.0)
    return warped.squeeze(0).long()


def weak_vector_matrix(params: WeakParams) -> Tensor:
    '''
    Parte linear `L` (2x2, sobre vetores `(dx, dy)` em pixels) da geometria fraca `G = afim ∘ flip`.

    Um campo de deslocamento definido no grid da imagem, `F`, vira no grid aumentado
    `F'(y) = L · F(G^-1(y))`: a reamostragem espacial é a mesma da imagem; os VETORES ainda precisam
    ser girados/escalados (e o dx trocar de sinal no flip). A translação não afeta vetores.
    '''
    from torchvision.transforms.v2.functional._geometry import _get_inverse_affine_matrix
    flip, angle, _, scale = params
    inverse = _get_inverse_affine_matrix([0.0, 0.0], angle, [0.0, 0.0], scale, [0.0, 0.0])
    linear = torch.linalg.inv(torch.tensor([[inverse[0], inverse[1]], [inverse[3], inverse[4]]],
                                           dtype=torch.float64))
    if flip:
        linear = linear @ torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64))
    return linear


def _warp_flow(flow: Tensor, params: WeakParams) -> Tensor:
    '''`[K, 2, H, W]` (canal 0 = dx): reamostra como imagem e transforma os vetores por `L`.'''
    resampled = _apply_weak_geometry(flow.float(), params, InterpolationMode.BILINEAR, fill=0.0)
    linear = weak_vector_matrix(params).to(device=flow.device, dtype=resampled.dtype)
    return torch.einsum("ij,kjhw->kihw", linear, resampled).to(flow.dtype)


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
        params = _draw_weak_params(image.shape[-2], image.shape[-1], generator, device)
        images.append(_apply_weak_geometry(image[i], params, InterpolationMode.BILINEAR, fill=-1.0))
        for slot, mask in enumerate(masks):
            if mask is not None:
                warped[slot].append(_warp_mask(mask[i], params))

    out_image = torch.stack(images, dim=0)
    out_masks = [
        torch.stack(rows, dim=0) if mask is not None else None
        for mask, rows in zip(masks, warped)
    ]
    return out_image, out_masks


def weak_augment_temporal(
    image: Tensor,
    masks: Sequence[Optional[Tensor]],
    neighbors: Tensor,
    flows: Tensor,
    confidences: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, List[Optional[Tensor]], Tensor, Tensor, Optional[Tensor]]:
    '''
    `weak_augment_multi` para o MT temporal: a MESMA geometria por linha vai para o central, as
    máscaras, os K vizinhos, o fluxo e o M_noc, de modo que o transporte continua válido:

        warp(aug(vizinho), aug(fluxo)) ≈ aug(warp(vizinho, fluxo))

    (vale nas duas `warp_convention`, porque o sinal do fluxo não muda sob `L`). O M_noc é reamostrado
    com zero fora do quadro: pixels que a geometria trouxe de fora não têm evidência de movimento.
    Os sorteios são os de `weak_augment_multi` (mesma ordem), então o central sai idêntico para o
    mesmo `generator`.

    Args:
        image: `[B, C, H, W]` em [-1, 1].
        masks: como em `weak_augment_multi`.
        neighbors: `[B, K, C, H, W]` em [-1, 1].
        flows: `[B, K, 2, H, W]`, em pixels, canal 0 = dx.
        confidences: `[B, K, 1, H, W]` ou None.
    '''
    device = image.device
    images: List[Tensor] = []
    warped: List[List[Tensor]] = [[] for _ in masks]
    out_neighbors: List[Tensor] = []
    out_flows: List[Tensor] = []
    out_conf: List[Tensor] = []

    for i in range(image.shape[0]):
        params = _draw_weak_params(image.shape[-2], image.shape[-1], generator, device)
        images.append(_apply_weak_geometry(image[i], params, InterpolationMode.BILINEAR, fill=-1.0))
        for slot, mask in enumerate(masks):
            if mask is not None:
                warped[slot].append(_warp_mask(mask[i], params))
        out_neighbors.append(_apply_weak_geometry(neighbors[i], params, InterpolationMode.BILINEAR, fill=-1.0))
        out_flows.append(_warp_flow(flows[i], params))
        if confidences is not None:
            out_conf.append(_apply_weak_geometry(confidences[i].float(), params,
                                                 InterpolationMode.BILINEAR, fill=0.0).to(confidences.dtype))

    out_masks = [
        torch.stack(rows, dim=0) if mask is not None else None
        for mask, rows in zip(masks, warped)
    ]
    return (torch.stack(images, dim=0), out_masks, torch.stack(out_neighbors, dim=0),
            torch.stack(out_flows, dim=0), torch.stack(out_conf, dim=0) if confidences is not None else None)


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
