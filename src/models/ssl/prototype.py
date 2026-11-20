'''
Perda de protótipos do DTC-TRUS (L_proto), como mixin, para os métodos do E2 que a usam: o próprio
`DTCTRUSLitWrapper` e o `PrototypeTemporalMeanTeacherLitWrapper` (MT temporal + protótipos).

L_proto alinha protótipos entre um vizinho n e o central t de um par (n -> t):
  - Local (FG, features do último mapa antes da cabeça, transportadas pelo fluxo, na interseção FG atual x
    FG anterior transportado).
  - Global (BG, features do gargalo, sem fluxo).
  FG/BG por um limiar adaptativo (percentil 70 das probabilidades FG do frame atual, média do lote, sem
  gradiente). Combinação do original:
      L_proto = w_bg * L_local + w_fg * L_global,   w_fg = fração FG suave média,  w_bg = 1 - w_fg
  -- os pesos parecem TROCADOS em relação ao comentário do código original (o termo FG é pesado pela
  fração BG); reproduzimos como está.

As máscaras vêm de comparações e do limiar (sem gradiente): o gradiente só passa pelas FEATURES. Quem
decide em quais lados há gradiente é o chamador (no DTC-TRUS, nos dois frames; no MT temporal, no
central do student, com o vizinho do teacher, ou nos dois frames do student).

A mixin usa `self.proto_percentile` e `self.warp_convention`, definidos pelo wrapper.
'''

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.optical_flow import warp


def adaptive_threshold(fg_prob: Tensor, percentile: float = 0.7) -> float:
    '''Por amostra, o valor de ordem `int(n * percentile)` das probabilidades ordenadas; média do lote.'''
    flat = fg_prob.detach().flatten(1)
    index = min(int(flat.shape[1] * percentile), flat.shape[1] - 1)
    return float(flat.sort(dim=1).values[:, index].mean().item())


def masked_avg_pool(features: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    '''`[N,C,h,w]` x `[N,1,h,w]` -> `[N,C]`; máscara vazia dá o vetor zero (como no original).'''
    return (features * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(eps)


def resize_flow(flow: Tensor, size: Tuple[int, int]) -> Tensor:
    '''Fluxo em pixels `[N,2,H,W]` levado a `size`, com os vetores escalados (convenção `align_corners`).'''
    height, width = flow.shape[-2:]
    if (height, width) == tuple(size):
        return flow
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale_x = (size[1] - 1) / max(width - 1, 1)
    scale_y = (size[0] - 1) / max(height - 1, 1)
    return torch.stack((resized[:, 0] * scale_x, resized[:, 1] * scale_y), dim=1)


def resize_mask(mask: Tensor, size: Tuple[int, int]) -> Tensor:
    return mask if mask.shape[-2:] == tuple(size) else F.interpolate(mask, size=size, mode="nearest")


class PrototypeLossMixin:

    def prototype_loss(self, current_probs: Tensor, neighbor_probs: Tensor, flow: Tensor,
                       current_local: Tensor, neighbor_local: Tensor,
                       current_global: Tensor, neighbor_global: Tensor) -> Tensor:
        '''L_proto de um lote de pares (n -> t), como no original (ver docstring do módulo).'''
        current_fg_prob = 1.0 - current_probs[:, :1]
        neighbor_fg_prob = 1.0 - neighbor_probs[:, :1]
        threshold = adaptive_threshold(current_fg_prob, self.proto_percentile)

        current_fg = (current_fg_prob > threshold).float()
        neighbor_fg = (neighbor_fg_prob > threshold).float()
        warped_neighbor_fg = (warp(neighbor_fg_prob, flow, self.warp_convention) > threshold).float()

        # Local: features do vizinho transportadas para t; FG na interseção.
        local_size = tuple(current_local.shape[-2:])
        warped_local = warp(neighbor_local, resize_flow(flow, local_size), self.warp_convention)
        fg_mask = resize_mask(current_fg * warped_neighbor_fg, local_size)
        local = 1.0 - F.cosine_similarity(masked_avg_pool(warped_local, fg_mask),
                                          masked_avg_pool(current_local, fg_mask), dim=-1)

        # Global: gargalo, BG de cada frame no próprio referencial (sem fluxo).
        global_size = tuple(current_global.shape[-2:])
        global_ = 1.0 - F.cosine_similarity(
            masked_avg_pool(neighbor_global, resize_mask(1.0 - neighbor_fg, global_size)),
            masked_avg_pool(current_global, resize_mask(1.0 - current_fg, global_size)), dim=-1)

        w_fg = current_fg_prob.mean(dim=(1, 2, 3)).mean().detach()
        w_bg = 1.0 - w_fg
        # Pesos cruzados como no original: o termo FG (local) pela fração BG, o BG (global) pela FG.
        return w_bg * local.mean() + w_fg * global_.mean()
