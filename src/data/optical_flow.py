'''
Transporte de predições entre frames por fluxo óptico, para a consistência temporal do E2.

Extraído de `notebooks/hyphotesis_testing/experiment_optical_flow_mask_transport.ipynb`, que por sua vez
segue o DTC-TRUS (https://github.com/DYDevelop/DTC-TRUS). Notação: o vizinho `n` (t-1, t+1, ...) tem sua
predição levada para o referencial do frame central `t`:

    P~_{n->t} = W(P_n, F)          M_noc = exp(-alpha * |I_t - W(I_n, F)|)

Duas convenções de amostragem (`warp_convention`), ambas lidas a partir do grid do DESTINO t:

    dtc_trus   F = F_{n->t}, amostra em x - F(x)   (o que o DTC-TRUS faz; lê a velocidade no destino)
    backward   F = F_{t->n}, amostra em x + F(x)   (o fluxo está definido no grid de t)

Em movimento localmente uniforme as duas coincidem; o notebook discute onde cada uma erra.

O que roda onde:
  - o FLUXO e o M_noc são calculados na CPU, nos workers do DataLoader (`flow_and_confidence`), porque
    dependem só das imagens -- não do modelo -- e o Farneback do OpenCV não roda na GPU;
  - o WARP da predição do teacher roda na GPU dentro do `training_step` (`warp`), porque a predição só
    existe lá.
`warp` é a mesma função nos dois lados, então a imagem usada no M_noc e a predição transportada passam
pelo mesmo amostrador.

Métodos: `farneback` (padrão), `dis`, `identity` (fluxo zero: consistência direta sem transporte). RAFT
fica como ponto de extensão (precisa de GPU e de um forward extra por par; ver `compute_flow`).
'''

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

FLOW_METHODS = ("farneback", "dis", "identity")
WARP_CONVENTIONS = ("dtc_trus", "backward")

# Pré-processamento dos métodos clássicos (tabela do notebook). `vfss` omite o CLAHE: o mapeamento dele é
# calculado por região, e a entrada do bolo muda o histograma local e imita movimento.
FLOW_PREPROCESS_CONFIGS = {
    "dtc_rgb": dict(bilateral=(5, 50, 50), clahe=True, gauss_sigma=0.3),
    "dtc_ultrasound": dict(bilateral=(9, 75, 75), clahe=True, gauss_sigma=0.5),
    "vfss": dict(bilateral=(5, 50, 50), clahe=False, gauss_sigma=0.3),
    "none": None,
}

# Padrão do DTC-TRUS.
DEFAULT_FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
                                poly_sigma=1.2, flags=0)
DEFAULT_FARNEBACK_MEDIAN_KSIZE = 3

# exp(-1.0*|soma nos canais|) sobre imagens normalizadas com ImageNet equivale, em cinza repetido e
# escala [0,1], a alpha = sum(1/std) ~= 13.27.
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_NOC_ALPHA = float(sum(1.0 / s for s in IMAGENET_STD))


def preprocess_for_flow(image: np.ndarray, preset: str = "vfss") -> np.ndarray:
    '''[H,W] em [0,1] -> uint8, com o pré-processamento do `preset` aplicado.'''
    import cv2

    if preset not in FLOW_PREPROCESS_CONFIGS:
        raise ValueError(f"flow_preprocess deve ser um de {tuple(FLOW_PREPROCESS_CONFIGS)}, recebeu '{preset}'")
    out = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    config = FLOW_PREPROCESS_CONFIGS[preset]
    if config is None:
        return out
    d, sigma_color, sigma_space = config["bilateral"]
    out = cv2.bilateralFilter(out, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    if config["clahe"]:
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    return cv2.GaussianBlur(out, (3, 3), config["gauss_sigma"])


def compute_flow(src: np.ndarray, dst: np.ndarray, method: str = "farneback", preprocess: str = "vfss",
                 farneback_params: Optional[Dict] = None,
                 median_ksize: int = DEFAULT_FARNEBACK_MEDIAN_KSIZE) -> np.ndarray:
    '''
    Fluxo F_{src->dst} em pixels, `[2,H,W]` float32, para imagens `[H,W]` em [0,1].

    `F(p)` diz que o pixel `p` de `src` vai para `p + F(p)` em `dst`.
    '''
    if method == "identity":
        return np.zeros((2, *src.shape), dtype=np.float32)
    if method.startswith("raft"):
        raise NotImplementedError(
            "RAFT não é calculado no DataLoader: exige GPU e um forward por par. Ponto de extensão -- "
            "calcular no `training_step` com torchvision.models.optical_flow e passar o fluxo a `warp`."
        )
    import cv2

    # Um thread por worker: o paralelismo vem dos workers do DataLoader. Com o padrão do OpenCV (um
    # thread por núcleo em cada worker) 16 workers disputam 96 núcleos x16 e atrasam o próprio treino.
    if cv2.getNumThreads() != 1:
        cv2.setNumThreads(1)
    a = preprocess_for_flow(src, preprocess)
    b = preprocess_for_flow(dst, preprocess)
    if method == "farneback":
        params = dict(DEFAULT_FARNEBACK_PARAMS)
        params.update(farneback_params or {})
        flow = cv2.calcOpticalFlowFarneback(a, b, None, **params)
        if median_ksize:
            flow = np.stack([cv2.medianBlur(np.ascontiguousarray(flow[..., i]), median_ksize)
                             for i in range(2)], axis=-1)
    elif method == "dis":
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)
    else:
        raise ValueError(f"flow_method deve ser um de {FLOW_METHODS}, recebeu '{method}'")
    return np.ascontiguousarray(flow.transpose(2, 0, 1)).astype(np.float32)


def sampling_flow(center: np.ndarray, neighbor: np.ndarray, convention: str = "dtc_trus", **flow_kwargs) -> np.ndarray:
    '''
    O fluxo que `warp` usa para levar algo do vizinho para o centro, já na convenção pedida:
    `dtc_trus` -> F_{n->t}; `backward` -> F_{t->n}.
    '''
    if convention == "dtc_trus":
        return compute_flow(neighbor, center, **flow_kwargs)
    if convention == "backward":
        return compute_flow(center, neighbor, **flow_kwargs)
    raise ValueError(f"warp_convention deve ser uma de {WARP_CONVENTIONS}, recebeu '{convention}'")


def _sign(convention: str) -> float:
    if convention == "dtc_trus":
        return -1.0
    if convention == "backward":
        return 1.0
    raise ValueError(f"warp_convention deve ser uma de {WARP_CONVENTIONS}, recebeu '{convention}'")


def warp(x: torch.Tensor, flow: torch.Tensor, convention: str = "dtc_trus") -> torch.Tensor:
    '''
    Amostra `x` `[B,C,H,W]` em `base + sign*flow` (fluxo em pixels, `[B,2,H,W]`, canal 0 = dx).

    `sign = -1` para `dtc_trus`, `+1` para `backward`. Bilinear, borda replicada, `align_corners=True`
    -- exatamente como no notebook.
    '''
    batch, _, height, width = x.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
                            torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype), indexing="ij")
    base = torch.stack((xs, ys), dim=-1)[None].expand(batch, -1, -1, -1)
    flow = flow.to(device=x.device, dtype=x.dtype)
    dx = 2.0 * flow[:, 0] / max(width - 1, 1)
    dy = 2.0 * flow[:, 1] / max(height - 1, 1)
    grid = base + _sign(convention) * torch.stack((dx, dy), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def motion_confidence(center: torch.Tensor, warped_neighbor: torch.Tensor, alpha: float = DEFAULT_NOC_ALPHA) -> torch.Tensor:
    '''M_noc = exp(-alpha*|I_t - W(I_n)|): perto de 1 onde o warp explica a imagem.'''
    return torch.exp(-alpha * (center - warped_neighbor).abs())


def flow_and_confidence(center: np.ndarray, neighbor: np.ndarray, convention: str = "dtc_trus",
                        alpha: float = DEFAULT_NOC_ALPHA, **flow_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Para um par (centro, vizinho) `[H,W]` em [0,1]: o fluxo de amostragem `[2,H,W]` e o M_noc `[1,H,W]`.
    É o que o dataset entrega por vizinho; o warp da predição do teacher usa o mesmo fluxo.
    '''
    flow = torch.from_numpy(sampling_flow(center, neighbor, convention=convention, **flow_kwargs))
    neighbor_t = torch.from_numpy(np.ascontiguousarray(neighbor, dtype=np.float32))[None, None]
    center_t = torch.from_numpy(np.ascontiguousarray(center, dtype=np.float32))[None, None]
    warped = warp(neighbor_t, flow[None], convention)
    confidence = motion_confidence(center_t, warped, alpha)[0]
    return flow, confidence
