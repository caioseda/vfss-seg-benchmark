'''
Consistência temporal (TC) das predições em vídeo, para a análise do E2.

Para cada par de frames consecutivos (t, t+1) de um vídeo, rotulados ou não:

    TC-IoU_t  = IoU (W(P_t, F_{t->t+1}), P_{t+1})
    TC-Dice_t = Dice(W(P_t, F_{t->t+1}), P_{t+1})

`P` é a máscara prevista (argmax do caminho de avaliação do método: o teacher, nos MT). O fluxo e o
warp são OS DO TREINO (`src/data/optical_flow.py`): Farneback com o preset `vfss` e convenção
`dtc_trus`, com o vizinho = t e o centro = t+1. Nessa convenção, `F = F_{t->t+1}` e o warp amostra
`P_t` em `x - F(x)` no grid de t+1. É literalmente a fórmula acima, e o mesmo transporte que a perda
temporal do `TemporalMeanTeacherLitWrapper` usa.

Variantes por par (todas no grid de t+1):
  - `*_mnoc`: IoU/Dice ponderados por M_noc = exp(-alpha*|I_{t+1} - W(I_t)|), o peso de confiança do treino;
  - `*_fb`:   só pixels que passam no teste de consistência forward-backward (Sundaram et al., 2010);
  - `*_identity`: fluxo zero (sobreposição crua de P_t e P_{t+1}), referência sem transporte.

Casos-limite: as duas máscaras vazias -> NaN (o par sai da média e é contado em `both_empty`); só
uma vazia -> 0.

Agregação (`aggregate_video_patient`): média dos pares por vídeo -> média dos vídeos por paciente
(cada vídeo pesa igual) -> pacientes. Assim, um vídeo longo não domina o paciente, e um paciente com
muitos vídeos não domina o conjunto.

Peso de confiança (`confidence_counts`/`auroc_from_counts`): histogramas de (confiança, erro) na faixa
de borda, acumulados por vídeo. A partir deles saem a AUROC de "1 - confiança prevê erro de
transporte" e a curva de confiabilidade, sem guardar mapas por pixel.
'''

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.optical_flow import DEFAULT_NOC_ALPHA, compute_flow, motion_confidence, warp

# Limiares do teste forward-backward de Sundaram, Brox & Keutzer (ECCV 2010), também usados por
# Ruder et al. (2016): |b + f^|^2 < a1 (|b|^2 + |f^|^2) + a2.
FB_ALPHA1 = 0.01
FB_ALPHA2 = 0.5
CONFIDENCE_BINS = 100
BAND_RADIUS = 5      # meia-largura (px) da faixa de borda onde se mede a AUROC (como no notebook de flow)
ROI_DILATE = 15      # px de dilatação da ROI das métricas fotométricas (como no notebook de flow)


# ---------------------------------------------------------------------------- morfologia (GPU)

def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    '''Dilatação binária com elemento quadrado `(2r+1)^2`, `[N,1,H,W]` bool.'''
    if radius <= 0:
        return mask
    return F.max_pool2d(mask.float(), 2 * radius + 1, stride=1, padding=radius) > 0.5


def boundary(mask: torch.Tensor) -> torch.Tensor:
    '''`mask & ~erode(mask)` com 8-vizinhança; fora da imagem conta como fundo (como `binary_erosion`).'''
    background = F.pad(1.0 - mask.float(), (1, 1, 1, 1), value=1.0)
    eroded = F.max_pool2d(background, 3, stride=1) < 0.5
    return mask & ~eroded


# ---------------------------------------------------------------------------- transporte e TC

def transport_masks(masks: torch.Tensor, flow_fw: torch.Tensor, convention: str = "dtc_trus") -> torch.Tensor:
    '''`W(P_t, F_{t->t+1})`: máscaras `[N,1,H,W]` bool de t levadas ao grid de t+1, limiarizadas em 0,5.'''
    return warp(masks.float(), flow_fw, convention) >= 0.5


def overlap_metrics(a: torch.Tensor, b: torch.Tensor, weight: Optional[torch.Tensor] = None,
                    eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    '''
    IoU e Dice por amostra entre máscaras `[N,1,H,W]` bool, opcionalmente ponderados por pixel
    (`weight` `[N,1,H,W]` em [0,1]; uma máscara binária de validade é um caso particular):

        IoU_w = sum w (a & b) / sum w (a | b)        Dice_w = 2 sum w (a & b) / (sum w a + sum w b)

    Denominador (quase) zero -> NaN: as duas vazias (ou todo o desacordo com peso zero). Uma vazia e a
    outra não -> 0.
    '''
    a, b = a.float(), b.float()
    w = torch.ones_like(a) if weight is None else weight.to(a)
    dims = (1, 2, 3)
    inter = (w * a * b).sum(dims)
    union = (w * torch.clamp(a + b, max=1.0)).sum(dims)
    total = (w * a).sum(dims) + (w * b).sum(dims)
    nan = torch.full_like(inter, float("nan"))
    return {
        "iou": torch.where(union > eps, inter / union.clamp_min(eps), nan),
        "dice": torch.where(total > eps, 2.0 * inter / total.clamp_min(eps), nan),
    }


def fb_consistency(flow_fw: torch.Tensor, flow_bw: torch.Tensor,
                   alpha1: float = FB_ALPHA1, alpha2: float = FB_ALPHA2) -> torch.Tensor:
    '''
    Teste forward-backward no grid de t+1 (onde a TC é medida). `flow_bw = F_{t+1->t}` está definido
    nesse grid; `f^ = F_{t->t+1}` amostrado em `x + b(x)`. Um pixel é consistente se ir e voltar quase se
    cancela: `|b + f^|^2 < alpha1 (|b|^2 + |f^|^2) + alpha2`. Pixels inconsistentes são, tipicamente,
    oclusões, saídas do quadro ou fluxo errado. Retorna `[N,1,H,W]` bool.
    '''
    f_hat = warp(flow_fw, flow_bw, "backward")
    lhs = ((flow_bw + f_hat) ** 2).sum(1, keepdim=True)
    rhs = alpha1 * ((flow_bw ** 2).sum(1, keepdim=True) + (f_hat ** 2).sum(1, keepdim=True)) + alpha2
    return lhs < rhs


def pair_confidence(photo_t: torch.Tensor, photo_t1: torch.Tensor, flow_fw: torch.Tensor,
                    flow_bw: torch.Tensor, alpha: float = DEFAULT_NOC_ALPHA,
                    convention: str = "dtc_trus") -> Dict[str, torch.Tensor]:
    '''
    Tudo o que depende só das imagens e do fluxo, no grid de t+1 (fotos `[N,1,H,W]` em [0,1]):
    `I_warp = W(I_t)`, `mnoc` (o peso do treino) e `fb_valid`.
    '''
    warped = warp(photo_t, flow_fw, convention)
    return {
        "I_warp": warped,
        "mnoc": motion_confidence(photo_t1, warped, alpha),
        "fb_valid": fb_consistency(flow_fw, flow_bw),
    }


def tc_pair_metrics(P_t: torch.Tensor, P_t1: torch.Tensor, flow_fw: torch.Tensor,
                    mnoc: torch.Tensor, fb_valid: torch.Tensor,
                    convention: str = "dtc_trus") -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    '''
    Métricas de TC de N pares (`P_t`, `P_t1` `[N,1,H,W]` bool). Devolve (métricas por par,
    `W(P_t)`), e a máscara transportada é reusada na análise do peso de confiança.
    '''
    transported = transport_masks(P_t, flow_fw, convention)
    out: Dict[str, torch.Tensor] = {}
    for suffix, a, weight in (("", transported, None), ("_mnoc", transported, mnoc),
                              ("_fb", transported, fb_valid.float()), ("_identity", P_t, None)):
        for name, value in overlap_metrics(a, P_t1, weight).items():
            out[f"tc_{name}{suffix}"] = value
    out["area_t"] = P_t.float().sum((1, 2, 3))
    out["area_t1"] = P_t1.float().sum((1, 2, 3))
    out["area_warp"] = transported.float().sum((1, 2, 3))
    out["both_empty"] = (out["area_warp"] == 0) & (out["area_t1"] == 0)
    return out, transported


# ---------------------------------------------------------------------------- qualidade do fluxo

def flow_quality(photo_t: torch.Tensor, photo_t1: torch.Tensor, I_warp: torch.Tensor,
                 flow_fw: torch.Tensor, mnoc: torch.Tensor, fb_valid: torch.Tensor,
                 roi: torch.Tensor) -> Dict[str, torch.Tensor]:
    '''
    Erro de reconstrução e estatísticas do fluxo por par, independentes do método:
    `l1_warp = |I_{t+1} - W(I_t)|` contra `l1_identity = |I_{t+1} - I_t|`, na imagem inteira e na ROI
    (região das estruturas segmentadas). `roi` `[N,1,H,W]` bool; ROI vazia -> NaN nas colunas `_roi`.
    '''
    dims = (1, 2, 3)
    err_warp = (photo_t1 - I_warp).abs()
    err_id = (photo_t1 - photo_t).abs()
    magnitude = flow_fw.norm(dim=1, keepdim=True)
    roi_f = roi.float()
    n_roi = roi_f.sum(dims)

    def roi_mean(x):
        return torch.where(n_roi > 0, (x * roi_f).sum(dims) / n_roi.clamp_min(1), torch.full_like(n_roi, float("nan")))

    def roi_quantile(x, q):
        values = []
        for xi, ri in zip(x, roi):
            sel = xi[ri]
            values.append(torch.quantile(sel.float(), q) if sel.numel() else torch.tensor(float("nan"), device=x.device))
        return torch.stack(values)

    return {
        "l1_warp_full": err_warp.mean(dims), "l1_identity_full": err_id.mean(dims),
        "l1_warp_roi": roi_mean(err_warp), "l1_identity_roi": roi_mean(err_id),
        "mnoc_full": mnoc.mean(dims), "mnoc_roi": roi_mean(mnoc),
        "fb_valid_full": fb_valid.float().mean(dims), "fb_valid_roi": roi_mean(fb_valid.float()),
        "flow_mag_mean_roi": roi_mean(magnitude), "flow_mag_p95_roi": roi_quantile(magnitude, 0.95),
        "roi_frac": roi_f.mean(dims),
    }


# ---------------------------------------------------------------------------- peso de confiança

def error_band(target: torch.Tensor, transported: torch.Tensor, radius: int = BAND_RADIUS) -> torch.Tensor:
    '''Faixa de `radius` px em torno das bordas de alvo e transportada, onde o erro de transporte vive.'''
    return dilate(boundary(target) | boundary(transported), radius)


def confidence_counts(confidence: torch.Tensor, error: torch.Tensor, region: torch.Tensor,
                      bins: int = CONFIDENCE_BINS) -> np.ndarray:
    '''
    Histograma `[bins, 2]` (colunas: acerto, erro) da confiança em [0,1], só nos pixels de `region`.
    Somado sobre os N pares. Um mapa binário (FB válido = 1) cai nos bins 0 e `bins-1`.
    '''
    idx = (confidence[region].float() * bins).long().clamp_(0, bins - 1)
    err = error[region].long()
    counts = torch.bincount(idx * 2 + err, minlength=2 * bins)
    return counts.view(bins, 2).cpu().numpy().astype(np.int64)


def auroc_from_counts(counts: np.ndarray) -> float:
    '''
    AUROC de `score = 1 - confiança` para prever erro, a partir de `confidence_counts`:
    P(conf_erro < conf_acerto) + 0,5 P(empate no mesmo bin). Com 100 bins, o empate dentro do bin é a
    única aproximação. Sem erros ou sem acertos -> NaN.
    '''
    counts = np.asarray(counts, dtype=float)
    neg, pos = counts[:, 0], counts[:, 1]
    if neg.sum() == 0 or pos.sum() == 0:
        return float("nan")
    neg_above = np.cumsum(neg[::-1])[::-1] - neg      # acertos com confiança em bin estritamente maior
    return float((pos * neg_above).sum() + 0.5 * (pos * neg).sum()) / (pos.sum() * neg.sum())


def reliability_from_counts(counts: np.ndarray, n_groups: int = 10) -> pd.DataFrame:
    '''Taxa de erro por faixa de confiança (agrupa os bins em `n_groups`), para a curva de confiabilidade.'''
    counts = np.asarray(counts, dtype=float)
    bins = counts.shape[0]
    group = np.arange(bins) * n_groups // bins
    rows = []
    for g in range(n_groups):
        sel = counts[group == g].sum(0)
        total = sel.sum()
        rows.append({"conf_lo": g / n_groups, "conf_hi": (g + 1) / n_groups, "n": total,
                     "error_rate": sel[1] / total if total else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- fluxo no worker (CPU)

def chunk_flows(photos: np.ndarray, flow_kwargs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fluxos de todos os pares consecutivos de `photos` `[n,H,W]` em [0,1]: `F_{t->t+1}` e `F_{t+1->t}`,
    `[n-1,2,H,W]` cada, com `compute_flow` -- o mesmo do dataset de treino.
    '''
    fw = [compute_flow(photos[i], photos[i + 1], **flow_kwargs) for i in range(len(photos) - 1)]
    bw = [compute_flow(photos[i + 1], photos[i], **flow_kwargs) for i in range(len(photos) - 1)]
    shape = (0, 2, *photos.shape[1:])
    return (np.stack(fw) if fw else np.zeros(shape, np.float32),
            np.stack(bw) if bw else np.zeros(shape, np.float32))


class VideoChunks(torch.utils.data.Dataset):
    '''
    Itera TODOS os frames de uma lista de vídeos em blocos de `chunk` pares consecutivos. Cada item traz
    `chunk + 1` frames: o último frame de um bloco é repetido como primeiro do seguinte, para que todo par
    (t, t+1) caia em exatamente um bloco. O worker decodifica cada frame uma vez e calcula os dois fluxos;
    a GPU fica só com o forward e o warp.

    `dataset` é um `VFSSFrameDatasetBase` (teste INCA ou UNM): dele vêm o pré-processamento do modelo
    (`_preprocess_image`) e o redimensionamento (`image_transform`). A foto do fluxo é
    `image_transform(pixel/255)`, a mesma de `VFSSTemporalNeighborsDataset._photo_image`.

    `videos`: lista de dicts com `video_id`, `paciente_id`, `frames_dir` (absoluto) e `n_frames`
    (frames 1..n_frames, contíguos).
    '''

    def __init__(self, dataset, videos: Sequence[Dict], flow_kwargs: Dict, chunk: int = 32):
        self.dataset = dataset
        self.videos = list(videos)
        self.flow_kwargs = dict(flow_kwargs)
        self.items: List[Tuple[int, int, int]] = []
        for index, video in enumerate(self.videos):
            n = int(video["n_frames"])
            for start in range(1, n, chunk):
                self.items.append((index, start, min(start + chunk, n)))

    def __len__(self) -> int:
        return len(self.items)

    def load_frame(self, frames_dir: Path, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''(entrada do modelo `[1,H,W]`, foto `[H,W]` em [0,1]) de um frame.'''
        from PIL import Image
        import torchvision.transforms as T

        raw = T.PILToTensor()(Image.open(Path(frames_dir) / f"{int(frame_id)}.png").convert("L"))
        model_input = self.dataset._preprocess_image(raw)
        photo = self.dataset.image_transform(raw.float() / 255.0).clamp(0.0, 1.0)[0]
        return model_input, photo

    def __getitem__(self, index: int) -> Dict:
        video_index, start, end = self.items[index]
        video = self.videos[video_index]
        frame_ids = list(range(start, end + 1))
        loaded = [self.load_frame(video["frames_dir"], f) for f in frame_ids]
        model = torch.stack([m for m, _ in loaded])
        photos = torch.stack([p for _, p in loaded])
        flow_fw, flow_bw = chunk_flows(photos.numpy(), self.flow_kwargs)
        return {"video_index": video_index, "frame_ids": torch.tensor(frame_ids), "model": model,
                "photo": photos[:, None], "flow_fw": torch.from_numpy(flow_fw),
                "flow_bw": torch.from_numpy(flow_bw)}


# ---------------------------------------------------------------------------- máscaras em disco

def save_packed_masks(path: Path, frame_ids: np.ndarray, masks: np.ndarray) -> None:
    '''Máscaras `[n,H,W]` bool -> npz com bits empacotados (8x menor) + `frame_ids`.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.asarray(masks, dtype=bool)
    np.savez_compressed(path, frame_ids=np.asarray(frame_ids), shape=np.asarray(masks.shape),
                        packed=np.packbits(masks.reshape(len(masks), -1), axis=1))


def load_packed_masks(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    '''Inverso de `save_packed_masks`: `(frame_ids [n], masks [n,H,W] bool)`.'''
    data = np.load(path)
    n, h, w = data["shape"]
    masks = np.unpackbits(data["packed"], axis=1, count=h * w).reshape(n, h, w).astype(bool)
    return data["frame_ids"], masks


# ---------------------------------------------------------------------------- runs

def find_run_dir(runs_root: Path, family: str, run_id: str) -> Optional[Path]:
    '''`<runs_root>/runs/<family>-<run_id>/<timestamp mais recente com best-*.ckpt>`, ou None.'''
    candidates = sorted((Path(runs_root) / "runs" / f"{family}-{run_id}").glob("*/"))
    candidates = [c for c in candidates if list((c / "checkpoints").glob("best-*.ckpt"))]
    return candidates[-1] if candidates else None


def best_checkpoint(run_dir: Path) -> Path:
    return sorted((Path(run_dir) / "checkpoints").glob("best-*.ckpt"))[-1]


def load_run_module(run_dir: Path, device: str = "cpu"):
    '''
    Reconstrói o LightningModule a partir do `config.yaml` do run e carrega o best checkpoint
    (como `Experiment._build_lit_module` + `E2Sweep.evaluate`). Os pesos pré-treinados não são baixados,
    porque o checkpoint os sobrescreve. Num método de duas etapas (DTC-TRUS), se o `teacher_checkpoint` registrado
    não existir mais (run movido), usa o próprio best checkpoint para construir o módulo: só a forma
    importa, porque o `load_state_dict` a seguir traz o professor certo.
    '''
    from omegaconf import OmegaConf
    from src.utils import get_obj_from_str

    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    for key in ("encoder_weights", "pretrain_ckpt"):
        if key in cfg.model.params:
            cfg.model.params[key] = None
    params = dict(cfg.lit_module.params) if cfg.get("lit_module") is not None else {}
    checkpoint = best_checkpoint(run_dir)
    if params.get("teacher_checkpoint") and not Path(str(params["teacher_checkpoint"])).exists():
        params["teacher_checkpoint"] = str(checkpoint)
    cls = get_obj_from_str(cfg.lit_module.target) if cfg.get("lit_module") is not None else None
    if cls is None:
        from src.models.base import LitWrapper as cls  # noqa: N813
    module = cls(model_cfg=cfg.model, optimizer_cfg=cfg.optimizer, lr_scheduler_cfg=cfg.get("lr_scheduler"), **params)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state)
    return cfg, module.to(device).eval(), checkpoint


def eval_forward(module):
    '''O caminho de avaliação do método: `_eval_forward` (teacher nos MT) ou o forward padrão.'''
    return getattr(module, "_eval_forward", None) or module.forward


# ---------------------------------------------------------------------------- agregação e testes

def aggregate_video_patient(pairs: pd.DataFrame, metrics: Sequence[str], by: Sequence[str] = (),
                            video_col: str = "video_id", patient_col: str = "paciente_id",
                            video_stat="mean") -> pd.DataFrame:
    '''
    Pares -> vídeo -> paciente. `video_stat` é a redução dentro do vídeo: "mean" ou um quantil em (0,1)
    (ex. 0.1, o p10, que capta quedas pontuais que a média esconde). Os NaN (pares com as duas máscaras
    vazias) são ignorados. Devolve uma linha por `by + paciente`, com `n_videos` e `n_pairs`.
    '''
    keys = list(by) + [patient_col, video_col]
    grouped = pairs.groupby(keys, observed=True, dropna=False)
    per_video = grouped[list(metrics)].mean() if video_stat == "mean" else grouped[list(metrics)].quantile(float(video_stat))
    per_video["n_pairs"] = grouped.size()
    per_video = per_video.reset_index()
    pkeys = list(by) + [patient_col]
    per_patient = per_video.groupby(pkeys, observed=True, dropna=False)[list(metrics)].mean()
    per_patient["n_videos"] = per_video.groupby(pkeys, observed=True, dropna=False).size()
    per_patient["n_pairs"] = per_video.groupby(pkeys, observed=True, dropna=False).n_pairs.sum()
    return per_patient.reset_index()


def holm(p_values: Iterable[float]) -> np.ndarray:
    '''Correção de Holm-Bonferroni (NaN passa direto e não conta no número de testes).'''
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running, m = 0.0, len(order)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def paired_patient_differences(per_patient: pd.DataFrame, metric: str, comparisons: Sequence[Tuple[str, str]],
                               method_col: str = "method", patient_col: str = "paciente_id",
                               n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    '''
    Para cada `(a, b)`: d_p = metric(a) - metric(b) nos pacientes com os dois valores. Devolve a média, o
    IC95% bootstrap por paciente (percentil), Wilcoxon pareado bilateral e p corrigido por Holm entre as
    comparações.
    '''
    from scipy.stats import wilcoxon

    wide = per_patient.pivot_table(index=patient_col, columns=method_col, values=metric, observed=True)
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        d = (wide[a] - wide[b]).dropna().to_numpy()
        if len(d) == 0:
            continue
        boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        try:
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
        except ValueError:
            p = float("nan")
        rows.append({"metric": metric, "a": a, "b": b, "n_patients": len(d), "mean_diff": d.mean(),
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "frac_a_better": float((d > 0).mean()), "p_wilcoxon": p})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table.p_wilcoxon)
    return table
