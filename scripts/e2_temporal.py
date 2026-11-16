#!/usr/bin/env python
'''
Consistência temporal (TC-IoU/TC-Dice) dos métodos do E2 em TODOS os frames dos vídeos de teste.

    python scripts/e2_temporal.py                                   # runs de logs/e2, INCA + UNM
    python scripts/e2_temporal.py --eval-sets inca_test             # só o teste INCA
    python scripts/e2_temporal.py --runs-root logs/_arquivo/e2-sem-aug --out /tmp/tc   # outro conjunto de runs
    python scripts/e2_temporal.py --max-videos 2 --max-frames-per-video 30 --out <scratch>   # fumaça

Uma passada por vídeo: o worker decodifica os frames e calcula o Farneback (o mesmo do treino; os
parâmetros saem do config.yaml de um run `mt-temporal-*`), e a GPU faz o forward de todos os runs e o
warp. Saídas em `--out` (padrão `<runs-root>/temporal`):

    pairs.parquet              um par (t, t+1) x run: TC-IoU/Dice (+ _mnoc, _fb, _identity), áreas
    flow_quality.parquet       um par: erro de reconstrução warp x identity, M_noc, FB, magnitude
    confidence_counts.parquet  histogramas (confiança, erro) na faixa de borda, por vídeo x camada x tipo
    gt_pairs.parquet           pares consecutivos ANOTADOS: transporte da GT (warp x sem warp)
    masks/<eval_set>/<run_id>/v<video>.npz   máscaras empacotadas (só --save-masks-for; padrão inca_test)
    settings.json              o que define o cálculo; um --out com outro settings é recusado

Retomável: cada vídeo concluído vira `parts/<eval_set>/v<video>.*.parquet` e é pulado na próxima vez.
Veja src/temporal_consistency.py para a definição da métrica.
'''
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.base import DataModuleFromConfig  # noqa: E402
from src.data.optical_flow import DEFAULT_NOC_ALPHA  # noqa: E402
from src.sweep import Sweep  # noqa: E402
from src.temporal_consistency import (  # noqa: E402
    BAND_RADIUS, CONFIDENCE_BINS, FB_ALPHA1, FB_ALPHA2, ROI_DILATE, VideoChunks, best_checkpoint,
    confidence_counts, dilate, error_band, eval_forward, find_run_dir, flow_quality, load_run_module,
    overlap_metrics, pair_confidence, save_packed_masks, tc_pair_metrics, transport_masks,
)
from src.utils import instantiate_from_config  # noqa: E402

EVAL_SETS = ("inca_test", "unm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="configs/sweep/e2.yaml")
    p.add_argument("--runs-root", default=None, help="padrão: logs/<family do sweep>")
    p.add_argument("--out", default=None, help="padrão: <runs-root>/temporal")
    p.add_argument("--eval-sets", nargs="+", default=list(EVAL_SETS), choices=EVAL_SETS)
    p.add_argument("--methods", nargs="+", default=None, help="padrão: todos os do eixo `method` com run concluído")
    p.add_argument("--chunk", type=int, default=32, help="pares por item do DataLoader")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-frames-per-video", type=int, default=None, help="válvula de custo/fumaça")
    p.add_argument("--max-videos", type=int, default=None, help="só os N primeiros vídeos de cada conjunto (fumaça)")
    p.add_argument("--save-masks-for", nargs="*", default=["inca_test"])
    p.add_argument("--overwrite", action="store_true", help="apaga um --out com settings diferentes")
    return p.parse_args()


# ---------------------------------------------------------------------------- runs e settings

def discover_runs(sweep: Sweep, runs_root: Path, methods):
    runs = {}
    for cell in sweep.plan.cells():
        if methods and str(cell["method"]) not in methods:
            continue
        run_id = sweep.plan.run_id(cell)
        run_dir = find_run_dir(runs_root, sweep.plan.family, run_id)
        if run_dir is None:
            print(f"[aviso] sem run concluído para {run_id} em {runs_root}/runs -- fica fora")
            continue
        runs[run_id] = {"cell": {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in cell.items()},
                        "run_dir": run_dir}
    if not runs:
        raise SystemExit(f"Nenhum run encontrado em {runs_root}/runs.")
    return runs


def training_flow_settings(runs):
    '''Parâmetros do fluxo do dataset de treino de um run temporal (o "mesmo flow do treino").'''
    from omegaconf import OmegaConf

    for run in runs.values():
        if str(run["cell"]["method"]).startswith("mt-temporal"):
            train = OmegaConf.load(run["run_dir"] / "config.yaml").data.params.train.params
            source = str(run["run_dir"])
            break
    else:
        train, source = {}, "padrões de VFSSTemporalNeighborsDataset (nenhum run mt-temporal-*)"
    get = (lambda k, d: train.get(k, d)) if train else (lambda k, d: d)
    alpha = get("noc_alpha", None)
    farneback_params = get("farneback_params", None)
    return {
        "flow_kwargs": {"method": str(get("flow_method", "farneback")),
                        "preprocess": str(get("flow_preprocess", "vfss")),
                        "farneback_params": dict(farneback_params) if farneback_params else None,
                        "median_ksize": int(get("farneback_median_ksize", 3))},
        "warp_convention": str(get("warp_convention", "dtc_trus")),
        "noc_alpha": float(DEFAULT_NOC_ALPHA if alpha is None else alpha),
        "source": source,
    }


def check_settings(out: Path, settings: dict, overwrite: bool) -> None:
    path = out / "settings.json"
    key = {k: v for k, v in settings.items() if k != "eval_sets"}   # dá para rodar um conjunto de cada vez
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != "eval_sets"} != key:
            if not overwrite:
                raise SystemExit(f"{path} foi gerado com outros settings; use outro --out ou --overwrite.")
            import shutil
            shutil.rmtree(out)
        else:
            settings = {**settings, "eval_sets": sorted(set(previous.get("eval_sets", [])) | set(settings["eval_sets"]))}
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, default=str))


# ---------------------------------------------------------------------------- dados

def build_dataset(eval_set: str, runs, sweep: Sweep):
    from omegaconf import OmegaConf

    if eval_set == "inca_test":
        first = next(iter(runs.values()))
        data_cfg = OmegaConf.load(first["run_dir"] / "config.yaml").data
        if not data_cfg.params.shared_dataset_params.get("holdout_test", False):
            raise SystemExit("O config do run não tem holdout_test=true: o split 'test' não é o teste INCA.")
        data_cfg.params.cache_splits = []
        data_module = instantiate_from_config(data_cfg)
        assert isinstance(data_module, DataModuleFromConfig)
        return instantiate_from_config(data_module.dataset_configs["test"])
    return instantiate_from_config(sweep.config.evaluation.unm)


def list_videos(dataset, max_videos=None, max_frames=None):
    df = dataset.video_frame_df
    videos = []
    for video_id, group in df.groupby("video_id", sort=True):
        frames_dir = Path(dataset.dataset_path) / os.path.dirname(group.iloc[0].image_path)
        ids = sorted(int(f[:-4]) for f in os.listdir(frames_dir) if f.endswith(".png") and f[:-4].isdigit())
        if ids != list(range(1, len(ids) + 1)):
            raise RuntimeError(f"Frames não contíguos (1..N) em {frames_dir}")
        n = len(ids) if max_frames is None else min(len(ids), int(max_frames))
        videos.append({"video_id": int(video_id), "paciente_id": str(group.iloc[0].paciente_id),
                       "frames_dir": str(frames_dir), "n_frames": n,
                       "labeled": sorted(int(f) for f in group.frame_id)})
    return videos[:max_videos] if max_videos else videos


def gt_masks(dataset, video):
    '''GT `[1,H,W]` bool dos frames anotados do vídeo que têm o seguinte também anotado (pares consecutivos).'''
    labeled = set(video["labeled"])
    needed = sorted({f for f in labeled if f + 1 in labeled and f + 1 <= video["n_frames"]} |
                    {f + 1 for f in labeled if f + 1 in labeled and f + 1 <= video["n_frames"]})
    return {f: (dataset.__getitem__(video_id=video["video_id"], frame_id=f)["segmentation"] > 0)[None]
            for f in needed}


# ---------------------------------------------------------------------------- uma passada por vídeo

class VideoAccumulator:
    def __init__(self, video, eval_set):
        self.video, self.eval_set = video, eval_set
        self.pairs, self.flow, self.gt = [], [], []
        self.counts = defaultdict(lambda: np.zeros((CONFIDENCE_BINS, 2), np.int64))
        self.masks = defaultdict(dict)     # run_id -> frame_id -> [H,W] bool

    def base(self, frame_t):
        return {"eval_set": self.eval_set, "video_id": self.video["video_id"],
                "paciente_id": self.video["paciente_id"], "frame_t": int(frame_t)}

    def frames(self, columns: dict, frame_ids, extra=None):
        table = pd.DataFrame({k: v.detach().cpu().numpy() for k, v in columns.items()})
        table.insert(0, "frame_t", np.asarray(frame_ids[:-1]))
        for k, v in {**{"eval_set": self.eval_set, "video_id": self.video["video_id"],
                        "paciente_id": self.video["paciente_id"]}, **(extra or {})}.items():
            table.insert(0, k, v)
        return table

    def write(self, parts_dir: Path, masks_dir):
        stem = parts_dir / f"v{self.video['video_id']}"
        pd.concat(self.pairs, ignore_index=True).to_parquet(f"{stem}.pairs.parquet")
        pd.concat(self.flow, ignore_index=True).to_parquet(f"{stem}.flow.parquet")
        (pd.DataFrame(self.gt) if self.gt else pd.DataFrame(columns=["eval_set"])).to_parquet(f"{stem}.gt.parquet")
        rows = []
        for (layer, conf_type), counts in self.counts.items():
            for b in np.flatnonzero(counts.sum(1)):
                rows.append({"eval_set": self.eval_set, "video_id": self.video["video_id"],
                             "paciente_id": self.video["paciente_id"], "layer": layer,
                             "confidence": conf_type, "bin": int(b),
                             "n_ok": int(counts[b, 0]), "n_err": int(counts[b, 1])})
        pd.DataFrame(rows).to_parquet(f"{stem}.conf.parquet")
        if masks_dir is not None:
            for run_id, frames in self.masks.items():
                ids = sorted(frames)
                save_packed_masks(Path(masks_dir) / run_id / f"v{self.video['video_id']}.npz",
                                  np.asarray(ids), np.stack([frames[i] for i in ids]))
        Path(f"{stem}.done").touch()


@torch.no_grad()
def process_item(item, acc: VideoAccumulator, modules, repeat_channels, settings, device, gts, keep_masks):
    convention, alpha = settings["warp_convention"], settings["noc_alpha"]
    frame_ids = item["frame_ids"].tolist()
    x = item["model"].to(device)
    if repeat_channels:
        x = x.repeat(1, 3, 1, 1)
    photo = item["photo"].to(device)
    fw, bw = item["flow_fw"].to(device), item["flow_bw"].to(device)
    conf = pair_confidence(photo[:-1], photo[1:], fw, bw, alpha, convention)
    mnoc, fb_valid = conf["mnoc"], conf["fb_valid"]

    union = torch.zeros(len(frame_ids) - 1, 1, *photo.shape[-2:], dtype=torch.bool, device=device)
    for run_id, (module, method) in modules.items():
        masks = (eval_forward(module)(x).argmax(1, keepdim=True) == 1)
        metrics, transported = tc_pair_metrics(masks[:-1], masks[1:], fw, mnoc, fb_valid, convention)
        acc.pairs.append(acc.frames(metrics, frame_ids, {"method": method, "run_id": run_id}))
        error = transported != masks[1:]
        band = error_band(masks[1:], transported, BAND_RADIUS)
        acc.counts[(run_id, "mnoc")] += confidence_counts(mnoc, error, band)
        acc.counts[(run_id, "fb")] += confidence_counts(fb_valid.float(), error, band)
        union |= masks[:-1] | masks[1:]
        if keep_masks:
            for f, m in zip(frame_ids, masks[:, 0].cpu().numpy()):
                acc.masks[run_id][f] = m

    quality = flow_quality(photo[:-1], photo[1:], conf["I_warp"], fw, mnoc, fb_valid, dilate(union, ROI_DILATE))
    acc.flow.append(acc.frames(quality, frame_ids))

    # Camada GT: pares (t, t+1) anotados -- o transporte do fluxo contra a anotação.
    for i, t in enumerate(frame_ids[:-1]):
        if t in gts and t + 1 in gts:
            G_t, G_t1 = gts[t][None].to(device), gts[t + 1][None].to(device)
            warped = transport_masks(G_t, fw[i:i + 1], convention)
            row = acc.base(t)
            for name, value in overlap_metrics(warped, G_t1).items():
                row[f"gt_{name}_warp"] = value.item()
            for name, value in overlap_metrics(G_t, G_t1).items():
                row[f"gt_{name}_identity"] = value.item()
            row.update({"mnoc_roi": quality["mnoc_roi"][i].item(), "fb_valid_roi": quality["fb_valid_roi"][i].item()})
            acc.gt.append(row)
            error, band = warped != G_t1, error_band(G_t1, warped, BAND_RADIUS)
            acc.counts[("gt", "mnoc")] += confidence_counts(mnoc[i:i + 1], error, band)
            acc.counts[("gt", "fb")] += confidence_counts(fb_valid[i:i + 1].float(), error, band)


def run_eval_set(eval_set, args, sweep, runs, modules, settings, out):
    dataset = build_dataset(eval_set, runs, sweep)
    videos = list_videos(dataset, args.max_videos, args.max_frames_per_video)
    parts_dir = out / "parts" / eval_set
    parts_dir.mkdir(parents=True, exist_ok=True)
    pending = [v for v in videos if not (parts_dir / f"v{v['video_id']}.done").exists()]
    n_pairs = sum(v["n_frames"] - 1 for v in pending)
    print(f"\n[{eval_set}] {len(videos)} vídeos, {dataset.video_frame_df.paciente_id.nunique()} pacientes; "
          f"pendentes {len(pending)} vídeos / {n_pairs} pares", flush=True)
    if not pending:
        return
    masks_dir = out / "masks" / eval_set if eval_set in (args.save_masks_for or []) else None
    chunks = VideoChunks(dataset, pending, settings["flow_kwargs"], chunk=args.chunk)
    loader = torch.utils.data.DataLoader(chunks, batch_size=None, shuffle=False, num_workers=args.workers,
                                         prefetch_factor=2 if args.workers else None)
    acc, gts, start, done_pairs = None, {}, time.perf_counter(), 0
    for item in loader:
        video = pending[int(item["video_index"])]
        if acc is None or acc.video is not video:
            if acc is not None:
                acc.write(parts_dir, masks_dir)
            acc, gts = VideoAccumulator(video, eval_set), gt_masks(dataset, video)
        process_item(item, acc, modules, dataset.repeat_channels, settings, args.device, gts,
                     keep_masks=masks_dir is not None)
        done_pairs += len(item["frame_ids"]) - 1
        if item["frame_ids"][-1].item() == video["n_frames"]:
            elapsed = time.perf_counter() - start
            print(f"  v{video['video_id']:>4} ({video['n_frames']} frames)  {done_pairs}/{n_pairs} pares  "
                  f"{done_pairs / max(elapsed, 1e-6):.0f} pares/s", flush=True)
    if acc is not None:
        acc.write(parts_dir, masks_dir)


def consolidate(out: Path, eval_sets):
    for kind, name in (("pairs", "pairs"), ("flow", "flow_quality"), ("gt", "gt_pairs"), ("conf", "confidence_counts")):
        files = [f for s in eval_sets for f in sorted((out / "parts" / s).glob(f"v*.{kind}.parquet"))]
        tables = [pd.read_parquet(f) for f in files]
        tables = [t for t in tables if len(t)]
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        table.to_parquet(out / f"{name}.parquet")
        print(f"{out / f'{name}.parquet'}: {len(table)} linhas")


def main():
    args = parse_args()
    sweep = Sweep(args.sweep)
    runs_root = Path(args.runs_root) if args.runs_root else sweep.family_dir
    runs_root = runs_root if runs_root.is_absolute() else ROOT / runs_root
    out = Path(args.out) if args.out else runs_root / "temporal"
    out = out if out.is_absolute() else ROOT / out

    runs = discover_runs(sweep, runs_root, args.methods)
    flow = training_flow_settings(runs)
    settings = {
        "runs": {run_id: str(best_checkpoint(r["run_dir"])) for run_id, r in runs.items()},
        "flow_kwargs": flow["flow_kwargs"], "warp_convention": flow["warp_convention"],
        "noc_alpha": flow["noc_alpha"], "flow_settings_from": flow["source"],
        "fb_alpha": [FB_ALPHA1, FB_ALPHA2], "confidence_bins": CONFIDENCE_BINS, "band_radius": BAND_RADIUS,
        "roi_dilate": ROI_DILATE, "max_frames_per_video": args.max_frames_per_video,
        "max_videos": args.max_videos, "eval_sets": list(args.eval_sets),
    }
    check_settings(out, settings, args.overwrite)
    print(f"Runs ({len(runs)}): {', '.join(runs)}\nFluxo: {flow['flow_kwargs']} | {flow['warp_convention']} "
          f"| alpha={flow['noc_alpha']:.3f} (de {flow['source']})\nSaída: {out}", flush=True)

    modules = {}
    for run_id, run in runs.items():
        _, module, _ = load_run_module(run["run_dir"], args.device)
        modules[run_id] = (module, str(run["cell"]["method"]))
    for eval_set in args.eval_sets:
        run_eval_set(eval_set, args, sweep, runs, modules, settings, out)
    consolidate(out, json.loads((out / "settings.json").read_text())["eval_sets"])


if __name__ == "__main__":
    main()
