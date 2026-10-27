#!/usr/bin/env python3
'''
Gera e congela o split do protocolo do artigo: 70% desenvolvimento (5 folds) / 10% validacao fixa /
20% teste INCA, em CONTAGEM DE PACIENTES, estratificado multi-video x video unico. A regra esta em
`src/data/splits.py:make_protocol_split`.

    python scripts/make_protocol_split.py
    python scripts/make_protocol_split.py --seed 42 --out configs/splits/inca-protocolo-s42.csv

Escreve o CSV (uma linha por paciente) e um JSON irmao com seed, ratios, contagens por particao e fold,
sha256 da tabela de frames de origem e git sha. Os dois vao para o git: o split e congelado, e por isso
o script se recusa a sobrescrever um arquivo existente sem `--force`.
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
import json
import subprocess
from datetime import datetime

from src.data.splits import PROTOCOL_RATIOS, make_protocol_split, summarize_protocol_split
from src.data.vfss_frame_dataset import load_video_frame_dataframe

DEFAULT_DATASET = "/data_ssd/caioseda/data/dataset_inca_all_frames/"
DEFAULT_TABLE = "inca-video-frame-dataset.csv"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE, help="CSV de frames anotados dentro do dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="CSV de saida (padrao: configs/splits/inca-protocolo-s<seed>.csv)")
    parser.add_argument("--force", action="store_true", help="sobrescreve um split ja congelado")
    args = parser.parse_args()

    out = Path(args.out) if args.out else PROJECT_ROOT / "configs" / "splits" / f"inca-protocolo-s{args.seed}.csv"
    meta_path = out.with_suffix(".json")
    if (out.exists() or meta_path.exists()) and not args.force:
        print(f"{out} ja existe -- o split e congelado. Use --force so se a mudanca for intencional.")
        return 1

    source = Path(args.dataset_path) / args.table
    frames = load_video_frame_dataframe(args.dataset_path, args.table)
    table = make_protocol_split(frames, seed=args.seed, ratios=PROTOCOL_RATIOS, n_folds=args.n_folds)
    summary = summarize_protocol_split(table)

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    meta = {
        "seed": args.seed,
        "ratios_dev_val_test": list(PROTOCOL_RATIOS),
        "ratios_unit": "patients",
        "n_folds": args.n_folds,
        "stratified_by": "n_videos > 1 (multi_video) vs 1 (single_video)",
        "source_table": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_sha": git_sha(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": json.loads(summary.astype(object).where(summary.notna(), None).to_json(orient="records")),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(summary.to_string(index=False))
    print(f"\n{out}\n{meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
