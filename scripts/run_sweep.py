#!/usr/bin/env python3
'''
Roda uma grade de experimentos em primeiro plano. E este processo que o tmux mantem vivo.

    python scripts/run_sweep.py --config configs/sweep/x1.yaml --dry-run
    python scripts/run_sweep.py --config configs/sweep/x1.yaml

Para que sobreviva a queda do SSH, use `scripts/submit.sh`, que o lanca numa sessao tmux destacada.
Rodar de novo apos uma queda e seguro e e o fluxo esperado: celulas concluidas sao puladas e a
celula interrompida retoma de `last.ckpt`.
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.sweep import Sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML do sweep (configs/sweep/*.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra a grade e o estado de cada celula, sem treinar")
    parser.add_argument("--log-dir", default=None,
                        help="raiz dos logs (padrao: <repo>/logs)")
    parser.add_argument("--force-resume", action="store_true",
                        help="retoma mesmo se a configuracao mudou desde a tentativa anterior")
    parser.add_argument("--fast-dev-run", nargs="?", const=True, default=False, type=int,
                        help="fast_dev_run do Lightning; nao grava resultados")
    args = parser.parse_args()

    sweep = Sweep(args.config, log_dir=args.log_dir)
    sweep.run(fast_dev_run=args.fast_dev_run, force_resume=args.force_resume, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
