#!/usr/bin/env python3
'''
Congela os pesos SSL escolhidos pelo tuning do E2 (docs/e2-tuning-ssl.md): o vencedor da confirmacao de
cada metodo do espaco de busca -> configs/frozen/e2-ssl-hparams.yaml, lido por e2-tuned.yaml e
e2-cv-tuned.yaml. Recusa se algum metodo (inclusive o MT temporal da fase 2) nao estiver confirmado.

    python scripts/freeze_e2_ssl_hparams.py --dry-run    # mostra os vencedores, nao grava
    python scripts/freeze_e2_ssl_hparams.py              # grava (uma vez; --force so de proposito)

Commitar o arquivo E o congelamento: depois dele nenhum peso pode ser reajustado com resultados do E2.
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from omegaconf import OmegaConf

from src.sweep import _resolve_sweep_config
from src.sweeps.e2_tuning import E2TuningConfirmSweep, freeze_ssl_hparams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sweep/e2-ssl-tuning-confirm.yaml")
    parser.add_argument("--out", default="configs/frozen/e2-ssl-hparams.yaml")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve() if args.log_dir else None
    plan = E2TuningConfirmSweep(OmegaConf.load(str(_resolve_sweep_config(args.config))), log_dir=log_dir)
    axes = plan._base_axes()
    group = {key: values[0] for key, values in axes.items() if key != "method"}
    for method in plan.method_spaces():
        winner = plan.winner({**group, "method": method})
        print(f"{method:>18}: trial {winner['trial']:2d}  val_dice_patient={winner['val_dice_patient']:.4f}  "
              f"{winner['params']}")
    if args.dry_run:
        return 0
    path = freeze_ssl_hparams(args.config, args.out, log_dir=log_dir, force=args.force)
    print(f"\ncongelado em {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
