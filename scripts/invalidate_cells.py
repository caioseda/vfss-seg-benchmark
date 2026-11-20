#!/usr/bin/env python3
'''
Invalida celulas concluidas de um sweep (na delaunay) para retreina-las, ARQUIVANDO -- nunca apagando -- os
runs, as linhas das CSVs e as entradas de state.json. As outras celulas da familia nao sao tocadas.

    # 1. ver o que sairia (nada muda)
    python scripts/invalidate_cells.py --config configs/sweep/e2.yaml --set "axes.method=[fixmatch]" \
        --label fixmatch-bn-dreno --reason "..." --dry-run
    # 2. aplicar; depois o mesmo nos hosts remotos que rodaram a familia (o sync traria tudo de volta):
    python scripts/remote.py invalidate dl05 configs/sweep/e2.yaml --set "axes.method=[fixmatch]" \
        --label fixmatch-bn-dreno --reason "..."
    # 3. retreinar SO essas celulas
    python scripts/remote.py submit dl05 configs/sweep/e2.yaml --set "axes.method=[fixmatch]"

Destino: logs/_arquivo/<familia>-<rotulo>/ (mesmos caminhos relativos a logs/, + MANIFESTO.json e
state_removido.json), registrado em logs/_arquivo/LEIA-ME.md. Logica: src/sweep_invalidate.py.
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import date

from src.sweep import archive_label, invalidation_targets
from src.sweep_invalidate import format_report, invalidate

ARCHIVE_ROOT_NAME = "_arquivo"


def append_readme(archive_root: Path, label: str, report: dict, reason: str) -> None:
    readme = archive_root / "LEIA-ME.md"
    text = readme.read_text() if readme.exists() else "# Arquivo\n"
    if f"`{label}/`" in text:
        return
    entry = (f"\n- `{label}/`: {len(report['run_ids'])} celula(s) de `{report['family']}` invalidadas em "
             f"{date.today().isoformat()} por `scripts/invalidate_cells.py` (retreino). {reason}\n"
             f"  Detalhes (o que saiu de onde): `{label}/MANIFESTO.json`.\n")
    readme.write_text(text.rstrip("\n") + "\n" + entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML do sweep (configs/sweep/*.yaml)")
    parser.add_argument("--set", nargs="+", required=True, metavar="CHAVE=VALOR",
                        help="recorte da grade, em dotlist (ex.: \"axes.method=[fixmatch]\")")
    parser.add_argument("--label", required=True, help="rotulo do arquivo (ex.: fixmatch-bn-dreno)")
    parser.add_argument("--reason", default="", help="motivo, gravado no MANIFESTO e no LEIA-ME")
    parser.add_argument("--log-dir", default=None, help="raiz dos logs (padrao: <repo>/logs)")
    parser.add_argument("--dry-run", action="store_true", help="so mostra o que sairia")
    parser.add_argument("--force", action="store_true", help="invalida mesmo celulas com status `running`")
    args = parser.parse_args()

    targets = invalidation_targets(args.config, args.set)
    logs_root = Path(args.log_dir).resolve() if args.log_dir else PROJECT_ROOT / "logs"
    label = archive_label(targets["family"], args.label)
    archive_root = logs_root / ARCHIVE_ROOT_NAME
    print(f"celulas: {', '.join(targets['run_ids'])}")
    report = invalidate(logs_root, targets["family"], targets["run_ids"], archive_root / label,
                        run_names=targets["run_names"], reason=args.reason,
                        dry_run=args.dry_run, force=args.force)
    print(format_report(report))
    if report.get("refused"):
        return 2
    if not args.dry_run and (report["csv_rows"] or report["state_cells"] or report["moved"]):
        append_readme(archive_root, label, report, args.reason)
        print("\nproximos passos:\n"
              "  - o mesmo em cada host remoto que rodou esta familia: python scripts/remote.py invalidate <host> ...\n"
              "  - tabelas/figuras derivadas (ex.: comparison_table.*) seguem velhas ate o notebook rodar de novo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
