'''
Invalida celulas JA CONCLUIDAS de um sweep para que sejam retreinadas, sem apagar nada e sem tocar nas
outras celulas da familia.

Uso tipico: um bug que afetou SO um metodo (ex.: o FixMatch do E2, 2026-11-17). As demais celulas seguem
valendo; as do metodo afetado precisam sair das fontes de verdade do sweep -- senao `run_sweep.py` as
pula (`Sweep.completed_run_ids`) e um retreino retomaria o `last.ckpt` velho (`run_uid` reaproveitado).

Para cada `run_id` alvo, numa raiz de logs (`<logs>/<familia>/`):

  - linhas das CSVs `results_*.csv` da familia e de cada shard (`_sweep/<tag>/`) -> removidas e GUARDADAS
    no arquivo, no mesmo caminho relativo;
  - entradas dos `state.json` (canonico e de cada shard) -> removidas e guardadas em `state_removido.json`;
  - `runs/<run_name>/` (todas as tentativas, com checkpoints) e `_sweep/<tag>/configs/<run_id>.yaml`
    -> MOVIDOS para o arquivo.

O arquivo e `<arquivo>/<familia>-<rotulo>/`, com um `MANIFESTO.json` (motivo, data, run_ids, o que saiu de
onde). Rodar de novo e idempotente: o que ja saiu nao e encontrado.

SO biblioteca padrao (Python >= 3.8), de proposito: o mesmo arquivo roda nos hosts remotos fora do
container, via `scripts/remote.py invalidate` (que o envia por `ssh ... python3 -`), onde o
`sync` traria de volta os runs e os `state.json` de shard se eles continuassem la.
'''
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

csv.field_size_limit(sys.maxsize)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _state_paths(family_dir: Path) -> List[Path]:
    sweep_root = family_dir / "_sweep"
    paths = [sweep_root / "state.json"]
    if sweep_root.is_dir():
        paths += sorted(path / "state.json" for path in sweep_root.iterdir() if path.is_dir())
    return [path for path in paths if path.exists()]


def _results_paths(family_dir: Path) -> List[Path]:
    paths = sorted(family_dir.glob("results_*.csv"))
    sweep_root = family_dir / "_sweep"
    if sweep_root.is_dir():
        for shard in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
            paths += sorted(shard.glob("results_*.csv"))
    return paths


def _split_csv(path: Path, run_ids: set):
    '''(fieldnames, linhas mantidas, linhas removidas) -- o texto de cada celula e preservado como esta.'''
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "run_id" not in fieldnames:
            return fieldnames, None, []
        kept, removed = [], []
        for row in reader:
            (removed if row["run_id"] in run_ids else kept).append(row)
    return fieldnames, kept, removed


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = append and path.exists()
    tmp = path.with_name(path.name + ".tmp")
    if exists:
        shutil.copyfile(path, tmp)
    with tmp.open("a" if exists else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Mesmo destino de uma invalidacao anterior: guarda ao lado, nunca sobrescreve.
        target = target.with_name(f"{target.name}.{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(source), str(target))


def invalidate(logs_root, family: str, run_ids: Iterable[str], archive_dir,
               run_names: Optional[Dict[str, str]] = None, reason: str = "",
               dry_run: bool = False, force: bool = False) -> dict:
    '''
    Args:
        logs_root: raiz dos logs (`<repo>/logs` na delaunay, `<root>/logs` num host remoto).
        family: nome da familia (`e2`, `e2-cv`).
        run_ids: celulas a invalidar (`SweepPlan.run_id`).
        archive_dir: destino (`.../<familia>-<rotulo>`); criado se preciso.
        run_names: `{run_id: run_name}` (o nome do diretorio em `runs/`); na falta, o `run_name` gravado no
            state e, por ultimo, `<familia>-<run_id>`.
        dry_run: so relata.
        force: invalida mesmo uma celula com status `running` (normalmente um treino em andamento).

    Returns:
        relatorio `{"run_ids", "csv_rows", "state_cells", "moved", "running", "archive_dir", "dry_run"}`.
    '''
    logs_root = Path(logs_root)
    family_dir = logs_root / family
    archive_dir = Path(archive_dir)
    run_ids = sorted(set(run_ids))
    targets = set(run_ids)
    names = dict(run_names or {})
    report = {"family": family, "run_ids": run_ids, "csv_rows": {}, "state_cells": {}, "moved": [],
              "running": [], "archive_dir": str(archive_dir), "dry_run": bool(dry_run)}
    if not family_dir.is_dir():
        report["missing_family_dir"] = str(family_dir)
        return report

    # 1. state.json: descobre run_name e recusa celulas em andamento antes de mexer em qualquer coisa.
    states = {path: _read_json(path) for path in _state_paths(family_dir)}
    for path, state in states.items():
        for run_id in targets & set(state.get("cells", {})):
            entry = state["cells"][run_id]
            names.setdefault(run_id, entry.get("run_name") or f"{family}-{run_id}")
            if entry.get("status") == "running":
                report["running"].append(f"{path.relative_to(logs_root)}: {run_id}")
    for run_id in run_ids:
        names.setdefault(run_id, f"{family}-{run_id}")
    if report["running"] and not force:
        report["refused"] = ("ha celulas com status `running` (treino em andamento ou interrompido); "
                             "confirme que nenhum sweep desta familia esta rodando e use --force")
        return report

    stamp = datetime.now().isoformat(timespec="seconds")
    removed_states: Dict[str, dict] = {}
    for path, state in states.items():
        cells = state.get("cells", {})
        hit = {run_id: cells[run_id] for run_id in targets & set(cells)}
        if not hit:
            continue
        relative = str(path.relative_to(logs_root))
        report["state_cells"][relative] = sorted(hit)
        removed_states[relative] = hit
        if not dry_run:
            for run_id in hit:
                del cells[run_id]
            state["updated_at"] = datetime.now().isoformat()
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(state, indent=2, default=str))
            tmp.replace(path)

    # 2. CSVs de resultado (canonicas e de shard).
    for path in _results_paths(family_dir):
        fieldnames, kept, removed = _split_csv(path, targets)
        if kept is None or not removed:
            continue
        relative = path.relative_to(logs_root)
        report["csv_rows"][str(relative)] = len(removed)
        if not dry_run:
            _write_csv(archive_dir / relative, fieldnames, removed, append=True)
            _write_csv(path, fieldnames, kept)

    # 3. Diretorios de run e configs geradas pelos shards.
    moves = []
    for run_id in run_ids:
        run_dir = family_dir / "runs" / names[run_id]
        if run_dir.is_dir():
            moves.append(run_dir)
    sweep_root = family_dir / "_sweep"
    if sweep_root.is_dir():
        for shard in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
            moves += [shard / "configs" / f"{run_id}.yaml" for run_id in run_ids
                      if (shard / "configs" / f"{run_id}.yaml").exists()]
    for source in moves:
        relative = source.relative_to(logs_root)
        report["moved"].append(str(relative))
        if not dry_run:
            _move(source, archive_dir / relative)

    if not dry_run and (removed_states or report["csv_rows"] or report["moved"]):
        archive_dir.mkdir(parents=True, exist_ok=True)
        state_file = archive_dir / "state_removido.json"
        previous = _read_json(state_file)
        for relative, cells in removed_states.items():
            previous.setdefault(relative, {}).update(cells)
        state_file.write_text(json.dumps(previous, indent=2, default=str))
        manifest_file = archive_dir / "MANIFESTO.json"
        manifest = _read_json(manifest_file) or {"invalidacoes": []}
        manifest["invalidacoes"].append({
            "quando": stamp, "motivo": reason, "familia": family, "logs_root": str(logs_root),
            "run_ids": run_ids, "csv_rows": report["csv_rows"], "state_cells": report["state_cells"],
            "moved": report["moved"]})
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return report


def format_report(report: dict) -> str:
    lines = [f"familia {report['family']}: {len(report['run_ids'])} run_id(s)"
             + ("  [dry-run: nada mudou]" if report.get("dry_run") else "")]
    if report.get("missing_family_dir"):
        lines.append(f"  (sem diretorio da familia: {report['missing_family_dir']})")
    if report.get("refused"):
        lines.append(f"  RECUSADO: {report['refused']}")
        lines += [f"    running: {item}" for item in report["running"]]
        return "\n".join(lines)
    for path, n in report["csv_rows"].items():
        lines.append(f"  csv   {path}: {n} linha(s)")
    for path, ids in report["state_cells"].items():
        lines.append(f"  state {path}: {len(ids)} celula(s)")
    for path in report["moved"]:
        lines.append(f"  move  {path}")
    if not (report["csv_rows"] or report["state_cells"] or report["moved"]):
        lines.append("  nada encontrado (ja invalidado?)")
    else:
        lines.append(f"  arquivo: {report['archive_dir']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    '''Entrada usada pelo `remote.py invalidate`: um JSON com os argumentos de `invalidate` em argv[0].'''
    argv = sys.argv[1:] if argv is None else argv
    kwargs = json.loads(argv[0])
    report = invalidate(**kwargs)
    print(format_report(report))
    print("REPORT " + json.dumps(report))
    return 2 if report.get("refused") else 0


if __name__ == "__main__":
    sys.exit(main())
