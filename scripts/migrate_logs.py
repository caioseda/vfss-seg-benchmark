#!/usr/bin/env python3
'''
Reorganiza `logs/` de uma arvore plana de 85 entradas para `logs/<familia>/runs/<run>/<uid>/`.

POR QUE
    Hoje `tensorboard --logdir logs/` mistura todos os experimentos de todos os meses. Pior:
    duas grades diferentes -- `x1-frames-per-video` (4 classes, C2/C4) e
    `x1-frames-per-video-multiclass-all` (8 classes) -- usam o MESMO template de `name`, e por
    isso gravam runs no MESMO diretorio, distinguiveis apenas pelo timestamp. O TensorBoard
    sobrepoe duas curvas incomparaveis, e `find_checkpoint()` pode devolver o checkpoint errado.

O QUE NAO SE MOVE (os notebooks dependem destes caminhos):
    logs/<familia>/results_*.csv     o registro cientifico
    logs/<familia>/predictions/*.npz cache de previsoes, indexado por run_id
    o aninhamento <run_name>/<uid>   o notebook do ACDC le `run_dir.parent.name`

Apenas os diretorios de run descem um nivel, para dentro de `<familia>/runs/`.

INVARIANTE
    `logs/<familia>/runs/` contem EXATAMENTE os runs que a CSV de resultado daquela familia
    referencia -- nada mais. E o que faz `tensorboard --logdir logs/x1/runs` mostrar as 16
    celulas da grade publicada, e nao tambem as tentativas anteriores e os orcamentos de um
    desenho que o proprio notebook declara nao comparavel.

    A atribuicao e portanto feita pela coluna `output_dir` das CSVs, por TIMESTAMP -- o nome do
    diretorio, sozinho, e ambiguo (e a colisao acima). O que sobra vai para `_archive/`, sem
    adivinhacao por prefixo:
        _archive/incomplete/  treinou parcialmente, sem checkpoints/
        _archive/unclaimed/   treinou inteiro, mas nenhuma CSV o referencia
    Nada e apagado, e o manifesto reverte tudo.

USO
    python scripts/migrate_logs.py                                      # dry-run (padrao)
    python scripts/migrate_logs.py --apply
    python scripts/migrate_logs.py --verify
    python scripts/migrate_logs.py --revert logs/_migration/<ts>/manifest.json
'''
import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS = PROJECT_ROOT / "logs"

ARCHIVE = "_archive"
RESERVED = {ARCHIVE, "_tracking", "_migration"}


def find_families() -> dict:
    '''Familias = diretorios de nivel 1 que contem uma CSV de resultado.'''
    return {
        d.name: sorted(d.glob("results_*.csv"))
        for d in sorted(LOGS.iterdir())
        if d.is_dir() and not d.name.startswith("_") and list(d.glob("results_*.csv"))
    }


def claims_from_csvs(families: dict) -> tuple[dict, list]:
    '''
    Mapeia run_dir absoluto -> familia, lendo `output_dir` de cada CSV de resultado.
    Devolve tambem os avisos, para que um `output_dir` fora de `logs/` nunca passe calado.
    '''
    claims, warnings = {}, []
    for family, csv_paths in families.items():
        for csv_path in csv_paths:
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "output_dir" not in rows[0]:
                warnings.append(f"{csv_path.relative_to(PROJECT_ROOT)}: sem coluna `output_dir`")
                continue
            for output_dir in {r["output_dir"] for r in rows if r.get("output_dir")}:
                path = Path(output_dir)
                if not path.is_absolute():
                    path = (PROJECT_ROOT / path).resolve()
                try:
                    path.relative_to(LOGS)
                except ValueError:
                    warnings.append(f"{csv_path.name}: `output_dir` fora de logs/: {output_dir}")
                    continue
                previous = claims.get(path)
                if previous and previous != family:
                    warnings.append(f"CONFLITO: {path} reivindicado por `{previous}` e `{family}`")
                claims[path] = family
    return claims, warnings


def plan_moves() -> tuple[list, list]:
    '''Devolve (movimentos, avisos). Cada movimento e um dict {src, dst, reason}.'''
    families = find_families()
    claims, warnings = claims_from_csvs(families)
    moves = []

    def add(src: Path, dst: Path, reason: str) -> None:
        moves.append({"src": str(src), "dst": str(dst), "reason": reason})

    # (A) benchmarks/ -- outro nivel de aninhamento, e uma era encerrada do projeto.
    if (LOGS / "benchmarks").is_dir():
        add(LOGS / "benchmarks", LOGS / ARCHIVE / "benchmarks", "benchmark antigo")

    for entry in sorted(LOGS.iterdir()):
        if not entry.is_dir() or entry.name in RESERVED or entry.name == "benchmarks":
            continue
        if entry.name in families:      # diretorio de familia: fica onde esta
            continue

        # (B) a era das arquiteturas (vfss-*): arquiva inteiro, de uma vez.
        if entry.name.startswith("vfss"):
            add(entry, LOGS / ARCHIVE / entry.name, "era das arquiteturas")
            continue

        # (C) diretorio de run: cada filho e um run_uid.
        for uid_dir in sorted(p for p in entry.iterdir() if p.is_dir()):
            family = claims.get(uid_dir.resolve())
            if family:
                add(uid_dir, LOGS / family / "runs" / entry.name / uid_dir.name, f"CSV de `{family}`")
            elif not (uid_dir / "checkpoints").is_dir():
                add(uid_dir, LOGS / ARCHIVE / "incomplete" / entry.name / uid_dir.name,
                    "incompleto: sem checkpoints/")
            else:
                # Treinou ate o fim, mas nenhuma CSV de resultado o referencia: ou e um re-run
                # superado por uma tentativa posterior, ou e de um desenho abandonado (os
                # orcamentos 10%/20% do X1, que o proprio notebook declara nao comparaveis).
                # Preservado, fora das familias -- ver o invariante no topo do arquivo.
                add(uid_dir, LOGS / ARCHIVE / "unclaimed" / entry.name / uid_dir.name,
                    "nao reivindicado por nenhuma CSV")
    return moves, warnings


def rewrite_csvs(moves: list, apply: bool) -> list:
    '''
    Reescreve a coluna `output_dir` das CSVs de resultado pelo mapa literal do manifesto.

    Mapa literal, e nao reconstrucao a partir do run_id, de proposito: assim a inconsistencia
    conhecida do ACDC (`run_id=supervised-repo-10pct-s1337` apontando para
    `acdc-supervised-10pct-s1337`) e corrigida de graca, em vez de virar um caso especial.
    '''
    remap = {m["src"]: m["dst"] for m in moves}
    report = []
    for family_dir in sorted(LOGS.iterdir()):
        if not family_dir.is_dir():
            continue
        for csv_path in sorted(family_dir.glob("results_*.csv")):
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows = list(reader)
            if not fieldnames or "output_dir" not in fieldnames:
                continue
            changed = 0
            for row in rows:
                new = remap.get(row.get("output_dir"))
                if new:
                    row["output_dir"] = new
                    changed += 1
            report.append((csv_path, changed, len(rows)))
            if apply and changed:
                shutil.copy2(csv_path, csv_path.with_suffix(csv_path.suffix + ".bak"))
                with csv_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
    return report


def verify() -> int:
    '''Afirma que todo `output_dir` das CSVs resolve e tem um checkpoint. Codigo de saida.'''
    problems = 0
    for family_dir in sorted(LOGS.iterdir()):
        if not family_dir.is_dir():
            continue
        for csv_path in sorted(family_dir.glob("results_*.csv")):
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            dirs = {r["output_dir"] for r in rows if r.get("output_dir")}
            if not dirs:
                continue
            missing = [d for d in sorted(dirs) if not Path(d).is_dir()]
            no_ckpt = [d for d in sorted(dirs)
                       if Path(d).is_dir() and not list(Path(d).glob("checkpoints/*.ckpt"))]
            status = "ok" if not (missing or no_ckpt) else "FALHA"
            print(f"[{status}] {csv_path.relative_to(LOGS)}: {len(dirs)} run_dir")
            for d in missing:
                print(f"         inexistente: {d}")
            for d in no_ckpt:
                print(f"         sem checkpoint: {d}")
            problems += len(missing) + len(no_ckpt)
    return 1 if problems else 0


def execute(moves: list, manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps({"created_at": datetime.now().isoformat(), "moves": moves}, indent=2))
    print(f"manifesto: {manifest.relative_to(PROJECT_ROOT)}")

    for move in moves:
        src, dst = Path(move["src"]), Path(move["dst"])
        if not src.exists():
            print(f"  [pula] ja movido: {src.relative_to(LOGS)}")
            continue
        if dst.exists():
            sys.exit(f"ABORTADO: destino ja existe: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)          # mesmo filesystem: instantaneo, nao copia bytes
    print(f"  {len(moves)} movimentos aplicados")

    # Diretorios de run que ficaram vazios apos os filhos sairem.
    for entry in sorted(LOGS.iterdir()):
        if entry.is_dir() and entry.name not in RESERVED and not any(entry.iterdir()):
            entry.rmdir()
            print(f"  [rmdir] {entry.relative_to(LOGS)} (vazio)")


def revert(manifest_path: Path) -> None:
    moves = json.loads(Path(manifest_path).read_text())["moves"]
    for move in reversed(moves):
        src, dst = Path(move["src"]), Path(move["dst"])
        if not dst.exists():
            print(f"  [pula] destino ausente: {dst}")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        os.rename(dst, src)
    print(f"  {len(moves)} movimentos revertidos")
    print("  ATENCAO: as CSVs nao sao revertidas aqui -- restaure os arquivos .bak a mao.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="executa (o padrao e apenas mostrar o plano)")
    parser.add_argument("--verify", action="store_true", help="checa que as CSVs apontam para runs validos")
    parser.add_argument("--revert", metavar="MANIFESTO", help="desfaz uma migracao")
    args = parser.parse_args()

    if args.revert:
        revert(Path(args.revert))
        return 0
    if args.verify:
        return verify()

    moves, warnings = plan_moves()
    by_reason: dict[str, int] = {}
    for move in moves:
        by_reason[move["reason"]] = by_reason.get(move["reason"], 0) + 1

    print(f"{'APLICANDO' if args.apply else 'DRY-RUN'} -- {len(moves)} movimentos\n")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count:>4}  {reason}")
    print()
    for move in moves:
        print(f"  {Path(move['src']).relative_to(LOGS)}")
        print(f"    -> {Path(move['dst']).relative_to(LOGS)}")

    report = rewrite_csvs(moves, apply=False)
    print("\nCSVs de resultado (coluna `output_dir`):")
    for csv_path, changed, total in report:
        print(f"  {csv_path.relative_to(LOGS)}: {changed} linhas reescritas de {total}")

    if warnings:
        print(f"\nAVISOS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if not args.apply:
        print("\nNada foi alterado. Re-execute com --apply.")
        return 0

    execute(moves, LOGS / "_migration" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    rewrite_csvs(moves, apply=True)
    print("\nVerificacao:")
    return verify()


if __name__ == "__main__":
    sys.exit(main())
