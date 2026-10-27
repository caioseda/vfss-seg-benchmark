#!/usr/bin/env python3
'''
Indexa os runs de `logs/` no MLflow e/ou no Aim, SEM copiar artefato nenhum.

E isto que faz os runs ja concluidos aparecerem nos dois trackers sem retreinar nada, e sem
duplicar os ~53 GB de checkpoints: cada run indexado leva params, a serie escalar e uma tag
`run_dir` apontando de volta para a unica copia canonica. Ver o contrato em `src/tracking.py`.

    python scripts/index_runs.py --dry-run                 # o que seria indexado
    python scripts/index_runs.py --backend mlflow
    python scripts/index_runs.py --backend both
    python scripts/index_runs.py --verify

Idempotente. Reexecutar nao duplica: no MLflow a chave `run_uid` e uma TAG consultada antes de
criar (o MLflow nao aceita run_id proprio); no Aim ela e o `run_hash`, mas reusar um hash ACRESCENTA
em vez de substituir, entao um param-sentinela `backfill_complete` decide. As duas mecanicas sao
diferentes de proposito, porque os dois backends sao diferentes.

Runs com `status: running` no `run.json` sao pulados: a serie deles esta incompleta e seria indexada
pela metade. Eles entram na proxima passada, quando terminarem.
'''
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from src.tracking import (MetricPoint, RunRecord, discover_runs, interpolate_timestamps,
                          log_published_metrics, run_time_span,
                          published_metrics_by_run_dir, read_metrics, resolve_tracking_config,
                          run_uid_for, tags_for_run)



def mlflow_uri(logs_root: Path) -> str:
    '''A MESMA URI do treino ao vivo -- uma unica fonte (`src.tracking.resolve_tracking_config`).'''
    return resolve_tracking_config(logs_root)["mlflow"]["tracking_uri"]


def aim_repo(logs_root: Path) -> str:
    return resolve_tracking_config(logs_root)["aim"]["repo"]

# O MLflow trunca chaves em 250 chars e valores de param em 6000; PL ja trunca params em 250.
MAX_PARAM_VALUE = 250
MAX_KEY_LENGTH = 250
MLFLOW_METRICS_PER_BATCH = 900     # limite real 1000; margem para params/tags no mesmo batch


# ---------------------------------------------------------------- metricas publicadas

# ---------------------------------------------------------------- helpers

def _clean_params(params: Dict[str, Any]) -> Dict[str, str]:
    cleaned = {}
    for key, value in params.items():
        key = str(key)[:MAX_KEY_LENGTH]
        if not key or key.startswith("_"):
            continue
        cleaned[key] = str(value)[:MAX_PARAM_VALUE]
    return cleaned


# ---------------------------------------------------------------- MLflow

def index_into_mlflow(records: List[RunRecord], tracking_uri: str,
                      published: Dict[str, Dict[str, float]], verbose: bool = True,
                      replace: bool = False) -> Counter:
    import mlflow
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    counts: Counter = Counter()

    for record in records:
        experiment = client.get_experiment_by_name(record.family)
        if experiment is None:
            # Raiz de artefatos fixa: o store a resolveria contra o CWD de quem cria o experimento.
            experiment_id = client.create_experiment(
                record.family,
                artifact_location=str(record.logs_root / "_tracking" / "mlartifacts"))
        else:
            experiment_id = experiment.experiment_id

        # Idempotencia: o MLflow nao aceita run_id proprio, entao `run_uid` e tag e a checagem e
        # uma busca antes de criar.
        existing = client.search_runs([experiment_id],
                                      filter_string=f"tags.run_uid = '{record.run_uid}'",
                                      max_results=1)
        if existing and replace:
            client.delete_run(existing[0].info.run_id)   # soft-delete: a busca padrao o ignora
            counts["substituido"] += 1
            existing = []
        if existing:
            # Run ja indexado -- inclusive os logados AO VIVO pelo treino. O que ele pode nao ter
            # e o numero publicado, que so nasce no `evaluate()`, depois de os loggers fecharem.
            wanted = published.get(record.relative_dir)
            if wanted and not any(k.startswith("published/") for k in existing[0].data.metrics):
                log_published_metrics(client, existing[0].info.run_id, wanted)
                counts["numero publicado anexado"] += 1
            else:
                counts["ja indexado"] += 1
            continue

        points, _ = read_metrics(record.run_dir)
        # Quando o treino ACONTECEU, nao quando o indexador rodou: sem isto a interface mostra
        # "Created: 5 hours ago" e "Duration: 1.7s" para um treino de horas de semanas atras.
        start_ms, end_ms = run_time_span(record)
        run = client.create_run(experiment_id=experiment_id, tags=tags_for_run(record),
                                run_name=record.run_name, start_time=start_ms)
        for key, value in _clean_params(record.params).items():
            try:
                client.log_param(run.info.run_id, key, value)
            except Exception:
                pass   # param repetido / invalido nao deve custar o run inteiro

        valid_points = [p for p in points if _valid_metric_key(p.key)]
        timestamps = interpolate_timestamps(valid_points, start_ms, end_ms)
        metrics = [Metric(key=p.key[:MAX_KEY_LENGTH],
                          value=float("nan") if p.is_nan else p.value,
                          timestamp=ts, step=p.step)
                   for p, ts in zip(valid_points, timestamps)]
        for name, value in published.get(record.relative_dir, {}).items():
            metrics.append(Metric(key=name[:MAX_KEY_LENGTH], value=value,
                                  timestamp=end_ms or start_ms or 0, step=0))

        for start in range(0, len(metrics), MLFLOW_METRICS_PER_BATCH):
            client.log_batch(run.info.run_id, metrics=metrics[start:start + MLFLOW_METRICS_PER_BATCH])
        client.set_terminated(run.info.run_id,
                              status="FINISHED" if record.status in ("done", "unknown") else "FAILED",
                              end_time=end_ms)
        counts["indexado"] += 1
        if verbose:
            print(f"  [mlflow] {record.relative_dir}  {len(metrics)} pontos")
    return counts


def _valid_metric_key(key: str) -> bool:
    '''O MLflow rejeita chave nao normalizada (`//`, `..`, barra final).'''
    import posixpath
    return bool(key) and posixpath.normpath(key) == key and not key.startswith("/")


# ---------------------------------------------------------------- Aim

def existing_aim_run_uids(repo_path: str) -> set:
    '''
    `run_uid` dos runs que o repo Aim ja contem.

    Diferente do que o nome `run_hash` sugere, o Aim NAO deixa escolher o hash de um run novo:
    `Run(run_hash=...)` ABRE um run existente e levanta `MissingRunError` se ele nao existe. Logo a
    identidade do backfill nao pode ser o hash -- e um param, consultado antes de criar, como no
    MLflow. As duas mecanicas acabam iguais, por um motivo diferente do que eu esperava.
    '''
    try:
        from aim import Repo, Run
        if not Path(repo_path, ".aim").exists():
            return set()
        repo = Repo.from_path(repo_path)
    except Exception:
        return set()

    # `repo.iter_runs()` le o container de INDICE, que so e populado quando o repo e indexado (pelo
    # `aim up`) -- num repo recem-escrito ele devolve zero e a idempotencia falharia em silencio,
    # duplicando tudo a cada passada. `list_all_runs()` le os chunks, que e onde os runs de fato
    # estao.
    uids = set()
    for run_hash in repo.list_all_runs():
        try:
            run = Run(run_hash=run_hash, repo=repo_path, read_only=True)
        except Exception:
            continue
        try:
            uid = run["run_uid"]
        except Exception:
            uid = None
        finally:
            try:
                run.close()
            except Exception:
                pass
        if uid:
            uids.add(str(uid))
    return uids


def assert_aim_persists(repo: str) -> None:
    '''
    Escreve um run descartavel e o le de volta ANTES de indexar 186 runs.

    Existe porque o aim 3.29.1 deste ambiente falha em silencio: cria a "casca" do run (o hash e o
    `name` persistem) mas NAO persiste params nem metricas. `run[key] = value` e um no-op silencioso
    -- `aim/ext/exception_resistant.py` engole a excecao --, `repo.iter_runs()` devolve zero e
    `Run.metrics()` volta vazio depois de reabrir. Sem esta checagem, indexar parece funcionar,
    escreve ~76 MB de chunks por 16 runs e produz um indice vazio; pior, a idempotencia nao vê nada
    e cada passada duplica tudo.

    Se um Aim futuro corrigir isso, esta funcao passa e o caminho volta a funcionar sem mudanca.
    '''
    from aim import Run

    probe = Run(repo=repo, experiment="_probe", system_tracking_interval=None,
                log_system_params=False)
    run_hash = probe.hash
    probe["_probe_key"] = "ok"
    probe.track(1.0, name="_probe_metric", step=0)
    probe.close()

    reopened = Run(run_hash=run_hash, repo=repo, read_only=True)
    persisted_param = reopened.get("_probe_key")
    persisted_metrics = [m.name for m in reopened.metrics()]

    if persisted_param != "ok" or "_probe_metric" not in persisted_metrics:
        raise RuntimeError(
            "O Aim instalado nao persiste o que escrevemos, entao indexar produziria um indice "
            "vazio em silencio.\n"
            f"  repo: {repo}\n"
            f"  param lido de volta: {persisted_param!r} (esperado 'ok')\n"
            f"  metricas lidas de volta: {persisted_metrics} (esperado conter '_probe_metric')\n"
            "Diagnostico: no aim 3.29.1 `run[chave] = valor` e um no-op silencioso, porque\n"
            "`aim/ext/exception_resistant.py` engole a excecao; `repo.iter_runs()` devolve zero e\n"
            "`Run.metrics()` volta vazio apos reabrir. O ultimo release do Aim e de 2025-05-08 e a\n"
            "questao de manutencao (aimhubio/aim#3414) segue sem resposta.\n"
            "Use --backend mlflow. O TensorBoard segue lendo `logs/<familia>/runs` direto, e nenhum\n"
            "resultado depende do Aim."
        )


def index_into_aim(records: List[RunRecord], repo: str,
                   published: Dict[str, Dict[str, float]], verbose: bool = True) -> Counter:
    from aim import Run

    assert_aim_persists(repo)
    already = existing_aim_run_uids(repo)
    counts: Counter = Counter()
    for record in records:
        if record.run_uid in already:
            counts["ja indexado"] += 1
            continue

        run = Run(repo=repo, experiment=record.family,
                  system_tracking_interval=None, log_system_params=False)
        try:
            run.name = record.run_name
            for key, value in tags_for_run(record).items():
                run[key] = value
            for key, value in _clean_params(record.params).items():
                run[f"params/{key}"] = value

            points, _ = read_metrics(record.run_dir)
            tracked = 0
            for point in points:
                if point.is_nan:
                    # O Aim nao tem equivalente da coluna `is_nan` do MLflow: um NaN entraria como
                    # um ponto indistinguivel de um valor real. Preferivel omitir e registrar a
                    # contagem do que gravar um numero que nao existe.
                    continue
                run.track(point.value, name=point.key, step=point.step)
                tracked += 1
            for name, value in published.get(record.relative_dir, {}).items():
                run.track(value, name=name, step=0)

            run["backfill_complete"] = True
            run["nan_points_omitted"] = len(points) - tracked
        finally:
            run.close()
        counts["indexado"] += 1
        if verbose:
            print(f"  [aim] {record.relative_dir}  {tracked} pontos "
                  f"({len(points) - tracked} NaN omitidos)")
    return counts


# ---------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default `mlflow`, nao `both`: o Aim deste ambiente nao persiste o que escrevemos (ver
    # `assert_aim_persists`), entao `--backend aim` falha de proposito, com o diagnostico.
    parser.add_argument("--backend", choices=["mlflow", "aim", "both"], default="mlflow")
    parser.add_argument("--logs", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra o que seria indexado, sem escrever em tracker nenhum")
    parser.add_argument("--verify", action="store_true",
                        help="reconcilia a contagem de pontos entre run dir e trackers")
    parser.add_argument("--replace", action="store_true",
                        help="apaga (soft-delete) e reindexa os runs ja existentes do escopo -- para "
                             "reparar um run cujo tracker falhou durante o treino")
    parser.add_argument("--include-running", action="store_true",
                        help="indexa tambem runs em andamento (serie incompleta)")
    parser.add_argument("--family", default=None, help="indexa so uma familia")
    parser.add_argument("--skip-archive", action="store_true",
                        help="nao indexa logs/_archive (indice mais enxuto; os runs seguem em disco "
                             "e legiveis pelo TensorBoard)")
    args = parser.parse_args()

    logs_root = Path(args.logs).resolve()
    records = discover_runs(logs_root)
    if args.family:
        records = [r for r in records if r.family == args.family]
    if args.skip_archive:
        records = [r for r in records if r.family != "_archive"]

    skipped_running = [r for r in records if r.status == "running" and not args.include_running]
    records = [r for r in records if r not in skipped_running]

    print(f"logs: {logs_root}")
    print(f"runs descobertos: {len(records) + len(skipped_running)}")
    for family, count in Counter(r.family for r in records).most_common():
        print(f"  {count:>4}  logs/{family}")
    if skipped_running:
        print(f"pulados por estarem em andamento: {len(skipped_running)}")
        for record in skipped_running:
            print(f"     {record.relative_dir}")

    published = published_metrics_by_run_dir(logs_root)
    print(f"runs com numero publicado nas CSVs de resultado: {len(published)}")

    if args.verify:
        return _verify(records, logs_root, args)

    if args.dry_run:
        total = 0
        for record in records:
            points, source = read_metrics(record.run_dir)
            total += len(points)
        print(f"\npontos que seriam indexados: {total:,}")
        print(f"layouts: {dict(Counter(r.layout for r in records))}")
        print("\n--dry-run: nada foi escrito.")
        return 0

    mlflow_target = mlflow_uri(logs_root)
    aim_target = aim_repo(logs_root)
    (logs_root / "_tracking").mkdir(parents=True, exist_ok=True)

    if args.backend in ("mlflow", "both"):
        print(f"\nMLflow -> {mlflow_target}")
        print(dict(index_into_mlflow(records, mlflow_target, published, replace=args.replace)))
    if args.backend in ("aim", "both"):
        print(f"\nAim -> {aim_target}")
        print(dict(index_into_aim(records, aim_target, published)))
    return 0


def _verify(records: List[RunRecord], logs_root: Path, args) -> int:
    '''Tres stores sem verificador divergem. Compara a contagem de pontos de cada um.'''
    on_disk = {r.run_uid: len(read_metrics(r.run_dir)[0]) for r in records}
    print(f"\nno disco: {len(on_disk)} runs, {sum(on_disk.values()):,} pontos")
    problems = 0

    if args.backend in ("mlflow", "both"):
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=mlflow_uri(logs_root))
            indexed = set()
            for experiment in client.search_experiments():
                for run in client.search_runs([experiment.experiment_id], max_results=50000):
                    uid = run.data.tags.get("run_uid")
                    if uid:
                        indexed.add(uid)
            missing = set(on_disk) - indexed
            print(f"MLflow: {len(indexed)} runs indexados, {len(missing)} ausentes")
            problems += len(missing)
        except Exception as error:
            print(f"MLflow: nao foi possivel verificar ({error})")
            problems += 1

    if args.backend in ("aim", "both"):
        try:
            indexed = existing_aim_run_uids(aim_repo(logs_root))
            missing = set(on_disk) - indexed
            print(f"Aim: {len(indexed)} runs indexados, {len(missing)} ausentes")
            problems += len(missing)
        except Exception as error:
            print(f"Aim: nao foi possivel verificar ({error})")
            problems += 1

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
