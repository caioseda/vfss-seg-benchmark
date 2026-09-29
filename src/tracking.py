'''
Os trackers como INDICE sobre o diretorio de run, nunca como copia dele.

O contrato, porque e o que justifica o desenho:

    O diretorio de run e a unica copia canonica. `logs/<familia>/runs/<run>/<uid>/` guarda
    `config.yaml`, `run.json`, `console.log`, `tensorboard/`, `csv/metrics.csv` e `checkpoints/`.
    O TensorBoard le essa arvore direto. MLflow e Aim recebem params, a serie escalar e uma
    tag/param `run_dir` apontando de volta -- e NENHUM artefato: `log_model=False` e nada de
    `log_artifact`.

Ser honesto sobre o limite: um tracker E um indice, e um indice guarda a serie escalar. Depois
disto cada escalar existe em quatro lugares (tfevents, `metrics.csv`, `mlflow.db`, `.aim/`). O que
o desenho garante e que nada PESADO e duplicado -- os ~53 GB de checkpoints, os event files e os
`.npz` de previsao ficam exatamente uma vez, e os tres trackers apontam para la. O indice custa
algumas centenas de MB contra 54 GB.

`DeviceStatsMonitor.*` e excluido da indexacao de proposito: sao ~15% dos pontos do acervo, com as
chaves mais longas, e nao respondem nenhuma pergunta de pesquisa. Ficam no TensorBoard.

Layouts. O acervo tem quatro, herdados de eras diferentes do projeto, e a descoberta fareja em vez
de assumir um template de caminho:
    <run>/csv/metrics.csv + <run>/tensorboard/    (moderno, 156 runs)
    <run>/tensorboard/ apenas                     (23 runs, quase todos so com hp_metric)
    <run>/metrics.csv + <run>/events.*            (legado, 5 runs)
    <run>/events.* apenas                         (2 runs)
Os runs de benchmark ficam um nivel mais fundo
(`_archive/benchmarks/<nome>/<NNN-ts>/<NN-cfg>/`), e e por isso que a busca e recursiva: um glob
`logs/*/runs/*/*/` perderia ~100 runs arquivados.
'''
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Partes de um run, nao runs. Sem isto a descoberta conta `<run>/csv` e `<run>/tensorboard` como
# runs proprios, porque cada um contem um `metrics.csv` ou um `events.*`.
COMPONENT_DIR_NAMES = {"csv", "tensorboard", "checkpoints"}
# `predictions/` guarda caches .npz por run_id, nao runs. `_migration`/`_tracking` sao nossos.
EXCLUDED_PARTS = {"_migration", "_tracking", "predictions", "_sweep"}

DEVICE_STATS_PREFIX = "DeviceStatsMonitor"
# Chaves que sao eixo, nao metrica -- o step vai no campo `step` de cada ponto.
NON_METRIC_COLUMNS = {"epoch", "step"}


@dataclass
class MetricPoint:
    key: str
    value: float
    step: int
    is_nan: bool = False
    wall_time: Optional[float] = None


@dataclass
class RunRecord:
    run_dir: Path
    logs_root: Path
    run_uid: str
    layout: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def relative_dir(self) -> str:
        return str(self.run_dir.relative_to(self.logs_root))

    @property
    def family(self) -> str:
        '''Primeiro nivel sob `logs/` -- a unidade pela qual os experimentos sao comparados.'''
        return self.run_dir.relative_to(self.logs_root).parts[0]

    @property
    def run_name(self) -> str:
        '''
        Nome do run. No layout `<familia>/runs/<nome>/<uid>` e o avo do uid; nos layouts antigos,
        o pai. `run.json` tem a ultima palavra quando existe.
        '''
        recorded = self.metadata.get("run_name")
        if recorded:
            return str(recorded)
        parts = self.run_dir.relative_to(self.logs_root).parts
        if len(parts) >= 3 and parts[1] == "runs":
            return parts[2]
        return parts[-2] if len(parts) >= 2 else parts[-1]

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "unknown"))


# ---------------------------------------------------------------- descoberta

def run_uid_for(run_dir: Path, logs_root: Path) -> str:
    '''
    Identidade estavel de um run: sha1 do caminho RELATIVO a `logs/`.

    Relativo, e nao absoluto, para sobreviver a mover `logs/` ou trocar o mount. E nao um hash do
    conteudo: um run retomado ou re-testado mudaria de identidade, e a reindexacao criaria um
    duplicado em vez de reconhecer o mesmo run.
    '''
    return hashlib.sha1(str(run_dir.relative_to(logs_root)).encode("utf-8")).hexdigest()


def _metric_sources(run_dir: Path) -> List[str]:
    sources = []
    if (run_dir / "csv" / "metrics.csv").is_file():
        sources.append("csv/metrics.csv")
    if any(run_dir.glob("tensorboard/events.out.tfevents.*")):
        sources.append("tensorboard/")
    if run_dir.name not in COMPONENT_DIR_NAMES:
        if (run_dir / "metrics.csv").is_file():
            sources.append("metrics.csv")
        if any(run_dir.glob("events.out.tfevents.*")):
            sources.append("events")
    return sources


def discover_runs(logs_root: Path) -> List[RunRecord]:
    '''Todo diretorio sob `logs/` que carrega alguma fonte de metrica, em qualquer dos 4 layouts.'''
    logs_root = Path(logs_root).resolve()
    records: List[RunRecord] = []
    for candidate in sorted(logs_root.rglob("*")):
        if not candidate.is_dir():
            continue
        relative_parts = candidate.relative_to(logs_root).parts
        if set(relative_parts) & EXCLUDED_PARTS or candidate.name in COMPONENT_DIR_NAMES:
            continue
        sources = _metric_sources(candidate)
        if not sources:
            continue
        record = RunRecord(
            run_dir=candidate,
            logs_root=logs_root,
            run_uid=run_uid_for(candidate, logs_root),
            layout=" + ".join(sources),
        )
        record.metadata = read_run_metadata(candidate)
        record.params = read_params(candidate)
        records.append(record)
    return records


# ---------------------------------------------------------------- leitura

def read_run_metadata(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((run_dir / "run.json").read_text())
    except Exception:
        return {}


def _flatten(config: Any, prefix: str = "") -> Dict[str, Any]:
    '''Mesmo achatamento de `Experiment._flatten_config`, que e o que gerou os hparams.yaml.'''
    flat: Dict[str, Any] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            flat.update(_flatten(value, f"{prefix}{key}/"))
    elif isinstance(config, (list, tuple)):
        flat[prefix.rstrip("/")] = str(list(config))
    else:
        flat[prefix.rstrip("/")] = config
    return flat


def read_params(run_dir: Path) -> Dict[str, Any]:
    '''
    Hiperparametros do run. Prioriza o `config.yaml` resolvido (que `Experiment` grava e que e
    re-executavel) e cai para o `hparams.yaml` achatado dos runs antigos.
    '''
    for path, needs_flatten in (
        (run_dir / "config.yaml", True),
        (run_dir / "csv" / "hparams.yaml", False),
        (run_dir / "tensorboard" / "hparams.yaml", False),
        (run_dir / "hparams.yaml", False),
    ):
        if not path.is_file():
            continue
        try:
            loaded = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(loaded, dict):
            continue
        return _flatten(loaded) if needs_flatten else loaded
    return {}


def _read_metrics_csv(path: Path) -> List[MetricPoint]:
    '''
    Le `metrics.csv` com o modulo csv, NAO com pandas, para preservar uma distincao que importa
    aqui: uma celula vazia significa "nunca logado nesta linha", enquanto a string `nan` significa
    um NaN de verdade. O pandas colapsa as duas em NaN, e o artefato de NaN em ASSD/HD95 e conhecido
    neste projeto -- confundir as duas corromperia o registro.
    '''
    points: List[MetricPoint] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = row.get("step")
            try:
                step_value = int(float(step)) if step not in (None, "") else 0
            except (TypeError, ValueError):
                step_value = 0
            for key, raw in row.items():
                if key in NON_METRIC_COLUMNS or key is None or raw in (None, ""):
                    continue
                if key.startswith(DEVICE_STATS_PREFIX):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                is_nan = math.isnan(value)
                points.append(MetricPoint(key=key, value=0.0 if is_nan else value,
                                          step=step_value, is_nan=is_nan))
    return points


def _read_event_files(run_dir: Path) -> List[MetricPoint]:
    '''
    Fallback para os runs que so tem event files.

    Usa `tensorboard.backend.event_processing` -- que ja esta instalado -- e NAO o
    `aim convert tensorboard`, que exige `import tensorflow` e, sem ele, falha em silencio.
    '''
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    points: List[MetricPoint] = []
    for event_file in sorted(run_dir.rglob("events.out.tfevents.*")):
        try:
            loader = EventFileLoader(str(event_file))
        except Exception:
            continue
        for event in loader.Load():
            for value in getattr(event.summary, "value", []):
                if value.tag.startswith(DEVICE_STATS_PREFIX) or value.tag in NON_METRIC_COLUMNS:
                    continue
                scalar: Optional[float] = None
                if value.HasField("simple_value"):
                    scalar = float(value.simple_value)
                elif value.HasField("tensor"):
                    try:
                        array = tensor_util.make_ndarray(value.tensor)
                        if array.size == 1:
                            scalar = float(array.reshape(-1)[0])
                    except Exception:
                        scalar = None
                if scalar is None:
                    continue
                is_nan = math.isnan(scalar)
                points.append(MetricPoint(key=value.tag, value=0.0 if is_nan else scalar,
                                          step=int(event.step), is_nan=is_nan,
                                          wall_time=float(event.wall_time)))
    return points


def read_metrics(run_dir: Path) -> Tuple[List[MetricPoint], str]:
    '''
    Serie escalar do run, com a fonte usada. Prioridade `csv/metrics.csv` -> `metrics.csv` plano ->
    event files: a CSV e mais barata de ler e ja esta em forma tabular; os event files existem para
    os runs antigos que nunca tiveram CSVLogger.
    '''
    for path, label in ((run_dir / "csv" / "metrics.csv", "csv/metrics.csv"),
                        (run_dir / "metrics.csv", "metrics.csv")):
        if path.is_file():
            points = _read_metrics_csv(path)
            if points:
                return points, label
    return _read_event_files(run_dir), "events"


# ---------------------------------------------------------------- loggers ao vivo

def build_loggers(run_dir: Path, family: Optional[str], run_name: str, run_uid: str,
                  logging_cfg: Optional[Any] = None) -> List[Any]:
    '''
    Loggers de um run ao vivo: TensorBoard + CSV sempre, MLflow e Aim quando habilitados.

    Nenhum dos dois recebe artefatos -- so params, escalares e o caminho do run dir. Ver o contrato
    no topo deste arquivo.
    '''
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from src.utils import instantiate_from_config

    save_dir = str(run_dir)
    loggers: List[Any] = [
        TensorBoardLogger(save_dir=save_dir, name="", version="tensorboard"),
        CSVLogger(save_dir=save_dir, name="", version="csv"),
    ]

    config = logging_cfg or {}
    tags = {"run_dir": str(run_dir), "run_uid": run_uid, "family": family or "", "run_name": run_name}

    if config.get("mlflow", {}).get("enabled"):
        loggers.append(_build_mlflow_logger(config["mlflow"], run_name, tags))
    if config.get("aim", {}).get("enabled"):
        loggers.append(_build_aim_logger(config["aim"], run_name, tags))

    for logger_cfg in (config.get("loggers") or []):
        loggers.append(instantiate_from_config(logger_cfg))
    return loggers


def _build_mlflow_logger(config: Any, run_name: str, tags: Dict[str, str]):
    from pytorch_lightning.loggers import MLFlowLogger
    return MLFlowLogger(
        experiment_name=str(config.get("experiment_name") or tags["family"] or "default"),
        run_name=run_name,
        tracking_uri=str(config["tracking_uri"]),
        tags=tags,
        # Sem artefatos: os checkpoints ficam no run dir, e a tag `run_dir` aponta para eles.
        log_model=False,
    )


def _build_aim_logger(config: Any, run_name: str, tags: Dict[str, str]):
    from aim.pytorch_lightning import AimLogger
    logger = AimLogger(
        repo=str(config["repo"]),
        experiment=str(config.get("experiment") or tags["family"] or "default"),
        # O default e 10 s, e injetaria ~10 mil pontos de `__system__` num run de 2 h -- mais que as
        # metricas reais do run, e ausentes nos runs retro-indexados, o que tornaria os dois
        # conjuntos incomparaveis.
        system_tracking_interval=None,
        log_system_params=False,
    )
    for key, value in tags.items():
        logger.experiment.set(key, value, strict=False)
    return logger
