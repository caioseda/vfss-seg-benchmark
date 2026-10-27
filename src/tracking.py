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
import os
import re
from datetime import datetime
from collections import defaultdict
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


# ---------------------------------------------------------------- quando o run aconteceu

# Os diretorios de run antigos sao nomeados pelo instante em que o treino comecou
# (`2026-09-14-17-46-29`), inclusive os de benchmark (`006-2026-03-02-18-53-26/01-vfss-unet`).
_DIR_TIMESTAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
_END_MARKERS = ("csv/metrics.csv", "metrics.csv", "console.log")


def run_time_span(record: RunRecord) -> Tuple[Optional[int], Optional[int]]:
    '''
    (inicio, fim) do treino, em milissegundos epoch -- ou None quando nao ha como saber.

    Existe porque o backfill cria o run no MLflow NO MOMENTO em que indexa e o encerra em seguida;
    sem isto a interface mostra "Created: 5 hours ago" e "Duration: 1.7s" para um treino que durou
    horas e aconteceu semanas antes, e o eixo de tempo dos graficos cai em 1970.

    Fontes, da mais para a menos confiavel:
      inicio  `run.json:started_at` -> timestamp no nome do diretorio -> None
      fim     `run.json:finished_at` -> mtime do ultimo arquivo de metrica/log -> None
    '''
    metadata = record.metadata

    def from_iso(value: Any) -> Optional[int]:
        try:
            return int(datetime.fromisoformat(str(value)).timestamp() * 1000)
        except Exception:
            return None

    start = from_iso(metadata.get("started_at")) if metadata.get("started_at") else None
    if start is None:
        # O ULTIMO componente do caminho que parece um timestamp: num run de benchmark o relevante
        # esta no pai (`NNN-<ts>`), e num run normal e o proprio diretorio.
        for part in reversed(record.run_dir.relative_to(record.logs_root).parts):
            match = _DIR_TIMESTAMP.search(part)
            if match:
                try:
                    start = int(datetime(*(int(g) for g in match.groups())).timestamp() * 1000)
                except ValueError:
                    start = None
                break

    end = from_iso(metadata.get("finished_at")) if metadata.get("finished_at") else None
    if end is None:
        mtimes = [(record.run_dir / name).stat().st_mtime for name in _END_MARKERS
                  if (record.run_dir / name).is_file()]
        mtimes += [event.stat().st_mtime for event in record.run_dir.rglob("events.out.tfevents.*")]
        if mtimes:
            end = int(max(mtimes) * 1000)

    if start is not None and end is not None and end < start:
        end = start      # relogio torto / arquivo copiado: duracao zero e melhor que negativa
    return start, end


def interpolate_timestamps(points: List[MetricPoint], start: Optional[int],
                           end: Optional[int]) -> List[int]:
    '''
    Timestamp (ms) de cada ponto, distribuindo os passos linearmente entre inicio e fim.

    O `metrics.csv` nao guarda quando cada linha foi escrita, so o passo. Sem timestamp o MLflow
    grava 0 e o grafico com eixo de tempo ("Time (Wall)") coloca tudo em 1970. A interpolacao e uma
    aproximacao -- os passos nao tem custo uniforme (validacao e teste pesam mais) --, entao serve
    para ordenar e dar a escala certa, nao para medir a duracao de um passo. Pontos que ja trazem
    `wall_time` (vindos de event files) mantem o valor real.
    '''
    if start is None:
        return [int((p.wall_time or 0) * 1000) for p in points]
    end = end if end is not None else start
    max_step = max((p.step for p in points), default=0) or 1
    return [int(p.wall_time * 1000) if p.wall_time
            else int(start + (end - start) * (p.step / max_step)) for p in points]


# ---------------------------------------------------------------- tags e numero publicado
#
# Vivem aqui, e nao em `scripts/index_runs.py`, porque o treino ao vivo e o backfill TEM de produzir
# a mesma identidade e as mesmas tags: um run logado ao vivo cujo `run_uid` diferisse do que o
# indexador calcula seria reindexado como se fosse outro, e apareceria duplicado.

MAX_PARAM_VALUE = 250


def tags_for_run(record: RunRecord) -> Dict[str, str]:
    metadata = record.metadata
    tags = {
        "run_uid": record.run_uid,
        "run_dir": str(record.run_dir),
        "run_dir_relative": record.relative_dir,
        "family": record.family,
        "run_name": record.run_name,
        "layout": record.layout,
        "status": record.status,
    }
    for key in ("seed", "host", "config_path", "config_fingerprint", "resumed_from_step"):
        if metadata.get(key) is not None:
            tags[key] = str(metadata[key])
    git = metadata.get("git") or {}
    if git.get("sha"):
        tags["git_sha"] = str(git["sha"])
        tags["git_dirty"] = str(bool(git.get("dirty")))
    return {k: v[:MAX_PARAM_VALUE] for k, v in tags.items()}


def _looks_numeric(rows: List[dict], key: str, sample: int = 20) -> bool:
    seen = 0
    for row in rows[:sample]:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            float(raw)
        except ValueError:
            return False
        seen += 1
    return seen > 0


def published_metrics_by_run_dir(logs_root: Path) -> Dict[str, Dict[str, float]]:
    '''
    O numero PUBLICADO de cada celula, lido das CSVs de resultado e agregado por run.

    Sem isto a tabela de runs do tracker mostra a curva de validacao, mas nao responde "qual foi o
    resultado desta celula da grade". As chaves saem com prefixo `published/` para nao se confundirem
    com o que o treino logou. O join e por caminho RELATIVO: as CSVs guardam `output_dir` absoluto, e
    um prefixo diferente (outro mount, outro clone) faria o join falhar em silencio.
    '''
    logs_root = Path(logs_root).resolve()
    published: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(logs_root.glob("*/results_*.csv")):
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "output_dir" not in rows[0]:
            continue
        numeric = [key for key in rows[0]
                   if key not in {"output_dir", "run_id"} and _looks_numeric(rows, key)]
        for row in rows:
            output_dir = row.get("output_dir")
            if not output_dir:
                continue
            try:
                relative = str(Path(output_dir).resolve().relative_to(logs_root))
            except ValueError:
                continue   # fora de logs/: nao ha como associar
            for key in numeric:
                raw = row.get(key)
                if raw in (None, ""):
                    continue
                try:
                    published[relative][key].append(float(raw))
                except ValueError:
                    pass
    return {run: {f"published/{key}": sum(values) / len(values)
                  for key, values in metrics.items() if values}
            for run, metrics in published.items()}


# ---------------------------------------------------------------- configuracao dos trackers

# Os trackers ficam LIGADOS por padrao para o MLflow e DESLIGADOS para o Aim. O Aim 3.29.1 nao
# persiste params nem metricas (ver `scripts/index_runs.py:assert_aim_persists`), entao liga-lo por
# padrao produziria runs vazios em silencio.
DEFAULT_TRACKING: Dict[str, Any] = {
    "mlflow": {"enabled": True, "tracking_uri": "sqlite:///{logs}/_tracking/mlflow.db"},
    "aim": {"enabled": False, "repo": "{logs}/_tracking/aim"},
}
TRACKING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"


def infer_logs_root(run_dir: Path, fallback: Optional[Path] = None) -> Path:
    '''O ancestral chamado `logs`; senao o fallback; senao dois niveis acima do run.'''
    run_dir = Path(run_dir).resolve()
    for ancestor in run_dir.parents:
        if ancestor.name == "logs":
            return ancestor
    return Path(fallback).resolve() if fallback else run_dir.parent.parent


def resolve_tracking_config(logs_root: Path, experiment_logging: Optional[Any] = None) -> Dict[str, Any]:
    '''
    Configuracao efetiva dos trackers. Precedencia, da menor para a maior:

        DEFAULT_TRACKING  <  configs/tracking.yaml  <  bloco `logging:` do YAML do experimento

    de modo que ligar ou desligar o MLflow e uma decisao de UM arquivo, e nao de 40 YAMLs de
    experimento. `VFSS_TRACKING=off` desliga tudo -- para testes e para o caso de o tracker estar
    atrapalhando, sem editar nada.
    '''
    from omegaconf import OmegaConf

    layers = [OmegaConf.create(DEFAULT_TRACKING)]
    if TRACKING_CONFIG_PATH.is_file():
        layers.append(OmegaConf.load(str(TRACKING_CONFIG_PATH)))
    if experiment_logging:
        container = (OmegaConf.to_container(experiment_logging, resolve=True)
                     if OmegaConf.is_config(experiment_logging) else dict(experiment_logging))
        layers.append(OmegaConf.create(container))
    config = OmegaConf.to_container(OmegaConf.merge(*layers), resolve=True)

    if os.environ.get("VFSS_TRACKING", "").lower() in {"off", "0", "false", "no"}:
        for backend in ("mlflow", "aim"):
            config.setdefault(backend, {})["enabled"] = False

    for backend, key in (("mlflow", "tracking_uri"), ("aim", "repo")):
        if isinstance(config.get(backend, {}).get(key), str):
            config[backend][key] = config[backend][key].format(logs=str(Path(logs_root).resolve()))
    return config


# ---------------------------------------------------------------- loggers ao vivo

def _safe_mlflow_logger_cls():
    '''
    `MLFlowLogger` que NUNCA derruba o treino.

    O tracker e um indice; a fonte da verdade e o `csv/metrics.csv`. Uma excecao do backend -- banco
    travado por outro processo, disco cheio, um valor que o MLflow recusa -- levantada dentro de
    `log_metrics` interromperia um treino de horas por causa de algo que nao e o experimento. Cada
    falha e contada e avisada (as tres primeiras), e o run segue.

    Definida sob demanda para nao pagar o import do MLflow a quem nao o usa.
    '''
    from pytorch_lightning.loggers import MLFlowLogger

    class SafeMLFlowLogger(MLFlowLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Por INSTANCIA, nao por classe: um sweep roda muitas celulas no mesmo processo, e um
            # contador de classe silenciaria os avisos de toda celula depois das tres primeiras
            # falhas somadas -- justamente quando mais importa saber que um run ficou incompleto.
            self.failures = 0

        def _guard(self, name, call, *args, **kwargs):
            try:
                return call(*args, **kwargs)
            except Exception as error:
                self.failures += 1
                if self.failures <= 3:
                    print(f"[tracking] MLflow.{name} falhou ({type(error).__name__}: "
                          f"{str(error)[:120]}). O run continua; a fonte da verdade e "
                          f"csv/metrics.csv -- repare depois com "
                          f"`scripts/index_runs.py --replace --family <familia>`.")
                return None

        def log_hyperparams(self, params):
            return self._guard("log_hyperparams", super().log_hyperparams, params)

        def log_metrics(self, metrics, step=None):
            return self._guard("log_metrics", super().log_metrics, metrics, step)

        def finalize(self, status="success"):
            return self._guard("finalize", super().finalize, status)

    return SafeMLFlowLogger


def build_loggers(run_dir: Path, logs_root: Path, run_name: str,
                  logging_cfg: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
    '''
    Loggers de um run ao vivo: TensorBoard + CSV sempre, MLflow e Aim conforme a configuracao.

    Nenhum dos dois recebe artefatos -- so params, escalares e o caminho do run dir. Ver o contrato
    no topo deste arquivo. A identidade (`run_uid`) e as tags saem de `tags_for_run`, as MESMAS que o
    indexador usa, entao um run logado ao vivo e reconhecido pelo backfill em vez de duplicado.

    Uma falha ao CONSTRUIR um tracker (nao instalado, URI invalida) e avisada e ignorada: o treino
    nao depende dele.
    '''
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from src.utils import instantiate_from_config

    run_dir = Path(run_dir).resolve()
    logs_root = Path(logs_root).resolve()
    save_dir = str(run_dir)
    loggers: List[Any] = [
        TensorBoardLogger(save_dir=save_dir, name="", version="tensorboard"),
        CSVLogger(save_dir=save_dir, name="", version="csv"),
    ]

    config = logging_cfg if logging_cfg is not None else resolve_tracking_config(logs_root)
    record = RunRecord(
        run_dir=run_dir, logs_root=logs_root, run_uid=run_uid_for(run_dir, logs_root),
        layout="csv/metrics.csv + tensorboard/",
        metadata={"run_name": run_name, "status": "running", **(metadata or {})},
    )
    tags = tags_for_run(record)

    for backend, builder in (("mlflow", _build_mlflow_logger), ("aim", _build_aim_logger)):
        if not (config.get(backend) or {}).get("enabled"):
            continue
        try:
            loggers.append(builder(config[backend], record, tags, logs_root))
        except Exception as error:
            print(f"[tracking] {backend} desabilitado neste run ({type(error).__name__}: "
                  f"{str(error)[:160]}). O treino segue sem ele.")

    for logger_cfg in (config.get("loggers") or []):
        loggers.append(instantiate_from_config(logger_cfg))
    return loggers


def find_mlflow_run_id(tracking_uri: str, experiment_name: str, run_uid: str) -> Optional[str]:
    '''Run ja existente para este `run_uid` -- o que faz uma RETOMADA continuar o mesmo run.'''
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    found = client.search_runs([experiment.experiment_id],
                               filter_string=f"tags.run_uid = '{run_uid}'", max_results=1)
    return found[0].info.run_id if found else None


def _build_mlflow_logger(config: Dict[str, Any], record: RunRecord, tags: Dict[str, str],
                         logs_root: Path):
    tracking_uri = str(config["tracking_uri"])
    experiment_name = str(config.get("experiment_name") or record.family or "default")

    # Uma retomada tem o MESMO diretorio e portanto o mesmo run_uid. Sem reaproveitar o run do
    # MLflow, cada tentativa criaria um segundo run para o que e um unico treino.
    existing = find_mlflow_run_id(tracking_uri, experiment_name, record.run_uid)
    if existing:
        print(f"[tracking] MLflow: continuando o run existente {existing[:8]} (retomada)")

    return _safe_mlflow_logger_cls()(
        experiment_name=experiment_name,
        run_name=record.run_name,
        tracking_uri=tracking_uri,
        tags=tags,
        run_id=existing,
        # O store resolve o artifact root contra o CWD de quem CRIA o experimento e grava o caminho
        # absoluto -- sem fixar, o mesmo experimento teria raizes diferentes conforme de onde o
        # treino foi lancado. So vale para experimentos novos (o parametro e de `create_experiment`).
        artifact_location=str(Path(logs_root) / "_tracking" / "mlartifacts"),
        # Sem artefatos: os checkpoints ficam no run dir, e a tag `run_dir` aponta para eles.
        log_model=False,
    )


def _build_aim_logger(config: Dict[str, Any], record: RunRecord, tags: Dict[str, str],
                      logs_root: Path):
    print("[tracking] AVISO: o Aim 3.29.1 nao persiste params nem metricas (ver "
          "scripts/index_runs.py:assert_aim_persists); este run provavelmente ficara vazio la.")
    from aim.pytorch_lightning import AimLogger
    logger = AimLogger(
        repo=str(config["repo"]),
        experiment=str(config.get("experiment") or record.family or "default"),
        # O default e 10 s, e injetaria ~10 mil pontos de `__system__` num run de 2 h -- mais que as
        # metricas reais do run, e ausentes nos runs retro-indexados, o que tornaria os dois
        # conjuntos incomparaveis.
        system_tracking_interval=None,
        log_system_params=False,
    )
    for key, value in tags.items():
        logger.experiment.set(key, value, strict=False)
    return logger


# ---------------------------------------------------------------- numero publicado ao vivo

def log_published_metrics(client, run_id: str, metrics: Dict[str, float]) -> int:
    '''Grava as metricas `published/*` num run do MLflow. Devolve quantas.'''
    from mlflow.entities import Metric

    entries = [Metric(key=name[:250], value=value, timestamp=0, step=0)
               for name, value in metrics.items()]
    for start in range(0, len(entries), 900):
        client.log_batch(run_id, metrics=entries[start:start + 900])
    return len(entries)


def attach_published_metrics(tracking_config: Dict[str, Any], run_dir: Path, logs_root: Path) -> int:
    '''
    Anexa ao run do MLflow o numero publicado da celula, DEPOIS de ele estar na CSV de resultado.

    Existe porque o numero publicado so nasce no `evaluate()`, depois de o treino ter terminado e de
    os loggers terem fechado -- um run logado ao vivo nunca o receberia, e o indexador, ao ver o run
    ja indexado, o pularia. Devolve quantas metricas anexou (0 se o MLflow esta desligado ou o run
    nao esta na CSV). Nunca levanta: e um acabamento, nao parte do treino.
    '''
    try:
        if not (tracking_config.get("mlflow") or {}).get("enabled"):
            return 0
        from mlflow.tracking import MlflowClient

        record_dir = Path(run_dir).resolve()
        logs_root = Path(logs_root).resolve()
        relative = str(record_dir.relative_to(logs_root))
        published = published_metrics_by_run_dir(logs_root).get(relative)
        if not published:
            return 0
        tracking_uri = str(tracking_config["mlflow"]["tracking_uri"])
        experiment_name = relative.split(os.sep)[0]
        run_id = find_mlflow_run_id(tracking_uri, experiment_name, run_uid_for(record_dir, logs_root))
        if not run_id:
            return 0
        client = MlflowClient(tracking_uri=tracking_uri)
        if any(key.startswith("published/") for key in client.get_run(run_id).data.metrics):
            return 0   # ja anexado
        return log_published_metrics(client, run_id, published)
    except Exception as error:
        print(f"[tracking] nao foi possivel anexar o numero publicado ({type(error).__name__}: "
              f"{str(error)[:120]}); `scripts/index_runs.py` o completa depois.")
        return 0
