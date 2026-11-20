'''
Roda uma grade de experimentos fora do notebook, de forma retomavel.

Retomada em DOIS niveis, que e o que faz uma queda de SSH deixar de custar trabalho:

  nivel 1  celulas ja concluidas sao puladas. A fonte de verdade e a CSV de resultado da familia
           -- a MESMA que os notebooks ja escrevem -- e so depois o `_sweep/state.json`. Por isso
           um sweep lancado hoje sobre a grade que o notebook ja rodou nao retreina nada.
  nivel 2  uma celula interrompida no meio retoma de `checkpoints/last.ckpt`, porque o `run_uid`
           e atribuido uma vez e reusado nas tentativas seguintes (ver `Experiment`).

Um status `running` encontrado no inicio NAO significa que algo esta rodando: um SIGKILL nao da
chance de escrever `failed`. Um `running` obsoleto e portanto tratado como retomavel, que e o caso
comum depois de uma conexao cair.

    python scripts/run_sweep.py --config configs/sweep/x1.yaml

Shards. `--shard K/N` (1-indexado) divide a grade entre processos -- tipicamente um por servidor
(`scripts/remote.py`). A divisao e por UNIDADE DE PAREAMENTO (`SweepPlan.pairing_key`), nao por
celula: todos os metodos de um fold/seed vao para a mesma GPU. Cada shard grava estado e CSVs em
`logs/<familia>/_sweep/<tag>/`, nunca nos arquivos canonicos da familia, e `merge_shards` os
consolida depois. Sem `--shard`, nada muda.
'''
import json
import os
import re
import socket
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.experiment import Experiment
from src.tracking import attach_published_metrics
from src.utils import _resolve_config_path, capture_console_output, get_obj_from_str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUMABLE_STATUSES = {"running", "failed"}
SHARD_TAG_ENV = "VFSS_SHARD_TAG"


def parse_shard(spec: str | Tuple[int, int] | None) -> Optional[Tuple[int, int]]:
    '''`"2/3"` -> `(2, 3)`. 1-indexado: os shards de uma divisao em tres sao 1/3, 2/3 e 3/3.'''
    if spec is None or spec == "":
        return None
    if isinstance(spec, (tuple, list)):
        index, total = int(spec[0]), int(spec[1])
    else:
        match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(spec))
        if not match:
            raise ValueError(f"shard invalido: {spec!r}. Use K/N, ex.: 1/2 e 2/2.")
        index, total = int(match.group(1)), int(match.group(2))
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"shard invalido: {index}/{total}. K vai de 1 a N (1-indexado).")
    return index, total


def default_shard_tag() -> str:
    '''O nome do host -- o que distingue o diretorio de cada shard depois que os logs se juntam.'''
    tag = os.environ.get(SHARD_TAG_ENV) or socket.gethostname().split(".")[0]
    return tag.lower()


def _valid_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or tag == "configs":
        raise ValueError(f"shard tag invalida: {tag!r} (letras, numeros, '.', '_', '-'; e nao 'configs').")
    return tag


def shard_cells(plan, cells: List[Dict[str, Any]], shard: Tuple[int, int]) -> List[Dict[str, Any]]:
    '''
    As celulas do shard K de N. As unidades de pareamento, na ordem em que aparecem na grade, sao
    distribuidas em rodizio: a i-esima unidade (0-indexada) vai para o shard `i % N + 1`. A ordem das
    celulas dentro do shard e a da grade.
    '''
    index, total = shard
    units: Dict[Tuple[Any, ...], int] = {}
    for cell in cells:
        units.setdefault(plan.pairing_key(cell), len(units))
    return [cell for cell in cells if units[plan.pairing_key(cell)] % total + 1 == index]


def _resolve_sweep_config(config_path: str | Path) -> Path:
    '''
    Resolve um caminho relativo contra a RAIZ DO REPOSITORIO, nao contra o cwd.

    Um notebook roda a partir do proprio diretorio, entao `"configs/sweep/x1.yaml"` apontaria para
    `notebooks/hyphotesis_testing/configs/...` e falharia. O mesmo motivo pelo qual `Experiment`
    documenta passar `log_dir` absoluto -- aqui e resolvido em vez de exigido do chamador, para que
    a mesma string funcione no notebook, no `scripts/run_sweep.py` e num teste.
    '''
    path = Path(config_path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def _normalize_overrides(overrides) -> List[str]:
    '''
    `{"axes.arch": ["fpn"], "training.max_steps": 200}` ou `["axes.arch=[fpn]"]` -> dotlist do OmegaConf.

    O dotlist é a forma que atravessa a fronteira de processo: `submit_sweep` a passa como `--set` para
    `scripts/run_sweep.py`, então o notebook e o processo do tmux enxergam a MESMA grade.
    '''
    if not overrides:
        return []
    if isinstance(overrides, dict):
        import yaml
        return [f"{key}={yaml.safe_dump(value, default_flow_style=True).strip().removesuffix('...').strip()}"
                for key, value in overrides.items()]
    return [str(item) for item in overrides]


class Sweep:
    def __init__(self, config_path: str | Path, log_dir: Optional[str | Path] = None,
                 overrides: Optional[Dict[str, Any] | List[str]] = None,
                 shard: Optional[str | Tuple[int, int]] = None, shard_tag: Optional[str] = None):
        '''
        Args:
            overrides: sobrescritas da config do SWEEP (não do experimento), como dict de chaves com
                pontos ou dotlist -- ex. `{"axes.arch": ["fpn"]}` para rodar o E2 com outra arquitetura
                sem editar o YAML.
            shard: `"K/N"` (1-indexado) para rodar só a K-ésima de N partes da grade. `"1/1"` roda a grade
                inteira, mas com os arquivos isolados no diretório do shard -- é o que um host remoto usa.
            shard_tag: nome do diretório do shard (`_sweep/<tag>/`); padrão `$VFSS_SHARD_TAG` ou o hostname.
        '''
        self.config_path = _resolve_sweep_config(config_path)
        self.config = self._load_config(self.config_path)
        self.overrides = _normalize_overrides(overrides)
        if self.overrides:
            self.config = OmegaConf.merge(self.config, OmegaConf.from_dotlist(self.overrides))
        self.log_dir = Path(log_dir).resolve() if log_dir else (PROJECT_ROOT / "logs")
        self.plan = get_obj_from_str(str(self.config.plan))(self.config, log_dir=self.log_dir)
        self.family_dir = self.log_dir / self.plan.family
        self.shard = parse_shard(shard)
        self.shard_tag: Optional[str] = None
        if self.shard is not None:
            if not getattr(self.plan, "shardable", True):
                raise ValueError(
                    f"{type(self.plan).__name__} nao pode ser dividido em shards: as celulas dependem de "
                    "estado compartilhado entre elas (ex.: um study do Optuna). Rode-o sem --shard."
                )
            self.shard_tag = _valid_tag(shard_tag or default_shard_tag())
            self.plan.shard_dir = self.family_dir / "_sweep" / self.shard_tag
        self.sweep_dir = self.plan.sweep_dir
        self.state_path = self.sweep_dir / "state.json"
        self.results_csv = self.plan.tables_dir / self.plan.results_csv_name()
        self.canonical_results_csv = self.family_dir / self.plan.results_csv_name()

    # ------------------------------------------------------------------ config

    @staticmethod
    def _load_config(config_path: Path) -> DictConfig:
        config = OmegaConf.load(str(_resolve_config_path(config_path)))
        for field in ("type", "family", "plan", "axes"):
            if field not in config:
                raise ValueError(f"Falta o campo obrigatorio `{field}` em {config_path}")
        if str(config.type) != "sweep":
            raise ValueError(f"`type` deve ser 'sweep', e '{config.type}' em {config_path}")
        return config

    # ------------------------------------------------------------------ state

    def read_state(self) -> Dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {"sweep": self.plan.name, "family": self.plan.family, "cells": {}}

    def shard_dirs(self) -> List[Path]:
        '''Diretórios de shard desta família (`_sweep/<tag>/`), locais ou trazidos por `remote.py sync`.'''
        root = self.family_dir / "_sweep"
        if not root.is_dir():
            return []
        return sorted(path for path in root.iterdir()
                      if path.is_dir() and (path / "state.json").exists())

    def all_cell_states(self) -> Dict[str, Dict[str, Any]]:
        '''
        O estado de cada célula visto por TODOS os processos: o `state.json` canônico mais o de cada shard,
        com o host em `host`. Quando a mesma célula aparece em mais de um, vale a atualização mais recente.
        '''
        sources = [(None, self.family_dir / "_sweep" / "state.json")]
        sources += [(path.name, path / "state.json") for path in self.shard_dirs()]
        merged: Dict[str, Dict[str, Any]] = {}
        for host, path in sources:
            try:
                cells = json.loads(path.read_text()).get("cells", {})
            except Exception:
                continue
            for run_id, entry in cells.items():
                entry = {**entry, "host": host or entry.get("host") or "local"}
                current = merged.get(run_id)
                if current is None or str(entry.get("updated_at", "")) >= str(current.get("updated_at", "")):
                    merged[run_id] = entry
        return merged

    def write_state(self, state: Dict[str, Any]) -> None:
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now().isoformat()
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    def _update_cell(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        state = self.read_state()
        entry = state["cells"].setdefault(run_id, {})
        entry.update(fields)
        entry["updated_at"] = datetime.now().isoformat()
        self.write_state(state)
        return entry

    # ------------------------------------------------------------------ results

    def load_results(self) -> pd.DataFrame:
        if self.results_csv.exists():
            return pd.read_csv(self.results_csv)
        return pd.DataFrame()

    def results_csvs(self) -> List[Path]:
        '''A CSV canônica da família e a de cada shard: juntas, a definição de "pronto".'''
        name = self.plan.results_csv_name()
        paths = [self.canonical_results_csv] + [path / name for path in self.shard_dirs()]
        if self.results_csv not in paths:
            paths.append(self.results_csv)
        return [path for path in paths if path.exists()]

    def completed_run_ids(self) -> set:
        '''
        Celulas que alguma CSV de resultado ja contem -- a canonica (a definicao de "pronto" que os
        notebooks usam) ou a de um shard ainda nao consolidado. Por isso relancar um shard, ou rodar sem
        shard depois de juntar os logs, nao retreina o que outro host ja concluiu.
        '''
        done = set()
        for path in self.results_csvs():
            try:
                done |= set(pd.read_csv(path, usecols=["run_id"]).run_id.unique())
            except (ValueError, pd.errors.EmptyDataError):
                continue   # CSV sem coluna run_id ou vazia
        return done

    def _append_results(self, rows: pd.DataFrame) -> None:
        results = pd.concat([self.load_results(), rows], ignore_index=True)
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.results_csv, index=False)

    # ------------------------------------------------------------------ run

    def _run_uid_for(self, run_id: str, state: Dict[str, Any]) -> str:
        '''
        Atribuido na primeira tentativa e reusado depois: e o que mantem o diretorio de run estavel
        entre tentativas e portanto o que torna a retomada possivel.
        '''
        existing = state["cells"].get(run_id, {}).get("run_uid")
        return existing or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def cells(self) -> List[Dict[str, Any]]:
        '''As células que ESTE processo roda: a grade inteira, ou a parte do shard.'''
        cells = self.plan.cells()
        return shard_cells(self.plan, cells, self.shard) if self.shard else cells

    def plan_summary(self) -> pd.DataFrame:
        '''
        Estado de cada célula deste processo. Sem shard, a grade inteira com o estado agregado de todos
        os shards (coluna `host`) -- é o que `watch_sweep` mostra na delaunay.
        '''
        done = self.completed_run_ids()
        states = self.all_cell_states() if self.shard is None else {
            run_id: {**entry, "host": self.shard_tag} for run_id, entry in self.read_state()["cells"].items()}
        show_host = any(entry.get("host") not in (None, "local") for entry in states.values())
        rows = []
        for cell in self.cells():
            run_id = self.plan.run_id(cell)
            entry = states.get(run_id, {})
            if run_id in done:
                status = "done (na CSV)"
            elif entry.get("status") in RESUMABLE_STATUSES:
                status = f"retomavel ({entry['status']})"
            else:
                status = entry.get("status", "pendente")
            row = {"run_id": run_id, **cell, "status": status}
            if show_host:
                row["host"] = entry.get("host", "")
            rows.append(row)
        return pd.DataFrame(rows)

    def run(self, fast_dev_run: bool | int = False, force_resume: bool = False,
            dry_run: bool = False) -> List[Dict[str, Any]]:
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        console_log = self.sweep_dir / "console.log"
        cells = self.cells()

        with capture_console_output(console_log):
            print(f"Sweep: {self.plan.name}  (familia {self.plan.family})")
            if self.shard:
                print(f"Shard: {self.shard[0]}/{self.shard[1]}  tag {self.shard_tag}  "
                      f"({len(cells)} de {len(self.plan.cells())} celulas)")
            print(f"Config: {self.config_path}")
            print(f"Celulas: {len(cells)}   Resultados: {self.results_csv}")
            print(f"Estado: {self.state_path}")
            print(self.plan_summary().to_string(index=False))
            if dry_run:
                print("\n--dry-run: nada foi treinado.")
                return []

            self.plan.begin(fast_dev_run=fast_dev_run)
            results: List[Dict[str, Any]] = []
            for cell in tqdm(cells, desc=f"grade {self.plan.name}", unit="cell"):
                run_id = self.plan.run_id(cell)
                if run_id in self.completed_run_ids():
                    tqdm.write(f"[pula] {run_id} (ja em {self.results_csv.name})")
                    continue

                state = self.read_state()
                run_uid = self._run_uid_for(run_id, state)
                attempts = int(state["cells"].get(run_id, {}).get("attempts", 0)) + 1
                tqdm.write(f"[treina] {run_id}  (tentativa {attempts}, uid {run_uid})")

                self.plan.start_cell(cell)
                experiment = Experiment(
                    config_path=self.plan.experiment_config(cell),
                    overrides=self.plan.overrides(cell),
                    log_dir=self.log_dir,
                    family=self.plan.family,
                    run_uid=run_uid,
                    callbacks=self.plan.callbacks(cell),
                )
                self._update_cell(run_id, status="running", run_uid=run_uid, cell=cell,
                                  run_name=self.plan.run_name(cell), attempts=attempts)
                try:
                    outcome = experiment.run(seed=int(cell["seed"]), fast_dev_run=fast_dev_run,
                                             force_resume=force_resume)
                except BaseException as error:
                    traceback.print_exc()  # inside the sweep's console.log capture
                    self._update_cell(run_id, status="failed",
                                      error=f"{type(error).__name__}: {error}")
                    raise

                per_sample = self.plan.evaluate(experiment)
                per_sample.insert(0, "run_id", run_id)
                for key, value in cell.items():
                    per_sample[key] = value
                per_sample["output_dir"] = str(experiment.experiment_run_dir)
                if not fast_dev_run:
                    self._append_results(per_sample)
                    # O numero publicado so passa a existir agora, depois do `evaluate()` -- os
                    # loggers do treino ja fecharam. Anexa-lo ao run do MLflow aqui evita que ele
                    # fique sem o resultado da celula ate alguem lembrar de rodar o indexador.
                    attach_published_metrics(experiment.tracking_config,
                                             experiment.experiment_run_dir,
                                             experiment.tracking_logs_root)
                # Depois da CSV, nunca antes: se o processo cair entre os dois, a CSV ja diz o que
                # aconteceu e o plano reconcilia no proximo `begin`.
                self.plan.finish_cell(cell, experiment, per_sample)

                self._update_cell(run_id, status="done",
                                  run_dir=str(experiment.experiment_run_dir),
                                  best_checkpoint_path=outcome.get("best_checkpoint_path"))
                results.append(outcome)

            print(f"\nConcluido. {len(results)} celulas treinadas nesta sessao.")
            return results


# ---------------------------------------------------------------------- shards

def merge_shards(family_dir: str | Path, dry_run: bool = False) -> Dict[str, Any]:
    '''
    Consolida os shards de uma família nos arquivos canônicos que os notebooks leem.

    Para cada `results_*.csv` encontrada em `_sweep/<tag>/`, acrescenta à CSV canônica de mesmo nome as
    linhas dos `run_id` que ela ainda não tem. A canônica tem precedência, e entre shards vale o primeiro
    em ordem alfabética -- um `run_id` em dois shards é avisado (não deveria acontecer: a divisão é
    disjunta). As células concluídas também entram no `_sweep/state.json` canônico, para que um método de
    duas etapas rodado depois sem shard encontre a etapa 1 treinada em outro host.

    Idempotente: rodar de novo sem shards novos não muda nada.
    '''
    family_dir = Path(family_dir)
    sweep_root = family_dir / "_sweep"
    shard_dirs = sorted(path for path in sweep_root.iterdir() if path.is_dir()) if sweep_root.is_dir() else []
    report: Dict[str, Any] = {"shards": [path.name for path in shard_dirs], "tables": {},
                              "duplicates": {}, "missing_output_dirs": [], "state_cells_added": 0}

    names = sorted({csv.name for path in shard_dirs for csv in path.glob("results_*.csv")})
    for name in names:
        canonical_path = family_dir / name
        canonical = pd.read_csv(canonical_path) if canonical_path.exists() else pd.DataFrame()
        known = set(canonical.run_id.unique()) if "run_id" in canonical.columns else set()
        added, owner = [], {}
        for shard in shard_dirs:
            path = shard / name
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if "run_id" not in frame.columns or not len(frame):
                continue
            for run_id in frame.run_id.unique():
                if run_id in owner:
                    report["duplicates"].setdefault(name, []).append(f"{run_id} ({owner[run_id]}, {shard.name})")
                else:
                    owner.setdefault(run_id, shard.name)
            fresh = frame[~frame.run_id.isin(known) & frame.run_id.map(owner).eq(shard.name)]
            if len(fresh):
                added.append(fresh)
        new_ids = {run_id for frame in added for run_id in frame.run_id.unique()}
        report["tables"][name] = {"run_ids_added": len(new_ids), "rows_added": sum(map(len, added))}
        for frame in added:
            if "output_dir" in frame.columns:
                report["missing_output_dirs"] += sorted(
                    str(d) for d in frame.output_dir.dropna().unique() if not Path(d).is_dir())
        if added and not dry_run:
            merged = pd.concat([canonical, *added], ignore_index=True)
            tmp = canonical_path.with_suffix(".csv.tmp")
            merged.to_csv(tmp, index=False)
            tmp.replace(canonical_path)

    state_path = sweep_root / "state.json"
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        state = {"family": family_dir.name, "cells": {}}
    for shard in shard_dirs:
        try:
            cells = json.loads((shard / "state.json").read_text()).get("cells", {})
        except Exception:
            continue
        for run_id, entry in cells.items():
            if entry.get("status") != "done" or state["cells"].get(run_id, {}).get("status") == "done":
                continue
            state["cells"][run_id] = {**entry, "host": shard.name}
            report["state_cells_added"] += 1
    report["missing_output_dirs"] = sorted(set(report["missing_output_dirs"]))
    if report["state_cells_added"] and not dry_run:
        state["updated_at"] = datetime.now().isoformat()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str))
    return report


# ---------------------------------------------------------------------- invalidar celulas

def invalidation_targets(config_path: str | Path, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    '''
    As celulas que `--set` recorta da grade, para `src.sweep_invalidate.invalidate`:
    `{"family", "run_ids", "run_names": {run_id: run_name}, "cells"}`.

    Exige ao menos uma sobrescrita: invalidar a grade inteira por esquecimento de `--set` jogaria fora
    todos os metodos que estao certos.
    '''
    overrides = _normalize_overrides(overrides)
    if not overrides:
        raise ValueError("passe --set para recortar a grade (ex.: \"axes.method=[fixmatch]\"); "
                         "invalidar a grade inteira nao e permitido por aqui.")
    sweep = Sweep(config_path, overrides=overrides)
    cells = sweep.plan.cells()
    return {
        "family": sweep.plan.family,
        "run_ids": [sweep.plan.run_id(cell) for cell in cells],
        "run_names": {sweep.plan.run_id(cell): sweep.plan.run_name(cell) for cell in cells},
        "cells": cells,
    }


def archive_label(family: str, label: str) -> str:
    '''`<familia>-<rotulo>`: o nome do diretorio em `logs/_arquivo/` (ou `<root>/arquivo/` num host).'''
    label = _valid_tag(label)
    return label if label.startswith(f"{family}-") else f"{family}-{label}"


# ---------------------------------------------------------------------- notebooks

def submit_sweep(config_path: str | Path, session: Optional[str] = None,
                 extra_args: tuple = (), overrides: Optional[Dict[str, Any] | List[str]] = None) -> str:
    '''
    Lanca um sweep numa sessao tmux destacada e devolve NA HORA.

    E isto que um notebook chama em vez de treinar no proprio kernel. O treino passa a viver num
    processo que nao pertence ao kernel, entao fechar o notebook, perder o SSH ou reiniciar o kernel
    nao o mata -- que era a razao inteira desta mudanca.

        submit_sweep("configs/sweep/x1.yaml")
        watch_sweep("x1")
    '''
    import subprocess

    script = PROJECT_ROOT / "scripts" / "submit.sh"
    extra_args = tuple(extra_args)
    dotlist = _normalize_overrides(overrides)
    if dotlist:
        extra_args += ("--set", *dotlist)
    command = [str(script), str(config_path)]
    if session or extra_args:
        command.append(session or Path(config_path).stem)
        command.extend(str(a) for a in extra_args)

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(completed.stdout.strip() or completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"submit.sh falhou (codigo {completed.returncode}). Uma sessao com esse nome pode ja "
            "estar rodando -- e nesse caso o treino esta seguindo, nao ha nada a fazer."
        )
    return session or Path(config_path).stem


def sweep_status(config_path: str | Path, log_dir: Optional[str | Path] = None,
                 overrides: Optional[Dict[str, Any] | List[str]] = None) -> pd.DataFrame:
    '''Estado atual de cada celula da grade, lido do disco. Nao depende do processo estar vivo.'''
    return Sweep(config_path, log_dir=log_dir, overrides=overrides).plan_summary()


def watch_sweep(config_path: str | Path, interval: int = 30, log_dir: Optional[str | Path] = None,
                tail: int = 3, overrides: Optional[Dict[str, Any] | List[str]] = None) -> None:
    '''
    Acompanha um sweep lendo `_sweep/state.json` e o fim do `console.log`.

    Interromper isto (Ctrl-C / botao de parar) NAO para o treino: quem treina e o processo do tmux.
    Por isso a mensagem final diz explicitamente o que acabou de acontecer -- a confusao entre
    "parei de olhar" e "parei o treino" e exatamente o que o fluxo antigo tornava impossivel de
    distinguir.
    '''
    import time

    sweep = Sweep(config_path, log_dir=log_dir, overrides=overrides)
    console_log = sweep.sweep_dir / "console.log"
    try:
        from IPython.display import clear_output
    except ImportError:
        clear_output = None

    try:
        while True:
            summary = sweep.plan_summary()
            done = int((summary.status.str.startswith("done")).sum())
            if clear_output:
                clear_output(wait=True)
            print(f"{sweep.plan.name}: {done}/{len(summary)} celulas concluidas "
                  f"({time.strftime('%H:%M:%S')})")
            print(summary.to_string(index=False))
            if console_log.exists() and tail:
                lines = console_log.read_text(errors="replace").splitlines()
                print("\n--- console ---")
                for line in lines[-tail:]:
                    print(line[:160])
            if done >= len(summary):
                print("\nGrade completa.")
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nParei de ACOMPANHAR. O treino segue rodando na sessao tmux "
              f"-- `watch_sweep({str(config_path)!r})` para voltar a olhar.")
