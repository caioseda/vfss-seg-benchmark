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
'''
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.experiment import Experiment
from src.tracking import attach_published_metrics
from src.utils import _resolve_config_path, capture_console_output, get_obj_from_str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUMABLE_STATUSES = {"running", "failed"}


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


class Sweep:
    def __init__(self, config_path: str | Path, log_dir: Optional[str | Path] = None):
        self.config_path = _resolve_sweep_config(config_path)
        self.config = self._load_config(self.config_path)
        self.plan = get_obj_from_str(str(self.config.plan))(self.config)
        self.log_dir = Path(log_dir).resolve() if log_dir else (PROJECT_ROOT / "logs")
        self.family_dir = self.log_dir / self.plan.family
        self.sweep_dir = self.family_dir / "_sweep"
        self.state_path = self.sweep_dir / "state.json"
        self.results_csv = self.family_dir / self.plan.results_csv_name()

    # ------------------------------------------------------------------ config

    @staticmethod
    def _load_config(config_path: Path) -> DictConfig:
        config = OmegaConf.load(str(_resolve_config_path(config_path)))
        for field in ("type", "plan", "family", "axes"):
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

    def completed_run_ids(self) -> set:
        '''
        Celulas que a CSV de resultado ja contem. Esta e a definicao de "pronto" que os notebooks
        usam, e mante-la e o que faz este runner ser continuo com o que ja foi rodado.
        '''
        results = self.load_results()
        if len(results) and "run_id" in results.columns:
            return set(results.run_id.unique())
        return set()

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

    def plan_summary(self) -> pd.DataFrame:
        done = self.completed_run_ids()
        state = self.read_state()
        rows = []
        for cell in self.plan.cells():
            run_id = self.plan.run_id(cell)
            entry = state["cells"].get(run_id, {})
            if run_id in done:
                status = "done (na CSV)"
            elif entry.get("status") in RESUMABLE_STATUSES:
                status = f"retomavel ({entry['status']})"
            else:
                status = entry.get("status", "pendente")
            rows.append({"run_id": run_id, **cell, "status": status})
        return pd.DataFrame(rows)

    def run(self, fast_dev_run: bool | int = False, force_resume: bool = False,
            dry_run: bool = False) -> List[Dict[str, Any]]:
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        console_log = self.sweep_dir / "console.log"
        cells = self.plan.cells()

        with capture_console_output(console_log):
            print(f"Sweep: {self.plan.name}  (familia {self.plan.family})")
            print(f"Config: {self.config_path}")
            print(f"Celulas: {len(cells)}   Resultados: {self.results_csv}")
            print(f"Estado: {self.state_path}")
            print(self.plan_summary().to_string(index=False))
            if dry_run:
                print("\n--dry-run: nada foi treinado.")
                return []

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

                experiment = Experiment(
                    config_path=self.plan.experiment_config(cell),
                    overrides=self.plan.overrides(cell),
                    log_dir=self.log_dir,
                    family=self.plan.family,
                    run_uid=run_uid,
                )
                self._update_cell(run_id, status="running", run_uid=run_uid, cell=cell,
                                  run_name=self.plan.run_name(cell), attempts=attempts)
                try:
                    outcome = experiment.run(seed=int(cell["seed"]), fast_dev_run=fast_dev_run,
                                             force_resume=force_resume)
                except BaseException as error:
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

                self._update_cell(run_id, status="done",
                                  run_dir=str(experiment.experiment_run_dir),
                                  best_checkpoint_path=outcome.get("best_checkpoint_path"))
                results.append(outcome)

            print(f"\nConcluido. {len(results)} celulas treinadas nesta sessao.")
            return results


# ---------------------------------------------------------------------- notebooks

def submit_sweep(config_path: str | Path, session: Optional[str] = None,
                 extra_args: tuple = ()) -> str:
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


def sweep_status(config_path: str | Path, log_dir: Optional[str | Path] = None) -> pd.DataFrame:
    '''Estado atual de cada celula da grade, lido do disco. Nao depende do processo estar vivo.'''
    return Sweep(config_path, log_dir=log_dir).plan_summary()


def watch_sweep(config_path: str | Path, interval: int = 30, log_dir: Optional[str | Path] = None,
                tail: int = 3) -> None:
    '''
    Acompanha um sweep lendo `_sweep/state.json` e o fim do `console.log`.

    Interromper isto (Ctrl-C / botao de parar) NAO para o treino: quem treina e o processo do tmux.
    Por isso a mensagem final diz explicitamente o que acabou de acontecer -- a confusao entre
    "parei de olhar" e "parei o treino" e exatamente o que o fluxo antigo tornava impossivel de
    distinguir.
    '''
    import time

    sweep = Sweep(config_path, log_dir=log_dir)
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
