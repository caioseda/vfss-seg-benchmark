'''
E2 -- tuning dos pesos SSL, no molde do tuning A.1 do E1 (`src/sweeps/e1.py`, `src/tuning.py`).

Protocolo e justificativa: docs/e2-tuning-ssl.md. Em uma frase: cada método SSL ajusta SÓ os pesos das
perdas sobre não rotulados (a força do sinal não supervisionado frente à âncora), com o MESMO orçamento
(25 trials TPE + MedianPruner, top-3 confirmadas), no fold 0 do desenvolvimento, selecionando pela
validação fixa. lr/wd seguem os do E1; as constantes do mecanismo (EMA, τ, T, percentil...) seguem os
valores publicados. Nada aqui carrega o teste INCA nem o UNM.

  E2TuningSweep         etapa 1 (busca). Um study do Optuna por método (e seed/arch/política). Os
                        métodos sem espaço de busca (o `supervised`) rodam UMA vez, pelo cronograma
                        completo: são a referência e, via `teacher_from`, a etapa 1 do DTC-TRUS.
  E2TuningConfirmSweep  etapa 2 (confirmação): as `top_k` de cada método retreinadas por 8000 passos,
                        sem pruning. A melhor é a congelada (`freeze_ssl_hparams`), e é ela que o
                        `E2Sweep` lê via `ssl_hparams_file` (configs/sweep/e2-tuned.yaml, e2-cv-tuned.yaml).

Herança (tuning incremental). `inherit: {param: método}` no espaço de um método fixa `param` no valor
que a CONFIRMAÇÃO escolheu para `método` (o MT temporal herda o `consistency_weight` do Mean Teacher e
ajusta só o `temporal_weight`). Por isso esses métodos rodam numa segunda fase, depois da confirmação do
método de origem, e vêm por último na grade.

Shards. O study é estado compartilhado só entre os trials de UM método; a unidade de pareamento é o
método (com o `supervised` no mesmo grupo do DTC-TRUS, que o usa como professor), e o journal do Optuna
fica no diretório do shard. Assim métodos diferentes podem rodar em hosts diferentes, cada study inteiro
numa GPU só.
'''
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.data.splits import resolve_project_path
from src.evaluation import evaluate_per_class
from src.sweeps.e2 import E2Sweep
from src.tuning import (
    OptunaPruning,
    journal_storage,
    make_pruner,
    objective_value,
    open_study,
    open_trial,
    rank_trials,
)

HP_PREFIX = "hp_"


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def hp_columns(frame: pd.DataFrame) -> List[str]:
    return [column for column in frame.columns if str(column).startswith(HP_PREFIX)]


def point_from_row(row: pd.Series) -> Dict[str, float]:
    '''As colunas `hp_*` não nulas de uma linha -> `{param: valor}`.'''
    return {str(column)[len(HP_PREFIX):]: float(row[column]) for column in row.index
            if str(column).startswith(HP_PREFIX) and pd.notna(row[column])}


def find_cell_state(family_dir: Path, run_id: str) -> Dict[str, Any]:
    '''A entrada de `run_id` no `state.json` canônico da família ou, se ausente, no de algum shard.'''
    sweep_dir = family_dir / "_sweep"
    candidates = [sweep_dir / "state.json"] + sorted(sweep_dir.glob("*/state.json"))
    for path in candidates:
        try:
            entry = json.loads(path.read_text()).get("cells", {}).get(run_id)
        except FileNotFoundError:
            continue
        if entry:
            return entry
    return {}


class _E2TuningStage(E2Sweep):
    '''
    O que busca e confirmação compartilham: treino em `data.train_folds` (fold 0) com o pool não rotulado
    SÓ desses vídeos, sem split de teste, avaliação por frame na validação fixa (a mesma que escolhe o
    checkpoint), colunas `hp_*` com o ponto avaliado.
    '''

    compared_axes = ("trial", "stage1_rank")

    # ------------------------------------------------------------------ spaces

    def search_config(self):
        '''O YAML da busca (o próprio, na busca; `source_sweep`, na confirmação).'''
        return self.config

    def method_spaces(self) -> Dict[str, Dict[str, Any]]:
        return _plain(self.search_config().search_space.methods) or {}

    def is_tuned(self, method: str) -> bool:
        return str(method) in self.method_spaces()

    def space(self, method: str) -> Dict[str, Any]:
        spaces = self.method_spaces()
        if method not in spaces:
            raise KeyError(f"`search_space.methods` não tem o método '{method}'")
        space = spaces[method]
        params = space.get("params") or {}
        inherit = space.get("inherit") or {}
        overlap = set(params) & set(inherit)
        if overlap:
            raise ValueError(f"'{method}': {sorted(overlap)} não podem ser ajustados e herdados ao mesmo tempo")
        for point in space.get("fixed_trials") or []:
            if set(point) != set(params):
                raise ValueError(f"fixed_trials {point} de '{method}' deve definir exatamente {sorted(params)}")
        return space

    def inherits(self, method: str) -> bool:
        return self.is_tuned(method) and bool(self.space(method).get("inherit"))

    # ------------------------------------------------------------------ cells

    def run_id(self, cell: Dict[str, Any]) -> str:
        base = f"{cell['arch']}-{cell['method']}-{cell['unlabeled_policy']}"
        trial = cell.get("trial")
        return f"{base}-s{cell['seed']}" if trial is None else f"{base}-t{int(trial):02d}-s{cell['seed']}"

    def pairing_key(self, cell: Dict[str, Any]) -> Tuple[Any, ...]:
        '''Unidade = método. A etapa 1 de um método de duas etapas cai no grupo dele (mesmo shard/state).'''
        method = str(cell["method"])
        for student, teacher in self._teacher_methods().items():
            if method == teacher:
                method = student
                break
        return (("seed", cell["seed"]), ("arch", cell["arch"]),
                ("unlabeled_policy", cell["unlabeled_policy"]), ("method", method))

    def _base_axes(self) -> Dict[str, List[Any]]:
        axes = self.axes()
        for reserved in ("trial", "stage1_rank", "fold"):
            if reserved in axes:
                raise ValueError(f"`{reserved}` não é eixo declarável deste sweep.")
        return axes

    # ------------------------------------------------------------------ config

    def _reference_steps(self) -> int:
        training = self.config.training
        return int(training.get("reference_max_steps", training.max_steps))

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        overrides = super().overrides(cell)
        params = overrides["data"]["params"]
        shared = params["shared_dataset_params"]
        shared["train_folds"] = [int(f) for f in self.config.data.train_folds]
        shared["cv_fold"] = None
        shared["holdout_test"] = False      # os YAMLs do E2 trazem True (treino final)
        params["test"] = None                # sem teste: `Experiment.run` pula `trainer.test`
        params["cache_splits"] = ["validation"]   # os YAMLs do E2 também fazem cache do 'test', que aqui não existe
        if cell.get("trial") is None:
            steps = self._reference_steps()
            overrides["trainer"]["params"]["max_steps"] = steps
            overrides["lr_scheduler"]["params"]["T_max"] = steps
        else:
            lit = overrides.setdefault("lit_module", {}).setdefault("params", {})
            lit.update(self.point(cell))
        return overrides

    def point(self, cell: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------ teacher (DTC-TRUS)

    def teacher_sibling(self, cell: Dict[str, Any], teacher_method: str) -> Dict[str, Any]:
        '''A etapa 1 é a célula de referência (sem trial) do método professor, não um trial.'''
        sibling = {key: value for key, value in cell.items() if key not in ("trial", "stage1_rank")}
        return {**sibling, "method": teacher_method}

    # ------------------------------------------------------------------ lifecycle

    def start_cell(self, cell: Dict[str, Any]) -> None:
        self._active_cell = dict(cell)

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, experiment) -> pd.DataFrame:
        '''Por frame, na VALIDAÇÃO FIXA, com os pesos do melhor checkpoint; o ponto vai nas colunas `hp_*`.'''
        module = experiment.lit_module
        if experiment.best_checkpoint_path:
            checkpoint = torch.load(experiment.best_checkpoint_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        self._eval_costs = {}
        per_sample = self._timed_eval(module, experiment.data_module.val_dataloader(), "val")
        optimizer = experiment.config.optimizer.params
        per_sample["lr"] = float(optimizer.lr)
        per_sample["weight_decay"] = float(optimizer.weight_decay)
        cell = getattr(self, "_active_cell", None)
        if cell is not None and cell.get("trial") is not None:
            for name, value in self.point(cell).items():
                per_sample[f"{HP_PREFIX}{name}"] = float(value)
        return per_sample


class E2TuningSweep(_E2TuningStage):
    '''
    Célula = (seed, arch, unlabeled_policy, trial, method); `trial` vem de `search_space.n_trials` e é
    None nas células de referência (métodos sem espaço). Ordem: referências, depois os trials (trial mais
    externo: um orçamento cortado deixa todos os métodos com o mesmo número de trials), e por último os
    métodos que herdam pesos da confirmação.
    '''

    def cells(self) -> List[Dict[str, Any]]:
        axes = self._base_axes()
        methods = [str(m) for m in axes.pop("method")]
        n_trials = int(self.config.search_space.n_trials)
        groups = [dict(zip(axes.keys(), values)) for values in _product(axes.values())]
        reference = [m for m in methods if not self.is_tuned(m)]
        first = [m for m in methods if self.is_tuned(m) and not self.inherits(m)]
        last = [m for m in methods if self.inherits(m)]
        cells = [{**group, "trial": None, "method": m} for group in groups for m in reference]
        for tuned in (first, last):
            cells += [{**group, "trial": trial, "method": m}
                      for group in groups for trial in range(n_trials) for m in tuned]
        return cells

    # ------------------------------------------------------------------ study

    @property
    def storage_path(self) -> Path:
        return self.sweep_dir / "optuna.journal"

    @cached_property
    def storage(self):
        return journal_storage(self.storage_path)

    def study_name(self, cell: Dict[str, Any]) -> str:
        return f"{self.family}-{cell['arch']}-{cell['method']}-{cell['unlabeled_policy']}-s{cell['seed']}"

    def study(self, cell: Dict[str, Any]):
        space = self.space(str(cell["method"]))
        pruner = make_pruner(_plain(self.config.optuna.pruner))
        return open_study(self.storage, self.study_name(cell), pruner, space.get("fixed_trials") or [])

    def trials_dataframe(self) -> pd.DataFrame:
        frames, seen = [], set()
        for cell in self.cells():
            if cell["trial"] is None or self.study_name(cell) in seen:
                continue
            seen.add(self.study_name(cell))
            frame = self.study(cell).trials_dataframe()
            if len(frame):
                frames.append(frame.assign(**{k: cell[k] for k in ("seed", "arch", "unlabeled_policy", "method")}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ inheritance

    @cached_property
    def confirm_plan(self) -> "E2TuningConfirmSweep":
        if not self.config.get("inherit_from"):
            raise KeyError("`inherit_from` (o YAML da confirmação) é obrigatório quando um método herda pesos.")
        from src.sweep import _resolve_sweep_config
        config = OmegaConf.load(str(_resolve_sweep_config(str(self.config.inherit_from))))
        return E2TuningConfirmSweep(config, log_dir=self.log_dir)

    def inherited(self, cell: Dict[str, Any]) -> Dict[str, float]:
        spec = self.space(str(cell["method"])).get("inherit") or {}
        values: Dict[str, float] = {}
        for param, source in spec.items():
            winner = self.confirm_plan.winner({**cell, "method": str(source)})
            if param not in winner["params"]:
                raise KeyError(f"A confirmação de '{source}' não tem o peso '{param}'.")
            values[param] = float(winner["params"][param])
        return values

    # ------------------------------------------------------------------ lifecycle

    def begin(self, fast_dev_run: bool | int = False) -> None:
        import optuna

        super().begin(fast_dev_run)
        self._trials: Dict[str, Any] = {}
        self._pruning: Dict[str, OptunaPruning] = {}
        if fast_dev_run:
            # Um fast_dev_run não grava na CSV; um trial aberto no journal ficaria RUNNING para sempre.
            self.storage = optuna.storages.InMemoryStorage()
            return
        self._reconcile()

    def _results(self) -> pd.DataFrame:
        frames = [pd.read_csv(path) for path in {self.tables_dir / self.results_csv_name(),
                                                  self.log_dir / self.family / self.results_csv_name()}
                  if path.exists()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _reconcile(self) -> None:
        '''Trials RUNNING cuja célula já está na CSV: o processo caiu entre gravar a CSV e o `tell`.'''
        from optuna.trial import TrialState

        results = self._results()
        if not len(results):
            return
        seen = set()
        for cell in self.cells():
            if cell["trial"] is None or self.study_name(cell) in seen:
                continue
            seen.add(self.study_name(cell))
            study = self.study(cell)
            for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                rows = results[results.run_id == self.run_id({**cell, "trial": frozen.number})]
                if len(rows):
                    self._tell(study, frozen.number, rows)

    @staticmethod
    def _tell(study, trial_number: int, per_sample: pd.DataFrame) -> None:
        from optuna.trial import TrialState

        if "pruned" in per_sample.columns and per_sample["pruned"].astype(bool).any():
            study.tell(trial_number, state=TrialState.PRUNED)
        else:
            study.tell(trial_number, objective_value(per_sample))

    def start_cell(self, cell: Dict[str, Any]) -> None:
        super().start_cell(cell)
        if cell.get("trial") is None:
            return
        if not hasattr(self, "_trials"):
            raise RuntimeError("`begin` não foi chamado: é o `Sweep.run` que abre a sessão do plano.")
        run_id = self.run_id(cell)
        # Herança resolvida ANTES de abrir o trial: sem a confirmação de origem, nada fica RUNNING no journal.
        inherited = self.inherited(cell)
        space = self.space(str(cell["method"]))
        trial = open_trial(self.study(cell), int(cell["trial"]), space.get("params") or {},
                           _plain(self.config.optuna.sampler))
        trial.set_user_attr("run_id", run_id)
        for param, value in inherited.items():
            trial.set_user_attr(f"inherited_{param}", value)
        self._trials[run_id] = trial
        self._inherited = getattr(self, "_inherited", {})
        self._inherited[run_id] = inherited
        self._pruning[run_id] = OptunaPruning(trial, monitor=str(self.config.optuna.monitor))

    def _trial(self, cell: Dict[str, Any]):
        run_id = self.run_id(cell)
        if run_id not in getattr(self, "_trials", {}):
            raise RuntimeError(f"O trial de {run_id} ainda não foi aberto: o ponto do TPE só existe "
                               "depois de `start_cell`.")
        return self._trials[run_id]

    def point(self, cell: Dict[str, Any]) -> Dict[str, float]:
        '''Herdados + o ponto do TPE (a ordem do dict não importa: o sorteio usa a do YAML).'''
        run_id = self.run_id(cell)
        trial = self._trial(cell)
        return {**self._inherited.get(run_id, {}), **{k: float(v) for k, v in trial.params.items()}}

    def callbacks(self, cell: Dict[str, Any]) -> List[Any]:
        if cell.get("trial") is None:
            return []
        return [self._pruning[self.run_id(cell)]]

    def evaluate(self, experiment) -> pd.DataFrame:
        per_sample = super().evaluate(experiment)
        per_sample["pruned"] = any(getattr(callback, "pruned", False)
                                   for callback in experiment.extra_callbacks)
        return per_sample

    def finish_cell(self, cell: Dict[str, Any], experiment, per_sample: pd.DataFrame) -> None:
        super().finish_cell(cell, experiment, per_sample)
        if cell.get("trial") is None:
            return
        trial = self._trial(cell)
        self._tell(trial.study, trial.number, per_sample)
        # Solta a célula (ver E1TuningSweep.finish_cell: o callback prende módulo/trainer na GPU).
        run_id = self.run_id(cell)
        self._pruning.pop(run_id, None)
        self._trials.pop(run_id, None)
        self._inherited.pop(run_id, None)


class E2TuningConfirmSweep(_E2TuningStage):
    '''
    Célula = (seed, arch, unlabeled_policy, stage1_rank, trial, method): o trial da busca naquela posição
    do ranking do método, retreinado do zero pelo cronograma completo. `stage1_rank` é o eixo mais
    externo. A etapa 1 do DTC-TRUS é a MESMA célula de referência da busca (lida do state.json da busca).
    '''

    @cached_property
    def source_config(self):
        from src.sweep import _resolve_sweep_config
        return OmegaConf.load(str(_resolve_sweep_config(str(self.config.source_sweep))))

    def search_config(self):
        return self.source_config

    def _source_csv(self, config, csv_key: str = "results_csv") -> Path:
        return self.log_dir / str(config.family) / str(config.get(csv_key, "results_per_sample.csv"))

    def candidates_for(self, cell: Dict[str, Any]) -> pd.DataFrame:
        '''As `top_k` do método na busca COMPLETA (todos os trials na CSV), sem as podadas.'''
        method = str(cell["method"])
        csv = self._source_csv(self.source_config)
        if not csv.exists():
            raise FileNotFoundError(f"{csv} não existe: rode a busca ({self.config.source_sweep}) antes.")
        results = pd.read_csv(csv)
        mask = ((results.method == method) & (results.arch == cell["arch"])
                & (results.unlabeled_policy == cell["unlabeled_policy"]) & (results.seed == cell["seed"])
                & results.trial.notna())
        results = results[mask]
        n_trials = int(self.source_config.search_space.n_trials)
        done = results.trial.nunique()
        if done < n_trials:
            raise RuntimeError(f"Busca de '{method}' incompleta: {done} de {n_trials} trials na CSV. "
                               "A confirmação só seleciona sobre a busca inteira.")
        ranked = rank_trials(results, by=["method"], extra_columns=hp_columns(results))
        top_k = int(self.config.top_k)
        if len(ranked) < top_k:
            raise RuntimeError(f"'{method}': menos de {top_k} trials completos (não podados).")
        ranked["trial"] = ranked["trial"].astype(int)
        return ranked[ranked["rank"] <= top_k].reset_index(drop=True)

    def cells(self) -> List[Dict[str, Any]]:
        axes = self._base_axes()
        methods = [str(m) for m in axes.pop("method")]
        untuned = [m for m in methods if not self.is_tuned(m)]
        if untuned:
            raise ValueError(f"A confirmação só tem métodos com espaço de busca; tire {untuned} de `axes.method`.")
        groups = [dict(zip(axes.keys(), values)) for values in _product(axes.values())]
        cells = []
        for rank in range(1, int(self.config.top_k) + 1):
            for group in groups:
                for method in methods:
                    candidates = self.candidates_for({**group, "method": method})
                    trial = int(candidates.loc[candidates["rank"] == rank, "trial"].iloc[0])
                    cells.append({**group, "stage1_rank": rank, "trial": trial, "method": method})
        return cells

    def point(self, cell: Dict[str, Any]) -> Dict[str, float]:
        candidates = self.candidates_for(cell)
        rows = candidates[candidates.trial == int(cell["trial"])]
        if len(rows) != 1:
            raise KeyError(f"trial {cell['trial']} não está entre os candidatos de '{cell['method']}'.")
        return point_from_row(rows.iloc[0])

    def teacher_state(self) -> Dict[str, Any]:
        '''O professor é a célula de referência da BUSCA: o mesmo para todos os trials e candidatos.'''
        family_dir = self.log_dir / str(self.source_config.family)
        state_dir = family_dir / "_sweep"
        cells: Dict[str, Any] = {}
        for path in [state_dir / "state.json"] + sorted(state_dir.glob("*/state.json")):
            try:
                for run_id, entry in json.loads(path.read_text()).get("cells", {}).items():
                    cells.setdefault(run_id, entry)
            except FileNotFoundError:
                continue
        return {"cells": cells}

    # ------------------------------------------------------------------ selection

    def winner(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        '''
        O vencedor da confirmação de um método: o maior Dice médio por paciente na validação entre as
        `top_k`. Exige as `top_k` na CSV -- um vencedor parcial mudaria quando as outras chegassem.
        '''
        method = str(cell["method"])
        csv = self._source_csv(self.config)
        if not csv.exists():
            raise FileNotFoundError(f"{csv} não existe: rode a confirmação ({self.name}) de '{method}' antes.")
        results = pd.read_csv(csv)
        results = results[(results.method == method) & (results.arch == cell["arch"])
                          & (results.unlabeled_policy == cell["unlabeled_policy"])
                          & (results.seed == cell["seed"])]
        expected = set(self.candidates_for(cell).trial.astype(int))
        found = set(results.trial.dropna().astype(int))
        if not expected <= found:
            raise RuntimeError(f"Confirmação de '{method}' incompleta: faltam os trials {sorted(expected - found)}.")
        results = results[results.trial.isin(expected)]
        ranked = rank_trials(results, by=["method"], completed_only=False, extra_columns=hp_columns(results))
        best = ranked.iloc[0]
        return {"params": point_from_row(best), "trial": int(best["trial"]), "val_dice_patient": float(best["dice"])}


def _product(values):
    from itertools import product
    return product(*[list(v) for v in values])


# ---------------------------------------------------------------------- freezing

def freeze_ssl_hparams(confirm_config_path: str | Path, path: str | Path, log_dir: Optional[str | Path] = None,
                       force: bool = False) -> Path:
    '''
    Grava `configs/frozen/e2-ssl-hparams.yaml` com o vencedor da confirmação de CADA método do espaço de
    busca (todos precisam estar confirmados). Commitá-lo é o congelamento: depois dele nenhum peso pode ser
    reajustado com resultados do E2/E2-CV.
    '''
    from datetime import datetime
    from src.sweep import _resolve_sweep_config

    path = resolve_project_path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} já existe -- os pesos já foram congelados. Use force=True só de propósito.")
    config = OmegaConf.load(str(_resolve_sweep_config(str(confirm_config_path))))
    plan = E2TuningConfirmSweep(config, log_dir=log_dir)
    spaces = plan.method_spaces()
    axes = plan._base_axes()
    axes.pop("method")
    groups = [dict(zip(axes.keys(), values)) for values in _product(axes.values())]
    if len(groups) != 1:
        raise ValueError(f"O congelamento espera uma única (seed, arch, política); a grade tem {groups}.")
    group = groups[0]
    hparams: Dict[str, Any] = {}
    for method, space in spaces.items():
        winner = plan.winner({**group, "method": method})
        entry = {"params": winner["params"], "trial": winner["trial"],
                 "val_dice_patient": winner["val_dice_patient"],
                 "tuned": sorted((space.get("params") or {}).keys())}
        if space.get("inherit"):
            entry["inherited"] = dict(space["inherit"])
        hparams[method] = entry
    document = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"{plan._source_csv(plan.config)} (busca: {plan._source_csv(plan.source_config)})",
        "selection": {"metric": "dice", "aggregation": "patient", "eval_set": "val",
                      "seed": group["seed"], "unlabeled_policy": group["unlabeled_policy"],
                      "train_folds": [int(f) for f in config.data.train_folds]},
        "hparams": {str(group["arch"]): hparams},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Gerado por src.sweeps.e2_tuning.freeze_ssl_hparams. Não editar à mão: é o registro do "
                    "congelamento.\n" + yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    return path
