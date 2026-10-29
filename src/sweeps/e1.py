'''
E1 -- arquiteturas supervisionadas sob o protocolo do artigo.

Tres grades com a mesma base (`configs/experiment/e1/`):

  E1TuningSweep   tuning A.1, etapa 1 (busca): TPE + MedianPruner do Optuna, um study por arquitetura,
                  trials curtos. Treina so no fold 0 do desenvolvimento, seleciona pela validacao fixa
                  e NAO toca em teste nenhum. Cada celula e um trial (`src/tuning.py`).
  E1ConfirmSweep  tuning A.1, etapa 2 (confirmacao): as `top_k` configuracoes de cada arquitetura na
                  busca, retreinadas pelo cronograma completo e sem pruning. A melhor e a congelada.
  E1CVSweep       Etapa B: CV 5-fold por paciente com os hiperparametros congelados; o fold retido e o
                  split 'test' do run, a validacao fixa segue escolhendo o checkpoint.

O que difere do X1 e nao pode virar YAML estatico: o split vem da tabela congelada
(`split_mode: protocol`), e a validacao e por PASSOS (`val_check_interval`) em vez de por epocas. A
epoca muda de tamanho de fold para fold (os folds tem 117 a 127 frames, o tuning treina com um fold so),
entao um `check_val_every_n_epoch` fixo validaria em passos diferentes por celula.
'''
from functools import cached_property
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.data.splits import resolve_project_path
from src.evaluation import evaluate_per_class
from src.sweeps.base import SweepPlan
from src.tuning import (
    OptunaPruning,
    journal_storage,
    make_pruner,
    objective_value,
    open_study,
    open_trial,
    rank_trials,
)


class _E1Sweep(SweepPlan):

    def experiment_config(self, cell: Dict[str, Any]) -> str:
        '''A arquitetura e o eixo que escolhe o YAML base, nao `method` (o metodo e sempre supervisionado).'''
        base = self.config.base_configs
        arch = cell["arch"]
        if arch not in base:
            raise KeyError(f"`base_configs` nao tem entrada para arch='{arch}'")
        return str(base[arch])

    def _protocol_overrides(self, cell: Dict[str, Any], lr: float, weight_decay: float,
                            train_folds: Optional[List[int]], cv_fold: Optional[int]) -> Dict[str, Any]:
        training = self.config.training
        return {
            "name": self.run_name(cell),
            "trainer": {"params": {
                "max_steps": int(training.max_steps),
                "val_check_interval": int(training.val_every_n_steps),
                "check_val_every_n_epoch": None,   # exigido pelo Lightning para validar por passos
            }},
            "lr_scheduler": {"params": {"T_max": int(training.max_steps)}},
            "optimizer": {"params": {"lr": float(lr), "weight_decay": float(weight_decay)}},
            "data": {"params": {
                "batch_size": int(training.batch_size),
                "shared_dataset_params": {
                    "split_mode": "protocol",
                    "split_file": str(self.config.data.split_file),
                    "train_folds": train_folds,
                    "cv_fold": cv_fold,
                },
            }},
        }

    def _class_ids(self) -> Dict[int, str]:
        return {int(k): v for k, v in
                OmegaConf.to_container(self.config.evaluation.class_ids, resolve=True).items()}


class _E1TuningStage(_E1Sweep):
    '''O que as duas etapas do tuning compartilham: treino em `data.train_folds`, sem teste, avaliacao na validacao.'''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['arch']}-t{int(cell['trial']):02d}-s{cell['seed']}"

    def _tuning_overrides(self, cell: Dict[str, Any], lr: float, weight_decay: float) -> Dict[str, Any]:
        overrides = self._protocol_overrides(
            cell, lr, weight_decay,
            train_folds=[int(f) for f in self.config.data.train_folds], cv_fold=None,
        )
        # Sem split de teste: o DataModule descarta splits None e `Experiment.run` pula `trainer.test`.
        overrides["data"]["params"]["test"] = None
        return overrides

    def evaluate(self, experiment) -> pd.DataFrame:
        '''
        Por frame, na VALIDACAO FIXA, com os pesos do melhor checkpoint. Sem `trainer.test`, nada mais
        carrega esses pesos no modulo -- no fim do fit ele esta com os pesos do ultimo passo.
        '''
        module = experiment.lit_module
        if experiment.best_checkpoint_path:
            checkpoint = torch.load(experiment.best_checkpoint_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        per_sample = evaluate_per_class(
            module, experiment.data_module.val_dataloader(),
            num_classes=int(self.config.evaluation.num_classes), class_ids=self._class_ids(),
            forward_fn=getattr(module, "_eval_forward", None),
        )
        optimizer = experiment.config.optimizer.params
        per_sample["eval_set"] = "val"
        per_sample["lr"] = float(optimizer.lr)
        per_sample["weight_decay"] = float(optimizer.weight_decay)
        return per_sample


class E1TuningSweep(_E1TuningStage):
    '''
    Celula = (trial, arch, seed), com um study do Optuna por (arch, seed). `trial` nao e declarado em
    `axes`: vem de `search_space.n_trials`, e e o eixo mais externo -- se o orcamento for cortado, todas as
    arquiteturas ficam com o mesmo numero de trials. O trial n da celula e o trial n do study.

    O ponto de um trial so existe depois que os anteriores do mesmo study terminaram, entao `overrides`
    so funciona depois de `start_cell`. O study vive em `_sweep/optuna.journal` da familia, ao lado do
    `state.json`; a CSV de resultados continua sendo a fonte de verdade de "pronto", e `begin` devolve ao
    study qualquer trial que a CSV ja tenha e o study ainda nao (queda entre a CSV e o `tell`).
    '''

    def axes(self) -> Dict[str, List[Any]]:
        axes = super().axes()
        if "trial" in axes:
            raise ValueError("`trial` vem de `search_space.n_trials`; nao o declare em `axes`.")
        trials = list(range(int(self.config.search_space.n_trials)))
        return {"seed": axes.pop("seed"), "trial": trials, **axes}

    # ------------------------------------------------------------------ study

    @property
    def storage_path(self):
        return self.sweep_dir / "optuna.journal"

    @cached_property
    def storage(self):
        return journal_storage(self.storage_path)

    def study_name(self, arch: str, seed: int) -> str:
        return f"{self.family}-{arch}-s{seed}"

    def study(self, arch: str, seed: int):
        space = self.config.search_space
        fixed = OmegaConf.to_container(space.fixed_trials, resolve=True) if space.get("fixed_trials") else []
        params = OmegaConf.to_container(space.params, resolve=True)
        for point in fixed:
            if set(point) != set(params):
                raise ValueError(f"fixed_trials {point} deve definir exatamente {sorted(params)}")
        pruner = make_pruner(OmegaConf.to_container(self.config.optuna.pruner, resolve=True))
        return open_study(self.storage, self.study_name(arch, seed), pruner, fixed)

    def trials_dataframe(self) -> pd.DataFrame:
        '''Todos os trials de todos os studies, com estado e valor -- o historico da busca.'''
        frames = []
        for seed in self.axes()["seed"]:
            for arch in self.axes()["arch"]:
                frame = self.study(arch, seed).trials_dataframe()
                if len(frame):
                    frames.append(frame.assign(arch=arch, seed=seed))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ lifecycle

    def begin(self, fast_dev_run: bool | int = False) -> None:
        import optuna

        self._trials: Dict[str, Any] = {}
        self._pruning: Dict[str, OptunaPruning] = {}
        if fast_dev_run:
            # Um fast_dev_run nao grava na CSV; um trial aberto no journal ficaria RUNNING para sempre.
            self.storage = optuna.storages.InMemoryStorage()
            return
        self._reconcile()

    def _reconcile(self) -> None:
        '''Trials RUNNING cuja celula ja esta na CSV: o processo caiu entre gravar a CSV e o `tell`.'''
        from optuna.trial import TrialState

        csv = self.log_dir / self.family / self.results_csv_name()
        if not csv.exists():
            return
        results = pd.read_csv(csv)
        for seed in self.axes()["seed"]:
            for arch in self.axes()["arch"]:
                study = self.study(arch, seed)
                for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                    rows = results[results.run_id == self.run_id({"arch": arch, "trial": frozen.number,
                                                                   "seed": seed})]
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
        if not hasattr(self, "_trials"):
            raise RuntimeError("`begin` nao foi chamado: e o `Sweep.run` que abre a sessao do plano.")
        run_id = self.run_id(cell)
        trial = open_trial(self.study(str(cell["arch"]), int(cell["seed"])), int(cell["trial"]),
                           OmegaConf.to_container(self.config.search_space.params, resolve=True),
                           OmegaConf.to_container(self.config.optuna.sampler, resolve=True))
        trial.set_user_attr("run_id", run_id)
        self._trials[run_id] = trial
        self._pruning[run_id] = OptunaPruning(trial, monitor=str(self.config.optuna.monitor))

    def _trial(self, cell: Dict[str, Any]):
        run_id = self.run_id(cell)
        if run_id not in getattr(self, "_trials", {}):
            raise RuntimeError(f"O trial de {run_id} ainda nao foi aberto: o ponto do TPE so existe "
                               "depois de `start_cell`.")
        return self._trials[run_id]

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        params = self._trial(cell).params
        return self._tuning_overrides(cell, params["lr"], params["weight_decay"])

    def callbacks(self, cell: Dict[str, Any]) -> List[Any]:
        return [self._pruning[self.run_id(cell)]]

    def evaluate(self, experiment) -> pd.DataFrame:
        per_sample = super().evaluate(experiment)
        per_sample["pruned"] = any(getattr(callback, "pruned", False)
                                   for callback in experiment.extra_callbacks)
        return per_sample

    def finish_cell(self, cell: Dict[str, Any], experiment, per_sample: pd.DataFrame) -> None:
        trial = self._trial(cell)
        self._tell(trial.study, trial.number, per_sample)
        # Solta a celula. O Trainer faz `callback.log = pl_module.log` em todo callback, entao o
        # `OptunaPruning` guardado aqui prendia o modulo -- e por ele o trainer, o datamodule, os
        # pesos e o estado do otimizador na GPU -- de TODA celula ja treinada. A busca de 2026-10-27
        # chegou a ~19 GB de GPU e 14 GB de RSS em 57 celulas, ficando ~3x mais lenta no caminho.
        run_id = self.run_id(cell)
        self._pruning.pop(run_id, None)
        self._trials.pop(run_id, None)


class E1ConfirmSweep(_E1TuningStage):
    '''
    Celula = (stage1_rank, arch, seed) -> o trial da busca nessa posicao. `stage1_rank` e o eixo mais
    externo pelo mesmo motivo do `trial` na busca. O run_id carrega o numero do trial, nao o rank: e a
    identidade da configuracao.

    Os candidatos so existem com a busca COMPLETA para a arquitetura (todos os trials na CSV): um ranking
    parcial mudaria quando os trials que faltam chegassem, e com ele a grade.
    '''

    @cached_property
    def source_config(self):
        from src.sweep import _resolve_sweep_config

        return OmegaConf.load(str(_resolve_sweep_config(str(self.config.source_sweep))))

    @cached_property
    def candidates(self) -> pd.DataFrame:
        source = self.source_config
        csv = self.log_dir / str(source.family) / str(source.get("results_csv", "results_per_sample.csv"))
        if not csv.exists():
            raise FileNotFoundError(f"{csv} nao existe: rode a busca ({self.config.source_sweep}) antes.")
        results = pd.read_csv(csv)

        n_trials = int(source.search_space.n_trials)
        archs = [str(arch) for arch in self.axes()["arch"]]
        done = results.groupby("arch").trial.nunique().reindex(archs, fill_value=0)
        missing = done[done < n_trials]
        if len(missing):
            raise RuntimeError(f"Busca incompleta (trials na CSV, de {n_trials}): {missing.to_dict()}. "
                               "A confirmacao so seleciona sobre a busca inteira.")

        ranked = rank_trials(results[results.arch.isin(archs)], by=["arch"])
        top_k = int(self.config.top_k)
        short = ranked.groupby("arch").size().reindex(archs, fill_value=0)
        if (short < top_k).any():
            raise RuntimeError(f"Menos de {top_k} trials completos (nao podados): {short[short < top_k].to_dict()}")
        return ranked[ranked["rank"] <= top_k].reset_index(drop=True)

    def cells(self) -> List[Dict[str, Any]]:
        axes = self.axes()
        candidates = self.candidates.set_index(["arch", "rank"])
        return [
            {"seed": seed, "stage1_rank": rank, "arch": arch,
             "trial": int(candidates.loc[(arch, rank), "trial"])}
            for seed in axes["seed"]
            for rank in range(1, int(self.config.top_k) + 1)
            for arch in axes["arch"]
        ]

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        rows = self.candidates[(self.candidates.arch == cell["arch"]) & (self.candidates.trial == int(cell["trial"]))]
        if len(rows) != 1:
            raise KeyError(f"trial {cell['trial']} nao esta entre os candidatos de arch='{cell['arch']}'.")
        row = rows.iloc[0]
        return self._tuning_overrides(cell, row["lr"], row["weight_decay"])


class E1CVSweep(_E1Sweep):
    '''
    Celula = (fold, arch, seed). O fold e o eixo mais externo: se a grade for cortada, as arquiteturas
    concluidas continuam pareadas fold a fold.
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['arch']}-f{int(cell['fold'])}-s{cell['seed']}"

    @cached_property
    def frozen_hparams(self) -> Dict[str, Dict[str, Any]]:
        '''Lido sob demanda: a grade pode ser listada antes do congelamento, so nao pode ser treinada.'''
        path = resolve_project_path(str(self.config.hparams_file))
        if not path.exists():
            raise FileNotFoundError(
                f"{path} nao existe: os hiperparametros ainda nao foram congelados. Rode o tuning "
                "(configs/sweep/e1-tuning-search.yaml, depois e1-tuning-confirm.yaml) e "
                "`src.tuning.write_frozen_hparams` antes do CV."
            )
        return yaml.safe_load(path.read_text())["hparams"]

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        arch = str(cell["arch"])
        if arch not in self.frozen_hparams:
            raise KeyError(f"{self.config.hparams_file} nao tem hiperparametros para arch='{arch}'.")
        hparams = self.frozen_hparams[arch]
        return self._protocol_overrides(cell, hparams["lr"], hparams["weight_decay"],
                                        train_folds=None, cv_fold=int(cell["fold"]))

    def evaluate(self, experiment) -> pd.DataFrame:
        '''O 'test' deste run e o fold retido do CV -- nunca o teste INCA.'''
        per_sample = super().evaluate(experiment)
        per_sample["eval_set"] = "cv"
        return per_sample
