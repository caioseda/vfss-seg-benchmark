'''
Tuning A.1 do protocolo do artigo: a busca (lr, weight_decay) com Optuna, a selecao e o congelamento.

O tuning tem duas etapas, cada uma um sweep (`src/sweeps/e1.py`):

  busca        TPE + MedianPruner, um study por arquitetura, trials curtos. Cada trial e uma celula
               comum de sweep: o ponto vem do study na hora em que a celula comeca (`open_trial`) e o
               resultado volta para ele quando ela termina. O pruning acontece DENTRO do treino, pelo
               callback `OptunaPruning`, que reporta a metrica de selecao a cada validacao.
  confirmacao  as `top_k` configuracoes de cada arquitetura treinadas pelo cronograma completo, sem
               pruning. A melhor na validacao e a congelada (`write_frozen_hparams`).

Por que nao `study.optimize`. O runner de sweep ja da retomada, CSV e MLflow por celula; um loop do
Optuna por cima teria de reimplementar os tres. E o TPE, diferente do `RandomSampler`, depende do
historico, entao os pontos nao podem mais ser pre-amostrados: o ask/tell manual e o meio-termo.

Reprodutibilidade. O estado do RNG de um sampler do Optuna nao e salvo no storage: um sampler recriado
com a mesma seed depois de uma queda repetiria os primeiros pontos. Por isso o sampler e recriado a cada
trial com seed `seed + numero do trial` (`make_sampler`): o ponto do trial n e funcao so de (seed, n,
historico do study), e uma retomada sorteia o mesmo que a execucao ininterrupta. Efeito colateral util:
na fase de aquecimento do TPE (sorteio aleatorio) todas as arquiteturas recebem os MESMOS pontos.
'''
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
import yaml
from pytorch_lightning.callbacks import Callback

from src.data.splits import resolve_project_path
from src.evaluation import aggregate_per_patient


def distributions(params: Dict[str, Dict]) -> Dict[str, Any]:
    '''`{nome: {low, high, log}}` -> `FloatDistribution` por hiperparametro. A ORDEM das chaves e a do sorteio.'''
    import optuna

    return {
        name: optuna.distributions.FloatDistribution(
            float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        for name, spec in params.items()
    }


def make_sampler(spec: Dict[str, Any], trial_number: int):
    '''`TPESampler` do trial `trial_number`, com seed `spec.seed + trial_number` (ver o docstring do modulo).'''
    import optuna

    with warnings.catch_warnings():
        # `multivariate` ainda e marcado como experimental no Optuna 4.x.
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(
            seed=int(spec.get("seed", 0)) + int(trial_number),
            n_startup_trials=int(spec.get("n_startup_trials", 10)),
            multivariate=bool(spec.get("multivariate", False)),
        )


def make_pruner(spec: Dict[str, Any]):
    import optuna

    return optuna.pruners.MedianPruner(
        n_startup_trials=int(spec.get("n_startup_trials", 5)),
        n_warmup_steps=int(spec.get("n_warmup_steps", 0)),
        interval_steps=int(spec.get("interval_steps", 1)),
    )


def journal_storage(path: Union[str, Path]):
    '''Storage em arquivo, so de acrescimo: sobrevive a um SIGKILL no meio de uma escrita e dispensa SQLAlchemy.'''
    import optuna
    from optuna.storages.journal import JournalFileBackend

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(JournalFileBackend(str(path)))


def open_study(storage, name: str, pruner, fixed_trials: Sequence[Dict[str, float]] = ()):
    '''
    Cria ou reabre o study `name`. `fixed_trials` so e enfileirado na criacao: reenfileirar a cada
    abertura duplicaria os pontos.
    '''
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                pruner=pruner, load_if_exists=True)
    if fixed_trials and not study.get_trials(deepcopy=False):
        for point in fixed_trials:
            study.enqueue_trial(dict(point))
    return study


def open_trial(study, number: int, params: Dict[str, Any], sampler_spec: Dict[str, Any]):
    '''
    O trial `number` do study, pronto para `report`/`tell`. Trials sao abertos em ordem, um por celula:

      - `number` ja existe e esta RUNNING: e uma retomada (a celula caiu no meio). Devolve o MESMO trial,
        com os mesmos parametros -- e o treino continua do `last.ckpt`.
      - `number` e o proximo: `study.ask` com o sampler deste trial.

    Um trial ja encerrado sem a celula estar na CSV e um estado que o runner nao produz (a CSV e gravada
    antes do `tell`), e por isso e recusado em vez de silenciosamente retreinado.
    '''
    import optuna
    from optuna.trial import TrialState

    study.sampler = make_sampler(sampler_spec, number)
    trials = study.get_trials(deepcopy=False)
    if number < len(trials) and trials[number].state != TrialState.WAITING:
        frozen = trials[number]
        if frozen.state != TrialState.RUNNING:
            raise RuntimeError(
                f"O trial {number} do study '{study.study_name}' ja esta {frozen.state.name}, mas a celula "
                "nao esta na CSV de resultados. Confira a CSV antes de retreinar."
            )
        return optuna.trial.Trial(study, frozen._trial_id)
    if number > len(trials):
        raise RuntimeError(f"Trial {number} pedido, mas o study '{study.study_name}' so tem {len(trials)}: "
                           "os trials sao abertos em ordem.")
    trial = study.ask(distributions(params))
    if trial.number != number:
        raise RuntimeError(f"study.ask devolveu o trial {trial.number}, esperado {number}.")
    return trial


def objective_value(per_sample: pd.DataFrame, metric: str = "dice") -> float:
    '''O valor que o study maximiza: a mesma regra de `rank_trials` (media por paciente, depois sobre as classes).'''
    return float(aggregate_per_patient(per_sample)[metric].mean())


class OptunaPruning(Callback):
    '''
    Reporta `monitor` ao trial a cada validacao e, se o pruner mandar, para o treino.

    Para com `trainer.should_stop` em vez de levantar `optuna.TrialPruned`: um erro dentro do `fit`
    marcaria o run como falho e o sweep o retomaria para sempre. Parado assim, o run termina normalmente,
    o melhor checkpoint ate ali e avaliado, e o sweep le `pruned` para fazer o `tell` como PRUNED.
    '''

    def __init__(self, trial, monitor: str = "val/dice_patient"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.pruned = False

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        # Numa retomada, os passos ja reportados sao ignorados pelo Optuna (com um aviso).
        self.trial.report(float(value), step=int(trainer.global_step))
        if self.trial.should_prune():
            self.pruned = True
            trainer.should_stop = True


def rank_trials(per_sample_val: pd.DataFrame, by: Sequence[str] = ("arch",), trial_column: str = "trial",
                metric: str = "dice", completed_only: bool = True) -> pd.DataFrame:
    '''
    Um valor por trial, na regra de selecao do protocolo: Dice medio POR PACIENTE na validacao fixa
    (`aggregate_per_patient`), media sobre as classes reportadas e, se houver mais de uma, sobre as seeds.
    Devolve a tabela ordenada dentro de cada `by`, com a coluna `rank` (1 = melhor).

    `completed_only` descarta os trials podados (coluna `pruned`): o valor deles e de um treino
    interrompido, e nao concorre com os que terminaram.
    '''
    if completed_only and "pruned" in per_sample_val.columns:
        per_sample_val = per_sample_val[~per_sample_val["pruned"].astype(bool)]
    keys = list(by) + [trial_column]
    extra = [column for column in ("lr", "weight_decay") if column in per_sample_val.columns]
    seed = ["seed"] if "seed" in per_sample_val.columns else []
    per_patient = aggregate_per_patient(per_sample_val, group_columns=keys + extra + seed)
    per_trial = (
        per_patient.groupby(keys + extra, dropna=False)[metric].mean()
        .reset_index()
        .sort_values(list(by) + [metric], ascending=[True] * len(by) + [False], kind="mergesort")
    )
    per_trial["rank"] = per_trial.groupby(list(by)).cumcount() + 1
    return per_trial.reset_index(drop=True)


def write_frozen_hparams(best: pd.DataFrame, path: Union[str, Path], source: str,
                         by: str = "arch", metric: str = "dice", force: bool = False) -> Path:
    '''
    Grava o arquivo que o CV (`E1CVSweep`) le. Commita-lo E o congelamento: depois dele nenhum
    hiperparametro pode ser reajustado com resultados de fold avaliado ou de teste.

    `best` tem uma linha por `by` (por exemplo `rank_trials(...)` filtrado em `rank == 1`).
    '''
    path = resolve_project_path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} ja existe -- os hiperparametros ja foram congelados. Use force=True so de proposito.")
    if best[by].duplicated().any():
        raise ValueError(f"`best` deve ter uma linha por `{by}`; duplicados: {best[by][best[by].duplicated()].tolist()}")

    hparams = {
        str(row[by]): {
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "trial": int(row["trial"]),
            f"val_{metric}_patient": float(row[metric]),
        }
        for _, row in best.iterrows()
    }
    document = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "selection": {"metric": metric, "aggregation": "patient", "eval_set": "val"},
        "hparams": hparams,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Gerado por src.tuning.write_frozen_hparams. Nao editar a mao: e o registro do congelamento.\n"
                    + yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    return path
