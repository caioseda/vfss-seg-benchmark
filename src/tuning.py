'''
Tuning A.1 do protocolo do artigo: os pontos (lr, weight_decay) de cada trial, a escolha do melhor e o
congelamento.

Por que os pontos sao PRE-amostrados em vez de um `study.optimize` em loop. O protocolo fixa o Optuna
com `RandomSampler` e seed. Esse sampler nao olha para os resultados anteriores, entao a sequencia de
pontos que ele produz e a mesma pedida de uma vez ou trial a trial. Pre-amostrar deixa cada trial ser uma
celula comum de sweep (`src/sweeps/e1.py`), com a retomada, a CSV e o MLflow que o runner ja tem. Tambem
torna a retomada exata: um `study.optimize` reiniciado recriaria o sampler a partir da seed e repetiria os
primeiros pontos. O historico da busca e a CSV de resultados mais `_sweep/state.json`.
'''
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd
import yaml

from src.data.splits import resolve_project_path
from src.evaluation import aggregate_per_patient


def sample_random_trials(params: Dict[str, Dict], n_trials: int, seed: int) -> List[Dict[str, float]]:
    '''
    `n_trials` pontos do `RandomSampler(seed)` do Optuna, um dict `{nome: valor}` por trial.

    `params` mapeia cada hiperparametro para `{low, high, log}` (`FloatDistribution`). A ORDEM das chaves
    faz parte da semente: o sampler sorteia os parametros nessa ordem, entao reordenar o YAML muda os
    pontos.
    '''
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    distributions = {
        name: optuna.distributions.FloatDistribution(
            float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        for name, spec in params.items()
    }
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=int(seed)))
    return [dict(study.ask(distributions).params) for _ in range(int(n_trials))]


def rank_trials(per_sample_val: pd.DataFrame, by: Sequence[str] = ("arch",), trial_column: str = "trial",
                metric: str = "dice") -> pd.DataFrame:
    '''
    Um valor por trial, na regra de selecao do protocolo: Dice medio POR PACIENTE na validacao fixa
    (`aggregate_per_patient`), media sobre as classes reportadas e, se houver mais de uma, sobre as seeds.
    Devolve a tabela ordenada dentro de cada `by`, com a coluna `rank` (1 = melhor).
    '''
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
