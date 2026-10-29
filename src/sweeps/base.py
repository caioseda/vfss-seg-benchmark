'''
What a sweep is: a grid of cells, plus how each cell becomes an `Experiment`.

This exists because the grid currently lives in notebook constants -- `SEEDS`, `METHODS`,
`LABEL_FRACTIONS`, `MAX_STEPS` -- and `build_overrides` / `run_id` / `train_and_evaluate` are
near-identical copies in five notebooks. That has two costs beyond the duplication: the experiment
is not reproducible from the repository alone (you need the notebook, in the state it was in), and
the runner cannot survive the notebook's kernel.

A sweep is therefore split in two:
  configs/sweep/<name>.yaml   the knobs, versioned and diffable
  src/sweeps/<name>.py        the logic, importable and testable

`SweepPlan` is the contract between them. Subclasses override only what is specific to their grid;
the cartesian product, the run naming and the per-sample evaluation have working defaults.
'''
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.evaluation import evaluate_per_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SweepPlan:
    def __init__(self, config: DictConfig, log_dir: Optional[str | Path] = None):
        self.config = config
        self.family: str = config.family
        self.name: str = config.get("name", config.family)
        # The same root the `Sweep` writes under. Only plans that keep state of their own (an Optuna
        # study) or read another family's results need it.
        self.log_dir = Path(log_dir).resolve() if log_dir else PROJECT_ROOT / "logs"

    @property
    def sweep_dir(self) -> Path:
        return self.log_dir / self.family / "_sweep"

    # ------------------------------------------------------------------ the grid

    def axes(self) -> Dict[str, List[Any]]:
        '''The varying dimensions, in declared order -- which fixes the order cells run in.'''
        return {key: list(values) for key, values in
                OmegaConf.to_container(self.config.axes, resolve=True).items()}

    def cells(self) -> List[Dict[str, Any]]:
        axes = self.axes()
        return [dict(zip(axes.keys(), combination)) for combination in product(*axes.values())]

    # ------------------------------------------------------------------ per cell

    def run_id(self, cell: Dict[str, Any]) -> str:
        '''Stable identity of a cell. It is the key in the results CSV, so it must never drift.'''
        raise NotImplementedError

    def run_name(self, cell: Dict[str, Any]) -> str:
        '''Directory name under `logs/<family>/runs/`. Defaults to `<family>-<run_id>`.'''
        return f"{self.family}-{self.run_id(cell)}"

    def experiment_config(self, cell: Dict[str, Any]) -> str:
        '''
        Which experiment YAML this cell starts from. The default reads a `base_configs` mapping
        keyed by the cell's `method`, which is how every grid in this repo is shaped.
        '''
        base = self.config.base_configs
        method = cell["method"]
        if method not in base:
            raise KeyError(f"`base_configs` nao tem entrada para method='{method}'")
        return str(base[method])

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        '''Partial config merged over the YAML (`OmegaConf.merge`, so nested keys merge).'''
        raise NotImplementedError

    # ------------------------------------------------------------------ lifecycle hooks
    #
    # No-ops for a fixed grid. A plan whose cells are decided while the sweep runs (an Optuna study
    # asks for the next point only after the previous one reported) hooks in here. The `Sweep` calls
    # them in this order: `begin` once, then per trained cell `start_cell` -> `overrides`/`callbacks`
    # -> training -> `evaluate` -> rows appended to the CSV -> `finish_cell`.

    def begin(self, fast_dev_run: bool | int = False) -> None:
        '''Once per `Sweep.run`, before any cell. Under `fast_dev_run` nothing it sets up may persist.'''

    def start_cell(self, cell: Dict[str, Any]) -> None:
        '''Right before the cell's `Experiment` is built -- including when the cell is being resumed.'''

    def callbacks(self, cell: Dict[str, Any]) -> List[Any]:
        '''Lightning callbacks added to this cell's trainer, on top of the ones its YAML declares.'''
        return []

    def finish_cell(self, cell: Dict[str, Any], experiment, per_sample: pd.DataFrame) -> None:
        '''After the cell's rows are in the CSV. A crash before this runs must be recoverable from the CSV.'''

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, experiment) -> pd.DataFrame:
        '''
        Per-sample test metrics, the rows that land in the family's results CSV.

        Mirrors the notebooks' `train_and_evaluate` exactly, including the Mean Teacher detail: it
        reports through the teacher, whose EMA weights are the model being selected, while the other
        wrappers report through the model itself.
        '''
        evaluation = self.config.evaluation
        forward_fn = getattr(experiment.lit_module, "_eval_forward", None)
        return evaluate_per_class(
            experiment.lit_module,
            experiment.data_module.test_dataloader(),
            num_classes=int(evaluation.num_classes),
            class_ids={int(k): v for k, v in
                       OmegaConf.to_container(evaluation.class_ids, resolve=True).items()},
            forward_fn=forward_fn,
        )

    def results_csv_name(self) -> str:
        return str(self.config.get("results_csv", "results_per_sample.csv"))
