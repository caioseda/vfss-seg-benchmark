from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.cli import get_cli_args

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.callbacks import OverallTrainingProgressBar, build_callbacks
from src.models import LitWrapper
from src.utils import (
    instantiate_from_config,
    get_obj_from_str,
    _to_scalar,
    _resolve_config_path,
    _print_metric_block,
    capture_console_output,
)

class Experiment:
    def __init__(self, 
                 config_path: str, 
                 benchmark_dir: Optional[str | Path] = None,
                 experiment_id: Optional[str] = None,
                 overrides: Optional[Union[Dict[str, Any], DictConfig]] = None,
                 log_dir: Optional[str | Path] = None,
            ):
        '''
        Args:
            config_path: path to the experiment YAML.
            benchmark_dir / experiment_id: set when running as part of a `Benchmark`.
            overrides: a partial config merged over the loaded YAML (`OmegaConf.merge`, so nested
                keys are merged, not replaced). This is how a notebook sweeps one knob -- e.g.
                `{"data": {"params": {"shared_dataset_params": {"label_fraction": 0.1}}}}` -- without
                writing a temporary YAML per cell of the grid.
            log_dir: root directory for run outputs. Defaults to `logs/` **relative to the current
                working directory** -- pass an absolute path when the caller's cwd is not the repo
                root (a notebook runs from its own directory, and would otherwise scatter
                checkpoints and TensorBoard logs next to itself).
        '''
        self.config_path = config_path
        self.config = self._load_config(config_path)
        if overrides:
            self.config = OmegaConf.merge(self.config, OmegaConf.create(overrides))
        config_filename = Path(self.config_path).stem
        self.experiment_name = self.config.get("name", self.config.get("experiment_name", config_filename))
        
        self.running_as_benchmark = bool(benchmark_dir or experiment_id)
        if self.running_as_benchmark:
            self.experiment_dirname = f"{experiment_id}-{self.experiment_name}"
        else:
            self.experiment_dirname = self.experiment_name 

        self.parent_dir = benchmark_dir if self.running_as_benchmark else (log_dir or "logs")
        self.parent_dir = Path(self.parent_dir)

        # Populated by `run`, so a caller (the X1 notebook) can evaluate the trained model itself.
        self.lit_module: Optional[pl.LightningModule] = None
        self.data_module: Optional[pl.LightningDataModule] = None
        self.best_checkpoint_path: Optional[str] = None


    def setup_exeperiment_run_dir(self) -> None:
        if self.running_as_benchmark:
            self.experiment_run_dir = self.parent_dir / self.experiment_dirname
        else:
            self._timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
            self.experiment_run_dir = self.parent_dir / self.experiment_dirname / self._timestamp
        self.experiment_run_dir.mkdir(parents=True, exist_ok=True)
        

    def _load_config(self, cfg_path: str) -> Dict[str, Any]:
        resolved_path = _resolve_config_path(cfg_path)
        config = OmegaConf.load(str(resolved_path))

        _required_fields = ['model', 'optimizer', 'data', 'trainer']
        for field in _required_fields:
            if field not in config:
                raise ValueError(f"Missing required field `{field}` in {config}")
        
        assert config.type == "experiment", f"Config type must be 'experiment', got '{config.type}' in {config}"
        return config


    def setup_loggers(self) -> List[Any]:
        '''
        Run loggers. TensorBoard + CSV always, so every run leaves a self-contained record on disk
        next to its checkpoints; an optional `logging.loggers` block in the YAML appends others
        (MLflow, W&B, ...) without touching this file:

            logging:
              loggers:
                - target: pytorch_lightning.loggers.MLFlowLogger
                  params:
                    experiment_name: x1
                    tracking_uri: file:./mlruns

        The CSV logger is what makes a sweep analysable offline: `metrics.csv` per run reads
        straight into pandas, which TensorBoard event files do not.
        '''
        save_dir = str(self.experiment_run_dir)
        loggers: List[Any] = [
            TensorBoardLogger(save_dir=save_dir, name="", version="tensorboard"),
            CSVLogger(save_dir=save_dir, name="", version="csv"),
        ]

        extra_loggers = self.config.get("logging", {}).get("loggers", []) or []
        for logger_cfg in extra_loggers:
            loggers.append(instantiate_from_config(logger_cfg))
        return loggers

    @staticmethod
    def _flatten_config(config: Any, prefix: str = "") -> Dict[str, Any]:
        '''Flatten the resolved config into `a/b/c: value` pairs a tracker can index and compare runs by.'''
        flat: Dict[str, Any] = {}
        if isinstance(config, dict):
            for key, value in config.items():
                flat.update(Experiment._flatten_config(value, f"{prefix}{key}/"))
        elif isinstance(config, (list, tuple)):
            # Loggers reject nested containers; a list is only ever read, never queried on.
            flat[prefix.rstrip("/")] = str(list(config))
        else:
            flat[prefix.rstrip("/")] = config
        return flat

    def log_run_hyperparameters(self, loggers: List[Any], seed: int) -> None:
        '''
        Record the whole resolved config as searchable hyperparameters.

        Without this a sweep produces N directories of curves with no machine-readable record of
        *what varied* -- `label_fraction`, the method, the seed and the consistency schedule all live
        only in the YAML. Logging them makes runs comparable inside the tracker instead of by
        cross-referencing paths by hand.
        '''
        hyperparameters = self._flatten_config(OmegaConf.to_container(self.config, resolve=True))
        hyperparameters["seed"] = seed
        for logger in loggers:
            try:
                logger.log_hyperparams(hyperparameters)
            except Exception as error:  # a tracker being picky must not kill the run
                print(f"Could not log hyperparameters to {type(logger).__name__}: {error}")

    def _build_lit_module(self) -> pl.LightningModule:
        '''
        Instantiate the LightningModule. An optional `lit_module:` block (`target`/`params`) selects
        a subclass -- that is how the X1 configs swap in the semi-supervised wrappers from
        `src.models.ssl`. Without the block the behaviour is the historical `LitWrapper`, so the
        existing configs are unaffected.
        '''
        lit_module_cfg = self.config.get("lit_module")
        lit_module_cls = LitWrapper
        extra_params: Dict[str, Any] = {}
        if lit_module_cfg is not None:
            lit_module_cls = get_obj_from_str(lit_module_cfg["target"])
            extra_params = dict(lit_module_cfg.get("params", {}) or {})

        return lit_module_cls(
            model_cfg=self.config.model,
            optimizer_cfg=self.config.optimizer,
            lr_scheduler_cfg=self.config.get("lr_scheduler"),
            **extra_params,
        )

    def _build_checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        '''
        Model selection on validation, configured by an optional `trainer.checkpoint` block.

        Without it the reported test numbers come from the *last* epoch, which is unfair precisely
        in the low-label regimes that overfit hardest -- exactly the regimes X1 is measuring.
        '''
        checkpoint_cfg = self.config.trainer.get("checkpoint")
        if checkpoint_cfg is None:
            return None
        params = OmegaConf.to_container(checkpoint_cfg, resolve=True) if isinstance(checkpoint_cfg, DictConfig) else dict(checkpoint_cfg)
        params.setdefault("dirpath", str(self.experiment_run_dir / "checkpoints"))
        return ModelCheckpoint(**params)

    def run(self, fast_dev_run: bool | int = False, seed=42) -> Dict[str, Any]:
        pl.seed_everything(seed)
        self.setup_exeperiment_run_dir()
        self.console_log_path = self.experiment_run_dir / "console.log"
        
        loggers = self.setup_loggers()
        with capture_console_output(self.console_log_path):
            print(f"Loaded configuration: {self.config_path}")
            print(f"Console logs: {self.console_log_path}")
            print(f"Seed: {seed}")
            self.log_run_hyperparameters(loggers, seed)

            model = self._build_lit_module()
            data_module = instantiate_from_config(self.config.data)
            self.lit_module = model
            self.data_module = data_module

            callbacks_cfg = self.config.trainer.get("callbacks", [])
            callbacks = build_callbacks(callbacks_cfg)
            if not any(isinstance(callback, OverallTrainingProgressBar) for callback in callbacks):
                callbacks.append(OverallTrainingProgressBar())

            checkpoint_callback = self._build_checkpoint_callback()
            if checkpoint_callback is not None:
                callbacks.append(checkpoint_callback)
            
            trainer_params = OmegaConf.to_container(self.config["trainer"]["params"], resolve=True)
            if fast_dev_run:
                trainer_params["fast_dev_run"] = fast_dev_run
                print(f"fast_dev_run enabled via CLI override: {fast_dev_run}")

            trainer = pl.Trainer(
                logger=loggers,
                callbacks=callbacks,
                **trainer_params,
            )

            trainer.fit(model, data_module)

            fit_metrics = {
                k: v
                for k, v in trainer.callback_metrics.items()
                if k.startswith("train/") or k.startswith("val/")
            }

            # `fast_dev_run` disables checkpointing entirely, so there is no "best" to restore.
            ckpt_path = None
            if checkpoint_callback is not None and not fast_dev_run and checkpoint_callback.best_model_path:
                ckpt_path = "best"
                self.best_checkpoint_path = checkpoint_callback.best_model_path
                print(f"Testing best checkpoint: {self.best_checkpoint_path}")

            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt_path)
            test_metrics = test_results[0] if test_results else {}

            _print_metric_block("Final train/validation metrics", fit_metrics)
            _print_metric_block("Final test metrics", test_metrics)

        return {
            "config_path": str(self.config_path),
            "experiment_name": self.experiment_name,
            "output_dir": str(self.experiment_run_dir),
            "best_checkpoint_path": self.best_checkpoint_path,
            "fit_metrics": {k: _to_scalar(v) for k, v in fit_metrics.items()},
            "test_metrics": {k: _to_scalar(v) for k, v in test_metrics.items()},
        }


if __name__ == "__main__":
    args = get_cli_args(default_config="configs/experiment/unet/vfss-inca-unet.yaml")
    experiment = Experiment(config_path=args.config)
    experiment.run(fast_dev_run=args.fast_dev_run)
