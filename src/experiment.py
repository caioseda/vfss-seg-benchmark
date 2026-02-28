from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from src.cli import get_cli_args

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.callbacks import OverallTrainingProgressBar, build_callbacks
from src.models import LitWrapper
from src.utils import (
    instantiate_from_config,
    _to_scalar,
    _resolve_config_path,
    _print_metric_block,
    capture_console_output,
)

class Experiment:
    def __init__(self, 
                 config_path: str, 
                 benchmark_dir: Optional[str | Path] = None,
                 experiment_id: Optional[str] = None
            ):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        config_filename = Path(self.config_path).stem
        self.experiment_name = self.config.get("name", self.config.get("experiment_name", config_filename))
        
        self.running_as_benchmark = bool(benchmark_dir or experiment_id)
        if self.running_as_benchmark:
            self.experiment_dirname = f"{experiment_id}-{self.experiment_name}"
        else:
            self.experiment_dirname = self.experiment_name 

        self.parent_dir = benchmark_dir if self.running_as_benchmark else "logs"
        self.parent_dir = Path(self.parent_dir)


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

        # Assert required fields
        _required_fields = ['model', 'optimizer', 'data', 'trainer']
        for field in _required_fields:
            if field not in config:
                raise ValueError(f"Missing required field `{field}` in {config}")
        
        assert config.type == "experiment", f"Config type must be 'experiment', got '{config.type}' in {config}"
        return config


    def setup_loggers(self) -> Dict[str, Any]:
        # Keep logger artifacts colocated with console output for this run.
        save_dir = str(self.experiment_run_dir)
        tb_logger = TensorBoardLogger(save_dir=save_dir, name="", version="tensorboard")
        csv_logger = CSVLogger(save_dir=save_dir, name="", version="csv")
        return {
            "tensorboard": tb_logger,
            "csv": csv_logger,
        }

    def run(self, fast_dev_run: bool | int = False, seed=42) -> Dict[str, Any]:
        pl.seed_everything(seed)
        self.setup_exeperiment_run_dir()
        self.console_log_path = self.experiment_run_dir / "console.log"
        
        loggers = self.setup_loggers()
        with capture_console_output(self.console_log_path):
            print(f"Loaded configuration: {self.config_path}")
            print(f"Console logs: {self.console_log_path}")
            
            model = LitWrapper(model_cfg=self.config.model, optimizer_cfg=self.config.optimizer)
            data_module = instantiate_from_config(self.config.data)

            callbacks_cfg = self.config.trainer.get("callbacks", [])
            callbacks = build_callbacks(callbacks_cfg)
            if not any(isinstance(callback, OverallTrainingProgressBar) for callback in callbacks):
                callbacks.append(OverallTrainingProgressBar())
            
            trainer_params = self.config["trainer"]["params"]
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
            test_results = trainer.test(model=model, datamodule=data_module)
            test_metrics = test_results[0] if test_results else {}

            _print_metric_block("Final train/validation metrics", fit_metrics)
            _print_metric_block("Final test metrics", test_metrics)

        return {
            "config_path": str(self.config_path),
            "experiment_name": self.experiment_name,
            "output_dir": str(self.experiment_run_dir),
            "fit_metrics": {k: _to_scalar(v) for k, v in fit_metrics.items()},
            "test_metrics": {k: _to_scalar(v) for k, v in test_metrics.items()},
        }


if __name__ == "__main__":
    args = get_cli_args(default_config="configs/experiment/unet/vfss-inca-unet.yaml")
    experiment = Experiment(config_path=args.config)
    experiment.run(fast_dev_run=args.fast_dev_run)
