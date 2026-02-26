from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.callbacks import OverallTrainingProgressBar, build_callbacks
from src.models import LitWrapper
from src.utils import instantiate_from_config, _to_scalar, _resolve_config_path, _print_metric_block


pl.seed_everything(42)

def run_experiment(
    config_path: str,
    run_suffix: Optional[str] = None,
    logs_root: Optional[str | Path] = None,
    experiment_subdir: Optional[str] = None,
    fast_dev_run: bool | int = False,
) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(config_path)
    config = OmegaConf.load(str(cfg_path))

    config_filename = cfg_path.stem
    experiment_name = config.get("experiment_name", config_filename)
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_version = run_timestamp if run_suffix is None else f"{run_timestamp}-{run_suffix}"
    
    print(f"Loaded configuration: {cfg_path}")
    print(f"Run version: {run_version}")

    if logs_root is None:
        tb_logger = TensorBoardLogger("logs", name=experiment_name, version=run_version)
        csv_logger = CSVLogger("logs", name=experiment_name, version=run_version)
        output_dir = Path("logs") / experiment_name / run_version
    else:
        benchmark_root = Path(logs_root).expanduser().resolve()
        benchmark_root.mkdir(parents=True, exist_ok=True)
        exp_dirname = experiment_subdir or experiment_name
        # Keep one top-level folder per experiment and separate logger outputs inside it.
        tb_logger = TensorBoardLogger(
            save_dir=str(benchmark_root), name=exp_dirname, version="tensorboard"
        )
        csv_logger = CSVLogger(
            save_dir=str(benchmark_root), name=exp_dirname, version="csv"
        )
        output_dir = benchmark_root / exp_dirname

    model = LitWrapper(model_cfg=config.model, optimizer_cfg=config.optimizer)
    data_module = instantiate_from_config(config.data)

    callbacks_cfg = config.trainer.get("callbacks", [])
    callbacks = build_callbacks(callbacks_cfg)
    if not any(isinstance(callback, OverallTrainingProgressBar) for callback in callbacks):
        callbacks.append(OverallTrainingProgressBar())
    trainer_params = config["trainer"]["params"]
    if fast_dev_run:
        trainer_params["fast_dev_run"] = fast_dev_run
        print(f"fast_dev_run enabled via CLI override: {fast_dev_run}")

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
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
        "config_path": str(cfg_path),
        "experiment_name": experiment_name,
        "run_version": run_version,
        "output_dir": str(output_dir),
        "fit_metrics": {k: _to_scalar(v) for k, v in fit_metrics.items()},
        "test_metrics": {k: _to_scalar(v) for k, v in test_metrics.items()},
    }
