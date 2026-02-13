from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.callbacks import build_callbacks
from src.models import LitWrapper
from src.utils import instantiate_from_config, _to_scalar, _resolve_config_path, _print_metric_block


pl.seed_everything(42)

def run_experiment(config_path: str, run_suffix: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(config_path)
    config = OmegaConf.load(str(cfg_path))

    config_filename = cfg_path.stem
    experiment_name = config.get("experiment_name", config_filename)
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_version = run_timestamp if run_suffix is None else f"{run_timestamp}-{run_suffix}"
    
    print(f"Loaded configuration: {cfg_path}")
    print(f"Run version: {run_version}")

    tb_logger = TensorBoardLogger("logs", name=experiment_name, version=run_version)
    csv_logger = CSVLogger("logs", name=experiment_name, version=run_version)

    model = LitWrapper(model_cfg=config.model, optimizer_cfg=config.optimizer)
    data_module = instantiate_from_config(config.data)

    callbacks_cfg = config.trainer.get("callbacks", [])
    callbacks = build_callbacks(callbacks_cfg)
    trainer_params = config["trainer"]["params"]

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
        "fit_metrics": {k: _to_scalar(v) for k, v in fit_metrics.items()},
        "test_metrics": {k: _to_scalar(v) for k, v in test_metrics.items()},
    }
