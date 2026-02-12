import pytorch_lightning as pl
import torch
from datetime import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import instantiate_from_config
from src.models import LitWrapper
from src.callbacks import build_callbacks
from src.cli import get_cli_args
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, MLFlowLogger
from omegaconf import OmegaConf

pl.seed_everything(42)


def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().mean().item()
    return value


def _print_metric_block(title, metrics):
    print(f"\n{title}")
    if not metrics:
        print("  (no metrics found)")
        return

    for name in sorted(metrics):
        value = _to_scalar(metrics[name])
        if isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")

if __name__ == "__main__":
    args = get_cli_args(default_config="configs/vfss-inca-unet.yaml")
    configs = OmegaConf.load(args.config)

    experiment_name = configs.get("experiment_name", "default_experiment")
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create loggers
    tb_logger = TensorBoardLogger(
        "logs", name=experiment_name, version=run_timestamp
    )
    csv_logger = CSVLogger(
        "logs", name=experiment_name, version=run_timestamp
    )

    # Create model
    model = LitWrapper(model_cfg=configs.model, optimizer_cfg=configs.optimizer)

    # Create data module
    data_module = instantiate_from_config(configs.data)

    callbacks_cfg = configs.trainer.get("callbacks", [])
    callbacks = build_callbacks(callbacks_cfg)

    trainer_params = configs['trainer']['params']
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        **trainer_params
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

    # trainer = pl.Trainer(max_epochs=1000)
    # trainer.fit(lit_model)
