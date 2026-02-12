import pytorch_lightning as pl
from omegaconf import OmegaConf
from datetime import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import instantiate_from_config
from src.models import LitWrapper
from src.callbacks import PrintDeviceCallback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

pl.seed_everything(42)

if __name__ == "__main__":

    base_cfg_path = "configs/vfss-unet.yaml"
    configs = OmegaConf.load(base_cfg_path)
    experiment_name = configs.get("experiment_name", "vfss-unet")
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create loggers: TensorBoard events + CSV metrics in the same timestamped folder.
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

    callbacks = [
        PrintDeviceCallback(),
        EarlyStopping(monitor="val/loss", mode="min", patience=5)
    ]

    trainer_params = configs['trainer']['params']
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        **trainer_params
    )

    trainer.fit(model, data_module)
    # trainer = pl.Trainer(max_epochs=1000)
    # trainer.fit(lit_model)
