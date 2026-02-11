from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import instantiate_from_config
from src.models import LitWrapper, UNet
from src.callbacks import PrintDeviceCallback
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    base_cfg_path = "configs/vfss-unet.yaml"
    configs = OmegaConf.load(base_cfg_path)

    # Create logger
    logger = TensorBoardLogger("logs", name="vfss-unet")

    # Create model
    model = LitWrapper(model_cfg=configs.model, optimizer_cfg=configs.optimizer)

    # Create data module
    data_module = instantiate_from_config(configs.data)

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        max_epochs=1000,
        callbacks=[PrintDeviceCallback()],
    )

    trainer.fit(model, data_module)
    # trainer = pl.Trainer(max_epochs=1000)
    # trainer.fit(lit_model)
