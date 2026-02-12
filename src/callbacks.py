from pytorch_lightning.callbacks import Callback
from src.utils import instantiate_from_config
from omegaconf import DictConfig, ListConfig, OmegaConf

def build_callbacks(callbacks_cfg):
    if callbacks_cfg is None:
        return []
    if not isinstance(callbacks_cfg, (list, ListConfig)):
        raise ValueError(
            "Invalid `trainer.callbacks`: expected a list of callback configs."
        )

    callbacks = []
    for index, callback_cfg_item in enumerate(callbacks_cfg):
        if not isinstance(callback_cfg_item, (dict, DictConfig)):
            raise ValueError(
                f"Invalid callback at index {index}: expected a mapping with `target` and optional `params`."
            )
        if "target" not in callback_cfg_item:
            raise ValueError(
                f"Invalid callback at index {index}: missing required key `target`."
            )
        try:
            callbacks.append(instantiate_from_config(callback_cfg_item))
        except Exception as exc:
            raise ValueError(
                f"Failed to instantiate callback at index {index} (`{callback_cfg_item['target']}`): {exc}"
            ) from exc
    return callbacks

class PrintDeviceCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        device = trainer.strategy.root_device
        print(f"Using device: {device}")
