from pytorch_lightning.callbacks import Callback
from src.utils import instantiate_from_config
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.auto import tqdm

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


class OverallTrainingProgressBar(Callback):
    def __init__(self, description: str = "Training Progress"):
        super().__init__()
        self.description = description
        self._progress = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        if trainer.fast_dev_run:
            total_epochs = 1
        else:
            total_epochs = trainer.max_epochs if isinstance(trainer.max_epochs, int) else 0

        if total_epochs is None or total_epochs <= 0:
            return

        self._progress = tqdm(
            total=total_epochs,
            desc=self.description,
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if self._progress is None:
            return
        completed_epochs = trainer.current_epoch + 1
        delta = completed_epochs - self._progress.n
        if delta > 0:
            self._progress.update(delta)

    def on_fit_end(self, trainer, pl_module):
        if self._progress is not None:
            self._progress.close()
            self._progress = None
