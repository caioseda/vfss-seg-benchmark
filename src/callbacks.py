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


class VolumetricValidation(Callback):
    '''
    Validate on reassembled 3-D volumes and log `<prefix>/dice_volume`.

    Why this is a callback and not a `validation_step` override: the ACDC reproduction has to select
    its best checkpoint the way the reference does -- on *volumetric* validation Dice -- while
    `LitWrapper.validation_step` scores slices, which is the right unit for VFSS and must not
    change. A callback gets both without touching any shared code: `Experiment` appends its
    `ModelCheckpoint` *after* the config-declared callbacks and `ModelCheckpoint` reads
    `trainer.callback_metrics` in `on_validation_end`, which runs after every callback's
    `on_validation_epoch_end`. So `checkpoint.monitor: val/dice_volume` resolves correctly.

    Slice-wise `val/*` metrics keep being logged by the module itself; they are simply not what the
    checkpoint is selected on. The gap between the two is informative in its own right.

    Args:
        dataset: `target`/`params` config for an `ACDCVolumeDataset`. Built here rather than taken
            from the datamodule because the datamodule's `validation` split yields slices.
        num_classes: including background.
        class_ids: `{class_index: display_name}`; defaults to the module's `report_class_ids`.
        prefix: metric namespace, `val` by default.
        size: network input resolution used when resampling each slice.
        surface_metrics: also log HD95/ASD. Off by default -- a Euclidean distance transform per
            class per volume, on every validation, would dominate the wall clock. The reported
            numbers come from the notebook's final pass, which turns them on.
        forward_attr: attribute to prefer for the forward pass, so Mean Teacher is evaluated with
            its teacher exactly as `evaluate_per_class` does.
    '''

    def __init__(self, dataset, num_classes: int, class_ids=None, prefix: str = "val",
                 size: int = 256, surface_metrics: bool = False,
                 forward_attr: str = "_eval_forward"):
        super().__init__()
        self.dataset_cfg = dataset
        self.num_classes = num_classes
        self.class_ids = class_ids
        self.prefix = prefix
        self.size = size
        self.surface_metrics = surface_metrics
        self.forward_attr = forward_attr
        self._dataset = None

    def _resolve_class_ids(self, pl_module):
        if self.class_ids is not None:
            if isinstance(self.class_ids, DictConfig):
                self.class_ids = OmegaConf.to_container(self.class_ids, resolve=True)
            return {int(k): str(v) for k, v in dict(self.class_ids).items()}
        return pl_module.report_class_ids

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        from src.evaluation_volume import aggregate_volumes, evaluate_volumes

        if self._dataset is None:
            self._dataset = instantiate_from_config(self.dataset_cfg)

        class_ids = self._resolve_class_ids(pl_module)
        was_training = pl_module.training
        per_volume = evaluate_volumes(
            pl_module, self._dataset, num_classes=self.num_classes, class_ids=class_ids,
            device=trainer.strategy.root_device,
            forward_fn=getattr(pl_module, self.forward_attr, None),
            size=self.size, surface_metrics=self.surface_metrics,
        )
        if was_training:
            pl_module.train()

        aggregated = aggregate_volumes(per_volume)
        logs = {}
        for _, row in aggregated.iterrows():
            for metric in ("dice", "jaccard", "hd95", "asd"):
                if metric in row and row[metric] == row[metric]:  # skip NaN
                    logs[f"{self.prefix}/{metric}_volume/{row['class']}"] = float(row[metric])
        for metric in ("dice", "jaccard", "hd95", "asd"):
            if metric in aggregated.columns:
                value = aggregated[metric].mean(skipna=True)
                if value == value:
                    logs[f"{self.prefix}/{metric}_volume"] = float(value)

        pl_module.log_dict(logs, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                           sync_dist=True)
