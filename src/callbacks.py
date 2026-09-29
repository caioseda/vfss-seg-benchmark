from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
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


class ResumeFidelity(Callback):
    '''
    Carries the RNG state through a checkpoint, so a resumed run continues the *same* trajectory
    rather than a merely plausible one.

    Lightning restores the model, the optimizer state, the LR scheduler and `global_step`, but
    **not** the random number generators: `_collect_rng_states` exists in
    `lightning_fabric/utilities/seed.py` and is used only by the multiprocessing launcher --
    `_CheckpointConnector.dump_checkpoint` never writes it. Without this, everything downstream of
    a draw diverges after a resume: the augmentations, the Mean Teacher's input noise, FixMatch's
    cutout and DiffRect's `q_sample`.

    What already survives a resume without help, and is therefore deliberately *not* touched here:

      - the consistency ramp and warm-up, because they are computed from `global_step` and
        `total_training_steps` rather than from a counter of their own (`src/models/ssl/base.py`);
      - the Mean Teacher EMA decay, `min(1 - 1/(global_step+1), ema_decay)`
        (`src/models/ssl/mean_teacher.py`);
      - the teacher weights themselves, which are a submodule and so travel in `state_dict`;
      - the LR, which `checkpoint["lr_schedulers"]` restores via the scheduler's `last_epoch`.

    Scope of the guarantee. The *schedule* is restored exactly -- LR, consistency weight and EMA
    decay at step N are identical whether or not the run was interrupted, because all three are
    functions of `global_step`. Bit-exactness of the *augmentation* draws additionally requires
    `num_workers=0`: with worker processes, each worker's seed is derived from a base seed drawn
    when the dataloader iterator is created, and the number of draws preceding that differs across
    a resume boundary. `tests/test_resume.py` pins the schedule unconditionally and the full
    trajectory at `num_workers=0`.
    '''

    def __init__(self) -> None:
        super().__init__()
        self._pending_rng_states = None

    def state_dict(self) -> dict:
        return {"rng_states": _collect_rng_states()}

    def load_state_dict(self, state_dict: dict) -> None:
        rng_states = state_dict.get("rng_states")
        if not rng_states:
            # A checkpoint written before this callback existed. Resuming is still correct for the
            # schedule; only the RNG stream restarts. Say so rather than failing or staying quiet.
            print("ResumeFidelity: checkpoint sem estado de RNG -- o cronograma retoma exato, "
                  "o fluxo aleatorio reinicia.")
            return
        # Stashed, not applied: see `on_train_epoch_start`.
        self._pending_rng_states = rng_states

    def on_train_start(self, trainer, pl_module) -> None:
        '''
        Apply the restored RNG here, not in `load_state_dict`.

        Two hooks look plausible and both are wrong. `load_state_dict` runs before Lightning has
        finished restoring, and `on_train_epoch_start` does **not** fire when a run resumes in the
        middle of an epoch -- the restore would silently land an epoch late, leaving the stream
        phase-shifted by however many batches remained. `on_train_start` fires on every `fit`,
        resumed or not, and before the epoch loop builds its data fetcher.

        This is only exact because `DataModuleFromConfig` gives its DataLoader an explicit
        `generator`: `_BaseDataLoaderIter.__init__` draws a `_base_seed` on every iterator creation,
        and a resumed run creates an iterator where the uninterrupted run did not. Drawing that from
        the loader's own generator keeps it out of the global stream.
        '''
        if self._pending_rng_states is None:
            return
        _set_rng_states(self._pending_rng_states)
        self._pending_rng_states = None
