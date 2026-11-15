import os
import re
import time
import torch
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


class PatientValidation(Callback):
    '''
    Validate per patient and log `<prefix>/<metric>_patient`, the article protocol's selection metric.

    Same mechanism as `VolumetricValidation`, for the same reason: `LitWrapper.validation_step`
    averages frames batch by batch, which lets the few patients with many frames dominate, while the
    protocol selects checkpoints on the mean over *patients* of their mean frame Dice. A callback adds
    that without touching the module, and `ModelCheckpoint` (appended by `Experiment` after the
    config-declared callbacks) reads it in `on_validation_end`, so `monitor: val/dice_patient` works.

    It runs a second pass over the validation loader -- ~110 frames, a few seconds every validation.
    The module's frame-level `val/*` keeps being logged alongside, as a diagnostic.

    Args:
        num_classes: including background.
        class_ids: `{class_index: display_name}`; defaults to the module's `report_class_ids`.
        prefix: metric namespace, `val` by default.
        forward_attr: attribute to prefer for the forward pass, so Mean Teacher is evaluated with
            its teacher exactly as `evaluate_per_class` does.
    '''

    def __init__(self, num_classes: int, class_ids=None, prefix: str = "val",
                 forward_attr: str = "_eval_forward"):
        super().__init__()
        self.num_classes = num_classes
        self.class_ids = class_ids
        self.prefix = prefix
        self.forward_attr = forward_attr

    def _resolve_class_ids(self, pl_module):
        if self.class_ids is not None:
            if isinstance(self.class_ids, DictConfig):
                self.class_ids = OmegaConf.to_container(self.class_ids, resolve=True)
            return {int(k): str(v) for k, v in dict(self.class_ids).items()}
        return pl_module.report_class_ids

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        from src.evaluation import METRIC_NAMES, aggregate_per_patient, evaluate_per_class

        was_training = pl_module.training
        per_sample = evaluate_per_class(
            pl_module, trainer.datamodule.val_dataloader(),
            num_classes=self.num_classes, class_ids=self._resolve_class_ids(pl_module),
            device=trainer.strategy.root_device,
            forward_fn=getattr(pl_module, self.forward_attr, None),
        )
        if was_training:
            pl_module.train()

        aggregated = aggregate_per_patient(per_sample)
        logs = {}
        for _, row in aggregated.iterrows():
            for metric in METRIC_NAMES:
                if row[metric] == row[metric]:  # skip NaN
                    logs[f"{self.prefix}/{metric}_patient/{row['class']}"] = float(row[metric])
        for metric in METRIC_NAMES:
            value = aggregated[metric].mean(skipna=True)
            if value == value:
                logs[f"{self.prefix}/{metric}_patient"] = float(value)

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


class ResourceMonitor(Callback):
    '''
    Custo de um run, para comparar métodos além da acurácia: tempo, VRAM, tamanho do modelo e carga
    por passo. Escrito para o E2, onde os métodos diferem justamente no custo (o teacher do MT dobra os
    pesos em memória; a variante temporal bidirecional passa 3x mais imagens pelo teacher; o fluxo óptico
    roda nos workers e pode deixar a GPU esperando).

    O que mede:
      tempo    parede do fit, treino sem validação, validação, teste; s/iteração (média, p50, p95);
               espera por dados = intervalo entre o fim de um passo e o início do próximo (o loader
               não entregou o batch a tempo). Sem `cuda.synchronize` por passo -- sincronizar distorceria
               o próprio tempo medido; os totais são exatos, os percentis por iteração são aproximados.
      VRAM     `max_memory_allocated`/`max_memory_reserved` separados por fase (treino, validação, teste),
               com reset entre fases. `sys/vram_peak_gb` também é logado a cada `log_every_n_steps`.
      modelo   parâmetros do student (totais/treináveis), do teacher quando existe, GFLOPs por frame
               (`torch.utils.flop_counter`, entrada 1 x C x `input_size`^2), tamanho do melhor checkpoint
               e passo em que ele foi salvo.
      carga    imagens por passo por rede, lidas de `pl_module.forward_counts` (ver
               `SemiSupervisedLitWrapper.count_forward`).
      latência inferência por frame, batch 1 e batch 8, pelo caminho de avaliação (`_eval_forward`, o
               teacher no MT), com sincronização -- aqui ela é necessária.

    Os acumulados viajam no checkpoint (`state_dict`), então um run retomado soma o tempo de todas as
    tentativas. `summary()` é o que `Experiment.run` grava em `run.json` (`resources`) e o que a tabela
    comparativa lê.
    '''

    def __init__(self, input_size: int = 256, log_every_n_steps: int = 50,
                 latency_batch_sizes=(1, 8), latency_iters: int = 20, measure_flops: bool = True):
        super().__init__()
        self.input_size = int(input_size)
        self.log_every_n_steps = int(log_every_n_steps)
        self.latency_batch_sizes = [int(b) for b in latency_batch_sizes]
        self.latency_iters = int(latency_iters)
        self.measure_flops = bool(measure_flops)

        # Acumulados (persistidos).
        self.totals = {"fit_s": 0.0, "train_s": 0.0, "val_s": 0.0, "data_wait_s": 0.0, "steps": 0,
                       "attempts": 0}
        self.iteration_s: list = []
        self.images: dict = {}
        self.peaks = {"train_alloc": 0.0, "train_reserved": 0.0, "val_alloc": 0.0, "test_alloc": 0.0}
        # Por tentativa.
        self._fit_start = None
        self._phase_start = None
        self._last_batch_end = None
        self._batch_start = None
        self.static: dict = {}

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        totals = dict(self.totals)
        if self._fit_start is not None:
            totals["fit_s"] += time.perf_counter() - self._fit_start
        return {"totals": totals, "iteration_s": list(self.iteration_s), "images": dict(self.images),
                "peaks": dict(self.peaks)}

    def load_state_dict(self, state_dict: dict) -> None:
        self.totals.update(state_dict.get("totals", {}))
        self.iteration_s = list(state_dict.get("iteration_s", []))
        self.images = dict(state_dict.get("images", {}))
        self.peaks.update(state_dict.get("peaks", {}))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _cuda() -> bool:
        return torch.cuda.is_available()

    def _read_peak(self) -> tuple:
        if not self._cuda():
            return 0.0, 0.0
        return (torch.cuda.max_memory_allocated() / 1024 ** 3, torch.cuda.max_memory_reserved() / 1024 ** 3)

    def _reset_peak(self) -> None:
        if self._cuda():
            torch.cuda.reset_peak_memory_stats()

    def _close_train_window(self) -> None:
        if self._phase_start is not None:
            self.totals["train_s"] += time.perf_counter() - self._phase_start
            self._phase_start = None
        alloc, reserved = self._read_peak()
        self.peaks["train_alloc"] = max(self.peaks["train_alloc"], alloc)
        self.peaks["train_reserved"] = max(self.peaks["train_reserved"], reserved)

    @staticmethod
    def _count_params(module) -> tuple:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _example_input(self, pl_module, batch_size: int):
        channels = int(getattr(getattr(pl_module, "model", None), "n_channels", 3) or 3)
        return torch.randn(batch_size, channels, self.input_size, self.input_size, device=pl_module.device)

    @torch.no_grad()
    def _measure_static(self, pl_module) -> None:
        network = getattr(pl_module, "model", pl_module)
        total, trainable = self._count_params(network)
        self.static["params_total_m"] = total / 1e6
        self.static["params_trainable_m"] = trainable / 1e6
        teacher = getattr(pl_module, "teacher", None)
        self.static["teacher_params_m"] = (self._count_params(teacher)[0] / 1e6) if teacher is not None else 0.0
        all_total, all_trainable = self._count_params(pl_module)
        self.static["module_params_total_m"] = all_total / 1e6
        self.static["module_params_trainable_m"] = all_trainable / 1e6
        if self.measure_flops:
            try:
                from torch.utils.flop_counter import FlopCounterMode
                was_training = network.training
                network.eval()
                counter = FlopCounterMode(display=False)
                with counter:
                    network(self._example_input(pl_module, 1))
                network.train(was_training)
                self.static["gflops_per_frame"] = counter.get_total_flops() / 1e9
            except Exception as error:  # medir FLOPs nunca pode derrubar um treino
                print(f"ResourceMonitor: FLOPs nao medidos ({type(error).__name__}: {error})")
                self.static["gflops_per_frame"] = float("nan")

    @torch.no_grad()
    def _measure_latency(self, pl_module) -> None:
        forward = getattr(pl_module, "_eval_forward", None) or pl_module.forward
        was_training = pl_module.training
        pl_module.eval()
        for batch_size in self.latency_batch_sizes:
            x = self._example_input(pl_module, batch_size)
            for _ in range(3):
                forward(x)
            if self._cuda():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(self.latency_iters):
                forward(x)
            if self._cuda():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / self.latency_iters
            self.static[f"latency_ms_per_frame_bs{batch_size}"] = 1000.0 * elapsed / batch_size
        pl_module.train(was_training)

    # ------------------------------------------------------------------ fit

    def on_fit_start(self, trainer, pl_module):
        self.totals["attempts"] = int(self.totals.get("attempts", 0)) + 1
        self._fit_start = time.perf_counter()
        self._measure_static(pl_module)
        self.static["gpu"] = torch.cuda.get_device_name(0) if self._cuda() else "cpu"
        self.static["torch"] = torch.__version__
        self.static["precision"] = str(getattr(trainer, "precision", ""))
        self.static["num_workers"] = getattr(trainer.datamodule, "num_workers", None) if trainer.datamodule else None

    def on_train_start(self, trainer, pl_module):
        self._reset_peak()
        self._phase_start = time.perf_counter()
        self._last_batch_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._phase_start is None:  # primeiro passo depois de uma validação
            self._phase_start = now
        if self._last_batch_end is not None:
            self.totals["data_wait_s"] += now - self._last_batch_end
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._last_batch_end is not None:
            self.iteration_s.append(now - self._last_batch_end)
        elif self._batch_start is not None:
            self.iteration_s.append(now - self._batch_start)
        self._last_batch_end = now
        self.totals["steps"] += 1
        for network, count in dict(getattr(pl_module, "forward_counts", {}) or {}).items():
            self.images[network] = self.images.get(network, 0) + int(count)
        if self.log_every_n_steps and trainer.global_step % self.log_every_n_steps == 0 and self._cuda():
            pl_module.log("sys/vram_peak_gb", torch.cuda.max_memory_allocated() / 1024 ** 3,
                          on_step=True, on_epoch=False, logger=True)

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.state.fn == "fit":
            self._close_train_window()
        self._reset_peak()
        self._val_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.totals["val_s"] += time.perf_counter() - getattr(self, "_val_start", time.perf_counter())
        self.peaks["val_alloc"] = max(self.peaks["val_alloc"], self._read_peak()[0])
        self._reset_peak()
        self._last_batch_end = None  # o intervalo da validação não é espera por dados

    def on_train_end(self, trainer, pl_module):
        self._close_train_window()

    def on_fit_end(self, trainer, pl_module):
        if self._fit_start is not None:
            self.totals["fit_s"] += time.perf_counter() - self._fit_start
            self._fit_start = None
        for callback in trainer.checkpoint_callbacks:
            best = getattr(callback, "best_model_path", "")
            if getattr(callback, "monitor", None) and best and os.path.exists(best):
                self.static["checkpoint_mb"] = os.path.getsize(best) / 1024 ** 2
                # Lido do checkpoint, não do nome do arquivo: os wrappers SSL logam uma métrica
                # `step` própria, e é ela (não o global_step) que o `{step}` do nome formata.
                try:
                    self.static["best_step"] = int(torch.load(best, map_location="cpu",
                                                              weights_only=False)["global_step"])
                except Exception:
                    match = re.search(r"step=(\d+)", os.path.basename(best))
                    self.static["best_step"] = int(match.group(1)) if match else None
                break
        metrics = {f"sys/{k}": float(v) for k, v in self.summary().items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        for logger in trainer.loggers:
            try:
                logger.log_metrics(metrics, step=trainer.global_step)
            except Exception as error:
                print(f"ResourceMonitor: {type(logger).__name__} recusou as metricas ({error})")

    # ------------------------------------------------------------------ test

    def on_test_start(self, trainer, pl_module):
        self._reset_peak()
        self._test_start = time.perf_counter()

    def on_test_end(self, trainer, pl_module):
        self.static["test_s"] = time.perf_counter() - getattr(self, "_test_start", time.perf_counter())
        self.peaks["test_alloc"] = max(self.peaks["test_alloc"], self._read_peak()[0])
        self.measure_latency(pl_module)

    def measure_latency(self, pl_module) -> None:
        '''
        Latência de inferência, medida DEPOIS do treino (no fim do teste, ou chamada por quem avalia).
        Medir no fim do fit pegava os workers persistentes do treino ainda vivos disputando CPU, e a
        DeepLabV3+ aparecia 20x mais lenta do que é isolada.
        '''
        try:
            self._measure_latency(pl_module)
        except Exception as error:
            print(f"ResourceMonitor: latencia nao medida ({type(error).__name__}: {error})")

    # ------------------------------------------------------------------ report

    def summary(self) -> dict:
        steps = max(int(self.totals["steps"]), 1)
        iterations = sorted(self.iteration_s)

        def percentile(q: float) -> float:
            if not iterations:
                return float("nan")
            return float(iterations[min(int(q * len(iterations)), len(iterations) - 1)])

        out = {
            "fit_wall_s": float(self.totals["fit_s"]),
            "train_s": float(self.totals["train_s"]),
            "val_s": float(self.totals["val_s"]),
            "steps": int(self.totals["steps"]),
            "attempts": int(self.totals.get("attempts", 0)),
            "s_per_step_mean": float(self.totals["train_s"]) / steps,
            "s_per_step_p50": percentile(0.50),
            "s_per_step_p95": percentile(0.95),
            "data_wait_s": float(self.totals["data_wait_s"]),
            "data_wait_frac": float(self.totals["data_wait_s"]) / max(float(self.totals["train_s"]), 1e-9),
            "vram_train_peak_gb": self.peaks["train_alloc"],
            "vram_train_reserved_gb": self.peaks["train_reserved"],
            "vram_val_peak_gb": self.peaks["val_alloc"],
            "vram_test_peak_gb": self.peaks["test_alloc"],
        }
        for network, count in self.images.items():
            out[f"images_per_step_{network}"] = count / steps
        out.update(self.static)
        return out
