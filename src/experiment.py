import hashlib
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.cli import get_cli_args

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.callbacks import OverallTrainingProgressBar, ResumeFidelity, build_callbacks
from src.models import LitWrapper
from src.utils import (
    instantiate_from_config,
    get_obj_from_str,
    _to_scalar,
    _resolve_config_path,
    _print_metric_block,
    capture_console_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Experiment:
    def __init__(self, 
                 config_path: str, 
                 benchmark_dir: Optional[str | Path] = None,
                 experiment_id: Optional[str] = None,
                 overrides: Optional[Union[Dict[str, Any], DictConfig]] = None,
                 log_dir: Optional[str | Path] = None,
                 family: Optional[str] = None,
                 run_uid: Optional[str] = None,
            ):
        '''
        Args:
            config_path: path to the experiment YAML.
            benchmark_dir / experiment_id: set when running as part of a `Benchmark`.
            overrides: a partial config merged over the loaded YAML (`OmegaConf.merge`, so nested
                keys are merged, not replaced). This is how a notebook sweeps one knob -- e.g.
                `{"data": {"params": {"shared_dataset_params": {"label_fraction": 0.1}}}}` -- without
                writing a temporary YAML per cell of the grid.
            log_dir: root directory for run outputs. Defaults to `logs/` **relative to the current
                working directory** -- pass an absolute path when the caller's cwd is not the repo
                root (a notebook runs from its own directory, and would otherwise scatter
                checkpoints and TensorBoard logs next to itself).
            family: the experiment family this run belongs to (`x1`, `x1b`, `acdc`, ...). Set by a
                sweep. It puts the run under `<log_dir>/<family>/runs/`, which is what makes
                `tensorboard --logdir logs/<family>/runs` show one experiment and nothing else.
                Left None, the historical flat layout is used, so existing callers are unaffected.
            run_uid: identifier for this run's directory, **stable across attempts**. A sweep passes
                the same value when re-entering a cell, and that is what makes resuming possible:
                minting a fresh timestamp on re-entry is exactly what produced the orphan
                directories `scripts/migrate_logs.py` had to sweep into `_archive/incomplete/`.
        '''
        self.config_path = config_path
        self.config = self._load_config(config_path)
        if overrides:
            self.config = OmegaConf.merge(self.config, OmegaConf.create(overrides))
        config_filename = Path(self.config_path).stem
        self.experiment_name = self.config.get("name", self.config.get("experiment_name", config_filename))
        
        self.running_as_benchmark = bool(benchmark_dir or experiment_id)
        if self.running_as_benchmark:
            self.experiment_dirname = f"{experiment_id}-{self.experiment_name}"
        else:
            self.experiment_dirname = self.experiment_name 

        self.parent_dir = benchmark_dir if self.running_as_benchmark else (log_dir or "logs")
        self.parent_dir = Path(self.parent_dir)
        self.family = family
        self._run_uid = run_uid

        # Populated by `run`, so a caller (the X1 notebook) can evaluate the trained model itself.
        self.lit_module: Optional[pl.LightningModule] = None
        self.data_module: Optional[pl.LightningDataModule] = None
        self.best_checkpoint_path: Optional[str] = None
        self.resumed_from_step: Optional[int] = None


    def setup_exeperiment_run_dir(self) -> None:
        '''
        `<log_dir>/<family>/runs/<name>/<run_uid>` when a family is given, else the historical
        `<log_dir>/<name>/<timestamp>`. Either way the run dir is self-contained: config, logs,
        curves, checkpoints and provenance live together, which is what lets every tracker index
        the run by pointing at it rather than by copying out of it.
        '''
        if self.running_as_benchmark:
            self.experiment_run_dir = self.parent_dir / self.experiment_dirname
        else:
            self._timestamp = self._run_uid or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            base = self.parent_dir / self.family / "runs" if self.family else self.parent_dir
            self.experiment_run_dir = base / self.experiment_dirname / self._timestamp
        self.experiment_run_dir.mkdir(parents=True, exist_ok=True)
        

    def _load_config(self, cfg_path: str) -> Dict[str, Any]:
        resolved_path = _resolve_config_path(cfg_path)
        config = OmegaConf.load(str(resolved_path))

        _required_fields = ['model', 'optimizer', 'data', 'trainer']
        for field in _required_fields:
            if field not in config:
                raise ValueError(f"Missing required field `{field}` in {config}")
        
        assert config.type == "experiment", f"Config type must be 'experiment', got '{config.type}' in {config}"
        return config


    def setup_loggers(self) -> List[Any]:
        '''
        Run loggers. TensorBoard + CSV always, so every run leaves a self-contained record on disk
        next to its checkpoints; an optional `logging.loggers` block in the YAML appends others
        (MLflow, W&B, ...) without touching this file:

            logging:
              loggers:
                - target: pytorch_lightning.loggers.MLFlowLogger
                  params:
                    experiment_name: x1
                    tracking_uri: file:./mlruns

        The CSV logger is what makes a sweep analysable offline: `metrics.csv` per run reads
        straight into pandas, which TensorBoard event files do not.
        '''
        save_dir = str(self.experiment_run_dir)
        loggers: List[Any] = [
            TensorBoardLogger(save_dir=save_dir, name="", version="tensorboard"),
            CSVLogger(save_dir=save_dir, name="", version="csv"),
        ]

        extra_loggers = self.config.get("logging", {}).get("loggers", []) or []
        for logger_cfg in extra_loggers:
            loggers.append(instantiate_from_config(logger_cfg))
        return loggers

    @staticmethod
    def _flatten_config(config: Any, prefix: str = "") -> Dict[str, Any]:
        '''Flatten the resolved config into `a/b/c: value` pairs a tracker can index and compare runs by.'''
        flat: Dict[str, Any] = {}
        if isinstance(config, dict):
            for key, value in config.items():
                flat.update(Experiment._flatten_config(value, f"{prefix}{key}/"))
        elif isinstance(config, (list, tuple)):
            # Loggers reject nested containers; a list is only ever read, never queried on.
            flat[prefix.rstrip("/")] = str(list(config))
        else:
            flat[prefix.rstrip("/")] = config
        return flat

    def log_run_hyperparameters(self, loggers: List[Any], seed: int) -> None:
        '''
        Record the whole resolved config as searchable hyperparameters.

        Without this a sweep produces N directories of curves with no machine-readable record of
        *what varied* -- `label_fraction`, the method, the seed and the consistency schedule all live
        only in the YAML. Logging them makes runs comparable inside the tracker instead of by
        cross-referencing paths by hand.
        '''
        hyperparameters = self._flatten_config(OmegaConf.to_container(self.config, resolve=True))
        hyperparameters["seed"] = seed
        for logger in loggers:
            try:
                logger.log_hyperparams(hyperparameters)
            except Exception as error:  # a tracker being picky must not kill the run
                print(f"Could not log hyperparameters to {type(logger).__name__}: {error}")

    def _build_lit_module(self) -> pl.LightningModule:
        '''
        Instantiate the LightningModule. An optional `lit_module:` block (`target`/`params`) selects
        a subclass -- that is how the X1 configs swap in the semi-supervised wrappers from
        `src.models.ssl`. Without the block the behaviour is the historical `LitWrapper`, so the
        existing configs are unaffected.
        '''
        lit_module_cfg = self.config.get("lit_module")
        lit_module_cls = LitWrapper
        extra_params: Dict[str, Any] = {}
        if lit_module_cfg is not None:
            lit_module_cls = get_obj_from_str(lit_module_cfg["target"])
            extra_params = dict(lit_module_cfg.get("params", {}) or {})

        return lit_module_cls(
            model_cfg=self.config.model,
            optimizer_cfg=self.config.optimizer,
            lr_scheduler_cfg=self.config.get("lr_scheduler"),
            **extra_params,
        )

    # ------------------------------------------------------------------ provenance

    @staticmethod
    def _git_info() -> Dict[str, Any]:
        def run(*args: str) -> Optional[str]:
            try:
                return subprocess.check_output(args, cwd=str(PROJECT_ROOT), text=True,
                                               stderr=subprocess.DEVNULL).strip()
            except Exception:
                return None
        return {"sha": run("git", "rev-parse", "HEAD"),
                "dirty": bool(run("git", "status", "--porcelain"))}

    def _resolved_config_yaml(self) -> str:
        try:
            return OmegaConf.to_yaml(self.config, resolve=True)
        except Exception:
            # An unresolvable interpolation must not cost us the record of what ran.
            return OmegaConf.to_yaml(self.config)

    def config_fingerprint(self) -> str:
        '''sha256 of the resolved config -- the identity of "what this run is training".'''
        return hashlib.sha256(self._resolved_config_yaml().encode("utf-8")).hexdigest()

    @property
    def run_metadata_path(self) -> Path:
        return self.experiment_run_dir / "run.json"

    def read_run_metadata(self) -> Dict[str, Any]:
        try:
            return json.loads(self.run_metadata_path.read_text())
        except Exception:
            return {}

    def write_run_metadata(self, **updates: Any) -> Dict[str, Any]:
        metadata = self.read_run_metadata()
        metadata.update(updates)
        self.run_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        return metadata

    def _record_provenance(self, seed: int) -> None:
        '''
        Freeze what ran, next to its outputs: the resolved config as YAML (today the effective
        config survives only flattened inside `hparams.yaml`, which cannot be re-run) plus a
        `run.json` carrying the git sha, the seed, the host and the run's status. This is the
        per-run record the NeurIPS reproducibility checklist asks for and that nothing in the repo
        produced before.
        '''
        (self.experiment_run_dir / "config.yaml").write_text(self._resolved_config_yaml())
        self.write_run_metadata(
            run_uid=self.experiment_run_dir.name,
            family=self.family,
            run_name=self.experiment_name,
            config_path=str(self.config_path),
            config_fingerprint=self.config_fingerprint(),
            seed=seed,
            host=socket.gethostname(),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            git=self._git_info(),
            started_at=datetime.now().isoformat(),
            status="running",
        )

    # ------------------------------------------------------------------ resume

    @property
    def last_checkpoint_path(self) -> Path:
        return self.experiment_run_dir / "checkpoints" / "last.ckpt"

    def _resolve_resume_checkpoint(self, resume: bool, force_resume: bool) -> Optional[str]:
        '''
        Decide whether to continue a previous attempt in this run dir.

        Refuses on a config fingerprint mismatch, because that failure is otherwise silent and
        corrupting: changing `max_steps` between attempts rewrites the consistency ramp and the
        cosine `T_max` **for the remaining steps only**, producing a run that followed a schedule no
        config describes. `force_resume` overrides, and the override is recorded in `run.json`.
        '''
        if not resume or not self.last_checkpoint_path.exists():
            return None

        previous = self.read_run_metadata()
        recorded = previous.get("config_fingerprint")
        current = self.config_fingerprint()
        if recorded and recorded != current:
            message = (
                f"A configuracao mudou desde a tentativa anterior neste run dir.\n"
                f"  anterior: {recorded[:12]}\n  atual:    {current[:12]}\n"
                f"  {self.experiment_run_dir}\n"
                "Retomar misturaria duas configuracoes -- e em silencio, porque os cronogramas "
                "ancorados em `max_steps` seriam reescritos so para o trecho restante. Use um "
                "run_uid novo, ou force_resume=True se a mudanca for irrelevante."
            )
            if not force_resume:
                raise RuntimeError(message)
            print(f"AVISO (force_resume): {message}")
            self.write_run_metadata(forced_resume_despite_config_change=True)
        return str(self.last_checkpoint_path)

    def _build_checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        '''
        Model selection on validation, configured by an optional `trainer.checkpoint` block.

        Without it the reported test numbers come from the *last* epoch, which is unfair precisely
        in the low-label regimes that overfit hardest -- exactly the regimes X1 is measuring.
        '''
        checkpoint_cfg = self.config.trainer.get("checkpoint")
        if checkpoint_cfg is None:
            return None
        params = OmegaConf.to_container(checkpoint_cfg, resolve=True) if isinstance(checkpoint_cfg, DictConfig) else dict(checkpoint_cfg)
        params.setdefault("dirpath", str(self.experiment_run_dir / "checkpoints"))
        return ModelCheckpoint(**params)

    def _build_resume_checkpoint_callback(self, every_n_train_steps: int = 500) -> ModelCheckpoint:
        '''
        A second, separate `ModelCheckpoint` whose only job is `last.ckpt`: the rolling snapshot a
        crashed run restarts from. Kept apart from the `best` callback on purpose -- `best` is model
        *selection* (monitored on validation, which only runs every N epochs), while this is crash
        *insurance* and has to fire on a step cadence. Bounds the loss from an interrupted run to
        `every_n_train_steps` steps instead of the whole run.

        It is overwritten in place, and `run` deletes it once the run completes, so it costs one
        checkpoint of disk while a run is live and nothing afterwards.
        '''
        return ModelCheckpoint(
            dirpath=str(self.experiment_run_dir / "checkpoints"),
            # `save_last=True` is what writes (and overwrites) `last.ckpt`; `save_top_k=0` turns off
            # the monitored top-k saving, which is the other callback's job. `monitor=None` matters:
            # with a monitor, `on_train_batch_end` defers the save until validation, which would
            # defeat the step cadence.
            monitor=None, save_last=True, save_top_k=0,
            every_n_train_steps=every_n_train_steps, save_on_train_epoch_end=False,
        )

    def run(self, fast_dev_run: bool | int = False, seed=42,
            resume: bool = True, force_resume: bool = False,
            resume_every_n_train_steps: int = 500) -> Dict[str, Any]:
        '''
        Args:
            resume: continue from `checkpoints/last.ckpt` when this run dir already holds a partial
                attempt. Requires a stable `run_uid` (a sweep supplies one). Refuses when the config
                fingerprint moved -- see `_resolve_resume_checkpoint`.
            force_resume: resume anyway despite a config change, and record that in `run.json`.
            resume_every_n_train_steps: cadence of the rolling `last.ckpt`, i.e. the most training
                an interruption can cost.
        '''
        pl.seed_everything(seed)
        self.setup_exeperiment_run_dir()
        self.console_log_path = self.experiment_run_dir / "console.log"

        resume_ckpt = self._resolve_resume_checkpoint(resume=resume, force_resume=force_resume)
        loggers = self.setup_loggers()
        with capture_console_output(self.console_log_path):
            print(f"Loaded configuration: {self.config_path}")
            print(f"Console logs: {self.console_log_path}")
            print(f"Seed: {seed}")
            print(f"Run dir: {self.experiment_run_dir}")
            self._record_provenance(seed)
            self.log_run_hyperparameters(loggers, seed)

            model = self._build_lit_module()
            data_module = instantiate_from_config(self.config.data)
            self.lit_module = model
            self.data_module = data_module

            callbacks_cfg = self.config.trainer.get("callbacks", [])
            callbacks = build_callbacks(callbacks_cfg)
            if not any(isinstance(callback, OverallTrainingProgressBar) for callback in callbacks):
                callbacks.append(OverallTrainingProgressBar())

            # Everything step-dependent (the consistency ramp, the EMA decay, the LR) is a function
            # of `global_step`, which the checkpoint restores; `ResumeFidelity` covers the one thing
            # Lightning does NOT checkpoint, the RNG state. Together with the sampler's own
            # `state_dict` that is what makes a resumed run follow the uninterrupted trajectory.
            if not any(isinstance(callback, ResumeFidelity) for callback in callbacks):
                callbacks.append(ResumeFidelity())

            checkpoint_callback = self._build_checkpoint_callback()
            if checkpoint_callback is not None:
                callbacks.append(checkpoint_callback)
            if not fast_dev_run:
                callbacks.append(self._build_resume_checkpoint_callback(resume_every_n_train_steps))
            
            trainer_params = OmegaConf.to_container(self.config["trainer"]["params"], resolve=True)
            if fast_dev_run:
                trainer_params["fast_dev_run"] = fast_dev_run
                print(f"fast_dev_run enabled via CLI override: {fast_dev_run}")

            trainer = pl.Trainer(
                logger=loggers,
                callbacks=callbacks,
                **trainer_params,
            )

            if resume_ckpt:
                # Lido do checkpoint, nao de `trainer.global_step` depois do fit -- depois do fit
                # global_step ja e o passo FINAL, e o que interessa registrar e onde a retomada
                # comecou, para que a analise saiba que este run foi interrompido e onde.
                try:
                    self.resumed_from_step = int(torch.load(
                        resume_ckpt, map_location="cpu", weights_only=False)["global_step"])
                except Exception as error:
                    print(f"Nao foi possivel ler o passo do checkpoint: {error}")
                print(f"Retomando de {resume_ckpt} (passo {self.resumed_from_step})")
            try:
                trainer.fit(model, data_module, ckpt_path=resume_ckpt)
            except BaseException as error:
                # `last.ckpt` is deliberately left in place: it is what the next attempt resumes
                # from. Recording the failure is what lets a sweep tell "never ran" from "died".
                self.write_run_metadata(status="failed", failed_at=datetime.now().isoformat(),
                                        error=f"{type(error).__name__}: {error}",
                                        last_step=int(trainer.global_step))
                raise
            fit_metrics = {
                k: v
                for k, v in trainer.callback_metrics.items()
                if k.startswith("train/") or k.startswith("val/")
            }

            # `fast_dev_run` disables checkpointing entirely, so there is no "best" to restore.
            ckpt_path = None
            if checkpoint_callback is not None and not fast_dev_run and checkpoint_callback.best_model_path:
                ckpt_path = "best"
                self.best_checkpoint_path = checkpoint_callback.best_model_path
                print(f"Testing best checkpoint: {self.best_checkpoint_path}")

            test_results = trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt_path)
            test_metrics = test_results[0] if test_results else {}

            _print_metric_block("Final train/validation metrics", fit_metrics)
            _print_metric_block("Final test metrics", test_metrics)

            # The run completed, so the crash-insurance snapshot has no reader left. Deleting it
            # keeps the resume machinery free of disk cost: one extra checkpoint while a run is
            # live, none once it is done.
            if not fast_dev_run and self.last_checkpoint_path.exists():
                self.last_checkpoint_path.unlink()

            self.write_run_metadata(
                status="done", finished_at=datetime.now().isoformat(),
                final_step=int(trainer.global_step),
                best_checkpoint_path=self.best_checkpoint_path,
                resumed_from_step=self.resumed_from_step,
            )

        return {
            "config_path": str(self.config_path),
            "experiment_name": self.experiment_name,
            "output_dir": str(self.experiment_run_dir),
            "best_checkpoint_path": self.best_checkpoint_path,
            "run_uid": self.experiment_run_dir.name,
            "family": self.family,
            "fit_metrics": {k: _to_scalar(v) for k, v in fit_metrics.items()},
            "test_metrics": {k: _to_scalar(v) for k, v in test_metrics.items()},
        }


if __name__ == "__main__":
    args = get_cli_args(default_config="configs/experiment/unet/vfss-inca-unet.yaml")
    experiment = Experiment(config_path=args.config)
    experiment.run(fast_dev_run=args.fast_dev_run)
