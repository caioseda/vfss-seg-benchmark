import copy
import inspect
import warnings
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..utils import get_obj_from_str, instantiate_from_config
from .samplers import frame_index_metadata, labeled_unlabeled_indices


class _LenGetItemWrapper:
    """Wrap objects with __len__/__getitem__ into a plain dataset-like object."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


def _identity(sample: Any) -> Any:
    return sample


class _CachedDataset:
    '''
    Every sample of `dataset`, loaded once and served from memory.

    Exact only for a deterministic `__getitem__`: a random transform inside the dataset would be
    frozen at its first draw. The VFSS datasets only decode, min-max normalise and resize (the SSL
    augmentations run on the batch, inside the LightningModule), so the cached sample is the one
    `__getitem__` would have returned -- same values, same metadata.

    Attributes not defined here are read from the wrapped dataset, so `video_frame_df` and the
    samplers' helpers keep working.
    '''

    def __init__(self, dataset: Any, num_workers: int = 0):
        self.dataset = dataset
        # `collate_fn=_identity`, not the default: with `batch_size=None` the default is
        # `default_convert`, which turns numpy scalars in the metadata into tensors -- the cached
        # sample would no longer be the one `__getitem__` returns.
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                            collate_fn=_identity)
        self._samples = list(loader)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        return self._samples[index]

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":  # not set yet (e.g. while unpickling): avoid recursing forever
            raise AttributeError(name)
        return getattr(self.dataset, name)


def seed_worker(worker_id: int) -> None:
    """Set deterministic seeds for dataloader workers."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


_LOADER_GENERATOR_SEED = 0


class DataModuleFromConfig(pl.LightningDataModule):
    """Simple configurable datamodule for train/val/test/predict datasets."""

    def __init__(
        self,
        batch_size: int,
        train: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
        predict: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
        wrap: bool = False,
        use_worker_init_fn: bool = False,
        shuffle_val_dataloader: bool = False,
        shuffle_test_loader: bool = False,
        shared_dataset_params: Optional[Dict[str, Any]] = None,
        base_kwargs: Optional[Dict[str, Any]] = None,
        train_batch_sampler: Optional[Dict[str, Any]] = None,
        drop_last_train: bool = False,
        persistent_train_workers: bool = False,
        cache_splits: Optional[List[str]] = None,
        cache_num_workers: int = 8,
    ):
        '''
        Args:
            drop_last_train: drop the last incomplete batch of the plain (shuffled) train loader. A
                trailing batch of 1 frame crashes any BatchNorm that sees a single value per channel
                (the DeepLabV3+ ASPP pooling branch: `[1, 256, 1, 1]`). Ignored when a
                `train_batch_sampler` is set, which owns batch composition.
            persistent_train_workers: keep the train loader's worker processes alive across epochs
                instead of forking `num_workers` new ones at every epoch start. Pays off when epochs
                are short: the E1 tuning trains on one fold (~120 frames, ~15 steps per epoch), and
                the respawn measured ~2 s per epoch against ~0.05 s per step, i.e. most of the wall
                time. Train loader only -- validation runs a handful of times per run, and
                `PatientValidation` builds a fresh val loader each time anyway.

                Not bit-identical to `False`: a persistent iterator draws the workers' base seed from
                the loader generator once, not once per epoch, so the shuffle order diverges from the
                second epoch on. Same distribution, different sequence.
            cache_splits: splits (`train`, `validation`, ...) whose samples are loaded once in
                `setup` and then served from memory (`_CachedDataset`). A VFSS frame costs ~50 ms to
                decode and normalise at 1024x1024, and the same ~230 frames are re-read hundreds of
                times by an E1 tuning run. Memory is ~0.8 MB per frame at 256x256x3 float32, so this
                is for small splits -- not for the X1 unlabeled pool.

                With every split cached, `num_workers: 0` is the faster setting: the workers no
                longer have anything to decode, and shipping each batch back over IPC cost more than
                it saved (measured on the E1 U-Net: 0.125 -> 0.072 s/step).
            cache_num_workers: worker processes used once, to fill the cache. Separate from
                `num_workers` so that a cached run can train with `num_workers: 0` and still decode
                the frames in parallel.
        '''
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.drop_last_train = drop_last_train
        self.persistent_train_workers = bool(persistent_train_workers) and self.num_workers > 0
        self.cache_splits = list(cache_splits or [])
        self.cache_num_workers = int(cache_num_workers)
        self.wrap = wrap

        # Keep backward compatibility with `base_kwargs` while preferring `shared_dataset_params`.
        self.shared_dataset_params = dict(base_kwargs or {})
        if shared_dataset_params:
            self.shared_dataset_params.update(dict(shared_dataset_params))

        split_configs = {
            "train": train,
            "validation": validation,
            "test": test,
            "predict": predict,
        }
        self.dataset_configs = {
            name: self._build_dataset_config(cfg)
            for name, cfg in split_configs.items()
            if cfg is not None
        }

        self._shuffle_val = shuffle_val_dataloader
        self._shuffle_test = shuffle_test_loader
        self.datasets: Dict[str, Any] = {}
        self.train_batch_sampler_config = train_batch_sampler
        self._train_batch_sampler = None
        self._pending_sampler_state: Optional[Dict[str, Any]] = None
        # The DataLoader gets its own generator so that creating an iterator -- which draws a
        # `_base_seed` every time -- does not consume the global RNG stream. Without this, a
        # resumed run is permanently one draw ahead of the uninterrupted one and every
        # augmentation downstream differs. See `src.callbacks.ResumeFidelity`.
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(_LOADER_GENERATOR_SEED)

    def _build_dataset_config(self, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(dataset_cfg)
        merged_params = dict(self.shared_dataset_params)
        merged_params.update(dict(cfg.get("params", {})))
        cfg["params"] = merged_params
        return cfg

    def prepare_data(self) -> None:
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        # Build all configured datasets once and reuse.
        if self.datasets:
            return
        self.datasets = {
            name: instantiate_from_config(cfg)
            for name, cfg in self.dataset_configs.items()
        }
        if self.wrap:
            self.datasets = {
                name: _LenGetItemWrapper(dataset)
                for name, dataset in self.datasets.items()
            }
        unknown = set(self.cache_splits) - set(self.dataset_configs)
        if unknown:
            raise KeyError(f"cache_splits {sorted(unknown)} nao estao configurados "
                           f"(splits: {sorted(self.dataset_configs)})")
        for name in self.cache_splits:
            self.datasets[name] = _CachedDataset(self.datasets[name], num_workers=self.cache_num_workers)

        if self.train_batch_sampler_config is not None and "train" in self.datasets:
            labeled_idx, unlabeled_idx = labeled_unlabeled_indices(self.datasets["train"])
            if not unlabeled_idx:
                # No unlabeled rows at all. With `unlabeled_pool_size` set this should not happen at
                # any label budget -- the pool is drawn from frames that were never annotated, so it
                # survives `label_fraction=1.0`. Reaching this branch therefore means the pool is
                # off, and the batch composition silently stops being comparable across budgets:
                # every row becomes a labelled row, so a step carries twice the supervised gradient
                # of a two-stream step. Loud, because it invalidates the comparison rather than the
                # run.
                warnings.warn(
                    "TwoStreamBatchSampler is configured but the training set has no unlabeled "
                    "rows, so it was replaced by a plain DataLoader. Batches will be fully "
                    "labelled and this run is NOT comparable to two-stream runs. Set "
                    "`unlabeled_pool_size` on the dataset to keep a real unlabeled stream at every "
                    "label budget.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._train_batch_sampler = None
            else:
                # A plain dict, never the DictConfig the YAML produced: OmegaConf validates every
                # assigned value against its primitive types, so injecting numpy arrays raises, and
                # even the index lists would be copied into a ListConfig element by element.
                sampler_cfg = self.train_batch_sampler_config
                if isinstance(sampler_cfg, DictConfig):
                    sampler_cfg = OmegaConf.to_container(sampler_cfg, resolve=True)
                else:
                    sampler_cfg = copy.deepcopy(dict(sampler_cfg))

                params = dict(sampler_cfg.get("params") or {})
                params["labeled_indices"] = labeled_idx
                params["unlabeled_indices"] = unlabeled_idx

                # Video-conditioned unlabeled policies (`same_video`, `temporal_window`) need to know
                # which rows share a video and how far apart they are. Injected only when the sampler
                # declares the parameter, so a sampler that does not care is unaffected.
                sampler_cls = get_obj_from_str(sampler_cfg["target"])
                if "frame_metadata" in inspect.signature(sampler_cls).parameters:
                    params["frame_metadata"] = frame_index_metadata(self.datasets["train"])

                sampler_cfg["params"] = params
                self._train_batch_sampler = instantiate_from_config(sampler_cfg)
                pending = getattr(self, "_pending_sampler_state", None)
                if pending:
                    self._train_batch_sampler.load_state_dict(pending)
                    self._pending_sampler_state = None

    # ------------------------------------------------------------------ resume

    def state_dict(self) -> Dict[str, Any]:
        '''
        Carries the train batch sampler's position into the checkpoint, so `ckpt_path=` resumes the
        epoch where it stopped instead of restarting it. Lightning already saves and reloads the
        DataModule's `state_dict` (`_CheckpointConnector.dump_checkpoint` / `restore_datamodule`),
        so delegating here is all it takes.
        '''
        sampler = self._train_batch_sampler
        if sampler is None or not hasattr(sampler, "state_dict"):
            return {}
        return {"train_batch_sampler": sampler.state_dict(),
                "loader_generator": self._loader_generator.get_state()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        generator_state = (state_dict or {}).get("loader_generator")
        if generator_state is not None:
            self._loader_generator.set_state(generator_state)
        sampler_state = (state_dict or {}).get("train_batch_sampler")
        if not sampler_state:
            return
        sampler = self._train_batch_sampler
        if sampler is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(sampler_state)
        else:
            # `setup` has not built the sampler yet: hold the state and apply it when it appears.
            self._pending_sampler_state = sampler_state

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"Split '{split}' was not configured.")

        if split == "train" and self._train_batch_sampler is not None:
            return DataLoader(
                self.datasets[split],
                batch_sampler=self._train_batch_sampler,
            generator=self._loader_generator,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.persistent_train_workers,
                worker_init_fn=seed_worker if self.use_worker_init_fn else None,
            )

        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=self.drop_last_train and split == "train",
            generator=self._loader_generator,
            pin_memory=True,
            persistent_workers=self.persistent_train_workers and split == "train",
            worker_init_fn=seed_worker if self.use_worker_init_fn else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader("validation", shuffle=self._shuffle_val)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader("test", shuffle=self._shuffle_test)

    def predict_dataloader(self) -> DataLoader:
        return self._make_loader("predict", shuffle=False)
