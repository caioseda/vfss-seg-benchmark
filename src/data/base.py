import copy
import random
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..utils import instantiate_from_config
from .samplers import labeled_unlabeled_indices


class _LenGetItemWrapper:
    """Wrap objects with __len__/__getitem__ into a plain dataset-like object."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


def seed_worker(worker_id: int) -> None:
    """Set deterministic seeds for dataloader workers."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


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
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
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

        if self.train_batch_sampler_config is not None and "train" in self.datasets:
            labeled_idx, unlabeled_idx = labeled_unlabeled_indices(self.datasets["train"])
            sampler_cfg = copy.deepcopy(self.train_batch_sampler_config)
            sampler_cfg.setdefault("params", {})
            sampler_cfg["params"]["labeled_indices"] = labeled_idx
            sampler_cfg["params"]["unlabeled_indices"] = unlabeled_idx
            self._train_batch_sampler = instantiate_from_config(sampler_cfg)

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"Split '{split}' was not configured.")

        if split == "train" and self._train_batch_sampler is not None:
            return DataLoader(
                self.datasets[split],
                batch_sampler=self._train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker if self.use_worker_init_fn else None,
            )

        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
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
