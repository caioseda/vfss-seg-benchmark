'''
The two throughput options of `DataModuleFromConfig`: `cache_splits` and `persistent_train_workers`.

Both exist for the E1 tuning, where an epoch is ~15 steps and the same ~230 frames are re-read
hundreds of times. Neither may change *what* the model sees: the cache must hand back exactly the
sample `__getitem__` would have, metadata types included, and persistence must stay off the loaders
it was not asked for.

Run with:  python -m unittest tests.test_datamodule_speed -v
'''

import unittest

import numpy as np
import pandas as pd
import torch

from src.data.base import DataModuleFromConfig, _CachedDataset

N_FRAMES = 10


class ToyFrames(torch.utils.data.Dataset):
    '''Deterministic samples whose metadata carries numpy scalars, like `VFSSFrameDatasetBase`.'''

    getitem_calls = 0

    def __init__(self, offset: int = 0):
        self.offset = offset
        self.video_frame_df = pd.DataFrame({"video_id": np.arange(N_FRAMES), "frame_id": np.arange(N_FRAMES)})

    def __len__(self):
        return N_FRAMES

    def __getitem__(self, index):
        type(self).getitem_calls += 1
        value = index + self.offset
        return {
            "image": torch.full((3, 4, 4), float(value)),
            "segmentation": torch.full((4, 4), value % 2, dtype=torch.long),
            "metadata": {"video_id": np.int64(value), "frame_id": int(value), "is_labeled": True},
        }


def _datamodule(**params):
    toy = {"target": f"{__name__}.ToyFrames"}
    return DataModuleFromConfig(batch_size=2, train=toy, validation={**toy, "params": {"offset": 100}},
                                **params)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


class CacheSplitsTest(unittest.TestCase):

    def test_cached_sample_is_the_getitem_sample(self):
        for workers in (0, 2):
            with self.subTest(cache_num_workers=workers):
                dm = _datamodule(num_workers=0, cache_splits=["train", "validation"],
                                 cache_num_workers=workers)
                dm.setup()
                for split in ("train", "validation"):
                    cached = dm.datasets[split]
                    self.assertIsInstance(cached, _CachedDataset)
                    for index in range(N_FRAMES):
                        # numpy scalars must survive: the default `batch_size=None` convert would
                        # have turned `video_id` into a tensor.
                        self.assertTrue(_same(cached[index], cached.dataset[index]), (split, index))

    def test_each_frame_is_loaded_once(self):
        dm = _datamodule(num_workers=0, cache_splits=["train"], cache_num_workers=0)
        dm.setup()
        ToyFrames.getitem_calls = 0
        for _ in range(3):
            for _batch in dm.train_dataloader():
                pass
        self.assertEqual(ToyFrames.getitem_calls, 0)

    def test_attributes_are_read_from_the_wrapped_dataset(self):
        dm = _datamodule(num_workers=0, cache_splits=["train"])
        dm.setup()
        self.assertEqual(len(dm.datasets["train"].video_frame_df), N_FRAMES)

    def test_unconfigured_split_is_refused(self):
        dm = _datamodule(num_workers=0, cache_splits=["test"])
        with self.assertRaises(KeyError):
            dm.setup()

    def test_default_is_no_cache(self):
        dm = _datamodule(num_workers=0)
        dm.setup()
        self.assertIsInstance(dm.datasets["train"], ToyFrames)


class PersistentTrainWorkersTest(unittest.TestCase):

    def test_train_loader_only(self):
        dm = _datamodule(num_workers=2, persistent_train_workers=True)
        dm.setup()
        self.assertTrue(dm.train_dataloader().persistent_workers)
        self.assertFalse(dm.val_dataloader().persistent_workers)

    def test_off_by_default_and_without_workers(self):
        for params in ({"num_workers": 2}, {"num_workers": 0, "persistent_train_workers": True}):
            with self.subTest(**params):
                dm = _datamodule(**params)
                dm.setup()
                self.assertFalse(dm.train_dataloader().persistent_workers)


if __name__ == "__main__":
    unittest.main()
