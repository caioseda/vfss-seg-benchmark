from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler


def labeled_unlabeled_indices(dataset) -> Tuple[List[int], List[int]]:
    '''Split a dataset's row order into (labeled_indices, unlabeled_indices) using its
    `video_frame_df['is_labeled']` column. Works for any `VFSSFrameDatasetBase` subclass.'''
    is_labeled = dataset.video_frame_df['is_labeled'].to_numpy()
    labeled = np.nonzero(is_labeled)[0].tolist()
    unlabeled = np.nonzero(~is_labeled)[0].tolist()
    return labeled, unlabeled


class TwoStreamBatchSampler(Sampler[List[int]]):
    '''
    Yield batches combining a fixed number of labeled + unlabeled example indices per batch.

    Generic combined-batch sampler (Mean Teacher / FixMatch / Cross-Pseudo-Supervision-style),
    intentionally decoupled from any specific semi-supervised algorithm: it only knows how to
    interleave two index pools into fixed-composition batches. Pass it to `DataLoader` via
    `batch_sampler=`, not `sampler=`/`batch_size=`.

    One "epoch" is one full pass over the labeled pool (`len(labeled_indices) // labeled_batch_size`
    batches); the unlabeled pool is treated as an infinite, self-reshuffling stream, so it need not be
    an exact multiple of `unlabeled_batch_size` and can be arbitrarily larger than the labeled pool.
    '''

    def __init__(self,
                 labeled_indices: Sequence[int],
                 unlabeled_indices: Sequence[int],
                 batch_size: int,
                 labeled_batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        if not labeled_indices:
            raise ValueError("labeled_indices must be non-empty.")
        if not unlabeled_indices:
            raise ValueError("unlabeled_indices must be non-empty (use a plain DataLoader if there is no unlabeled data).")
        if not (0 < labeled_batch_size <= batch_size):
            raise ValueError("labeled_batch_size must be in (0, batch_size].")

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0  # bumped each __iter__ call, folded into the per-epoch shuffle seed

    def __len__(self) -> int:
        return len(self.labeled_indices) // self.labeled_batch_size

    def _epoch_rng(self) -> np.random.RandomState:
        if self.seed is None:
            return np.random.RandomState()
        return np.random.RandomState(self.seed + self._epoch)

    def _eternal_shuffle(self, pool: List[int], rng: np.random.RandomState) -> Iterator[int]:
        while True:
            order = pool if not self.shuffle else rng.permutation(pool).tolist()
            for i in order:
                yield i

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._epoch_rng()
        labeled_order = self.labeled_indices if not self.shuffle else rng.permutation(self.labeled_indices).tolist()
        unlabeled_stream = self._eternal_shuffle(self.unlabeled_indices, rng)

        n_batches = len(labeled_order) // self.labeled_batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * self.labeled_batch_size
            labeled_batch = labeled_order[start:start + self.labeled_batch_size]
            unlabeled_batch = [next(unlabeled_stream) for _ in range(self.unlabeled_batch_size)]
            yield labeled_batch + unlabeled_batch

        self._epoch += 1
