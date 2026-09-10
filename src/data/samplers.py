'''
Batch samplers for semi-supervised training, and the policies that decide **where the unlabeled
frames in a batch come from**.

The two-stream structure (a fixed number of labeled + unlabeled rows per batch) is standard: it
comes from the original Mean Teacher code and is copied verbatim by SSL4MIS and its descendants.
What is *not* standard, and is the subject of experiment X1B, is the choice of which unlabeled
frames accompany the labeled ones. On a video dataset that choice is a real degree of freedom:

    global           any unlabeled frame of any training video (what the literature does, because
                     its benchmarks are collections of independent slices, not videos)
    same_video       an unlabeled frame from the same video as one of the batch's labeled frames,
                     at any position in that video
    temporal_window  as above, but within +/- W frames of that labeled frame

See `notebooks/hyphotesis_testing/experiment_1b_unlabeled_sampling_policy.ipynb`.
'''

from collections import Counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler

UNLABELED_POLICIES = ("global", "same_video", "temporal_window")


def labeled_unlabeled_indices(dataset) -> Tuple[List[int], List[int]]:
    '''Split a dataset's row order into (labeled_indices, unlabeled_indices) using its
    `video_frame_df['is_labeled']` column. Works for any `VFSSFrameDatasetBase` subclass.'''
    is_labeled = dataset.video_frame_df['is_labeled'].to_numpy()
    labeled = np.nonzero(is_labeled)[0].tolist()
    unlabeled = np.nonzero(~is_labeled)[0].tolist()
    return labeled, unlabeled


def frame_index_metadata(dataset) -> Dict[str, np.ndarray]:
    '''
    Per-row `video_id` / `frame_id`, positionally aligned with the dataset's index order.

    This is what lets a sampler reason about *where* a frame sits without loading it: the
    video-conditioned policies need to know which rows share a video, and how far apart they are.
    '''
    df = dataset.video_frame_df
    return {
        "video_id": df["video_id"].to_numpy(),
        "frame_id": df["frame_id"].to_numpy(),
    }


class TwoStreamBatchSampler(Sampler[List[int]]):
    '''
    Yield batches combining a fixed number of labeled + unlabeled example indices per batch.

    Generic combined-batch sampler (Mean Teacher / FixMatch / Cross-Pseudo-Supervision-style),
    intentionally decoupled from any specific semi-supervised algorithm: it only knows how to
    interleave two index pools into fixed-composition batches. Pass it to `DataLoader` via
    `batch_sampler=`, not `sampler=`/`batch_size=`.

    One "epoch" is one full pass over the labeled pool (`len(labeled_indices) // labeled_batch_size`
    batches). The labeled stream is what defines the epoch and is untouched by the policy: the same
    labeled frames, in the same order, for every policy and every seed. Only the unlabeled slots
    change.

    **Unlabeled draws are uniform with replacement, for every policy.** Earlier versions streamed a
    reshuffled permutation of the unlabeled pool, which is only definable when the candidate set is
    constant -- under `same_video` / `temporal_window` it changes with every batch. Using one
    mechanism for all policies is what makes X1B a comparison of *candidate sets* rather than of
    sampling machinery. Expected visit counts are unchanged; only their variance is.
    '''

    def __init__(self,
                 labeled_indices: Sequence[int],
                 unlabeled_indices: Sequence[int],
                 batch_size: int,
                 labeled_batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 unlabeled_policy: str = "global",
                 temporal_window: Optional[int] = None,
                 frame_metadata: Optional[Dict[str, np.ndarray]] = None):
        '''
        Args:
            labeled_indices / unlabeled_indices: dataset row positions of each stream. Injected by
                `DataModuleFromConfig` from the training dataset.
            batch_size / labeled_batch_size: batch composition; the remainder is the unlabeled slots.
            shuffle: shuffle the labeled pass order each epoch.
            seed: base seed; the per-epoch RNG is `seed + epoch`, so a run is reproducible and two
                policies sharing a seed see the *same labeled batches*.
            unlabeled_policy: one of `UNLABELED_POLICIES`; see the module docstring.
            temporal_window: W, required by `temporal_window`. An unlabeled frame is eligible when
                `|frame_id - anchor_frame_id| <= W` within the anchor's video.
            frame_metadata: `{'video_id': array, 'frame_id': array}` positionally aligned with the
                dataset, from `frame_index_metadata`. Required by the video-conditioned policies.
        '''
        if not labeled_indices:
            raise ValueError("labeled_indices must be non-empty.")
        if not unlabeled_indices:
            raise ValueError("unlabeled_indices must be non-empty (use a plain DataLoader if there is no unlabeled data).")
        if not (0 < labeled_batch_size <= batch_size):
            raise ValueError("labeled_batch_size must be in (0, batch_size].")
        if unlabeled_policy not in UNLABELED_POLICIES:
            raise ValueError(f"unlabeled_policy must be one of {UNLABELED_POLICIES}, got '{unlabeled_policy}'")

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.unlabeled_policy = unlabeled_policy
        self.temporal_window = temporal_window
        self._epoch = 0  # bumped each __iter__ call, folded into the per-epoch shuffle seed

        self._unlabeled_array = np.asarray(self.unlabeled_indices, dtype=np.int64)

        # How many draws each policy level actually served. A `temporal_window` run that falls back
        # to `same_video` half the time is only half a W-run, and reporting the split is the only way
        # to know -- see `draw_report`.
        self.draw_counts: Counter = Counter()

        self._needs_metadata = unlabeled_policy != "global"
        if self._needs_metadata:
            if frame_metadata is None:
                raise ValueError(
                    f"unlabeled_policy='{unlabeled_policy}' needs `frame_metadata` "
                    "(from `src.data.samplers.frame_index_metadata`) to know which rows share a video."
                )
            if unlabeled_policy == "temporal_window":
                if temporal_window is None or int(temporal_window) < 1:
                    raise ValueError("temporal_window must be an integer >= 1 for unlabeled_policy='temporal_window'.")
                self.temporal_window = int(temporal_window)
            self._build_video_index(frame_metadata)

    # ------------------------------------------------------------------ construction

    def _build_video_index(self, frame_metadata: Dict[str, np.ndarray]) -> None:
        '''Group the unlabeled rows by video, sorted by frame id so a window is a slice.'''
        self._video_of = np.asarray(frame_metadata["video_id"])
        self._frame_of = np.asarray(frame_metadata["frame_id"])

        n_rows = len(self._video_of)
        for name, indices in (("labeled", self.labeled_indices), ("unlabeled", self.unlabeled_indices)):
            if indices and max(indices) >= n_rows:
                raise ValueError(
                    f"{name}_indices point past the end of frame_metadata "
                    f"({max(indices)} >= {n_rows}); the metadata is not aligned with the dataset."
                )

        unlabeled_videos = self._video_of[self._unlabeled_array]
        order = np.lexsort((self._frame_of[self._unlabeled_array], unlabeled_videos))
        sorted_indices = self._unlabeled_array[order]
        sorted_videos = unlabeled_videos[order]

        self._unlabeled_by_video: Dict[int, np.ndarray] = {}
        self._frames_by_video: Dict[int, np.ndarray] = {}
        boundaries = np.flatnonzero(np.diff(sorted_videos)) + 1
        for group in np.split(np.arange(len(sorted_indices)), boundaries):
            if group.size == 0:
                continue
            video_id = int(sorted_videos[group[0]])
            self._unlabeled_by_video[video_id] = sorted_indices[group]
            self._frames_by_video[video_id] = self._frame_of[sorted_indices[group]]

    # ------------------------------------------------------------------ policy

    def _candidates_for(self, anchor: int) -> Tuple[np.ndarray, str]:
        '''
        Eligible unlabeled rows for one unlabeled slot, given the labeled frame it is anchored to,
        plus the policy level that actually produced them.

        Falls back one level at a time -- `temporal_window` -> `same_video` -> `global` -- rather
        than silently widening W or dropping the slot. A batch must keep its composition, and the
        fallback has to be *visible* (`draw_counts`) instead of quietly turning a W=3 run into a
        partly-global one.
        '''
        video_id = int(self._video_of[anchor])
        same_video = self._unlabeled_by_video.get(video_id)

        if same_video is None or same_video.size == 0:
            # The anchor's video contributed no unlabeled frames (every frame of it is annotated and
            # revealed, or the pool sub-sampling skipped it).
            return self._unlabeled_array, "global"

        if self.unlabeled_policy == "same_video":
            return same_video, "same_video"

        frames = self._frames_by_video[video_id]
        anchor_frame = int(self._frame_of[anchor])
        low = np.searchsorted(frames, anchor_frame - self.temporal_window, side="left")
        high = np.searchsorted(frames, anchor_frame + self.temporal_window, side="right")
        window = same_video[low:high]
        # The anchor is a revealed labeled frame, so it is not in the unlabeled pool -- but a dataset
        # built differently could break that, and training a frame against itself is not consistency.
        window = window[window != anchor]

        if window.size == 0:
            return same_video, "same_video"
        return window, "temporal_window"

    def _draw_unlabeled(self, labeled_batch: List[int], rng: np.random.RandomState) -> List[int]:
        if self.unlabeled_policy == "global":
            self.draw_counts["global"] += self.unlabeled_batch_size
            picks = rng.randint(0, len(self._unlabeled_array), size=self.unlabeled_batch_size)
            return self._unlabeled_array[picks].tolist()

        # Each unlabeled slot is anchored to one labeled frame of the same batch, round-robin, so
        # the anchors are spread over the batch's labeled frames whatever the two sizes are.
        picks: List[int] = []
        for slot in range(self.unlabeled_batch_size):
            anchor = labeled_batch[slot % len(labeled_batch)]
            candidates, level = self._candidates_for(anchor)
            self.draw_counts[level] += 1
            picks.append(int(candidates[rng.randint(0, candidates.size)]))
        return picks

    # ------------------------------------------------------------------ reporting

    def reset_draw_counts(self) -> None:
        self.draw_counts = Counter()

    def draw_report(self) -> Dict[str, float]:
        '''
        Share of unlabeled draws served by each policy level since the last reset.

        `served_as_requested` is the number that matters when reading X1B: a `temporal_window` cell
        whose value is 0.6 is 60% a W-run and 40% a same-video run, and its result has to be read
        that way.
        '''
        total = sum(self.draw_counts.values())
        report = {level: self.draw_counts.get(level, 0) / total if total else 0.0
                  for level in UNLABELED_POLICIES}
        report["total_draws"] = total
        report["served_as_requested"] = report.get(self.unlabeled_policy, 0.0)
        return report

    # ------------------------------------------------------------------ iteration

    def __len__(self) -> int:
        return len(self.labeled_indices) // self.labeled_batch_size

    def _epoch_rng(self) -> np.random.RandomState:
        if self.seed is None:
            return np.random.RandomState()
        return np.random.RandomState(self.seed + self._epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._epoch_rng()
        labeled_order = self.labeled_indices if not self.shuffle else rng.permutation(self.labeled_indices).tolist()

        n_batches = len(labeled_order) // self.labeled_batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * self.labeled_batch_size
            labeled_batch = labeled_order[start:start + self.labeled_batch_size]
            yield labeled_batch + self._draw_unlabeled(labeled_batch, rng)

        self._epoch += 1
