'''
The frame-count label regime: revealing N labeled frames per video instead of a fraction of groups.

Unlike `apply_label_regime` (X1's original design, which reveals a *fraction* of whole groups --
e.g. patients), `apply_label_regime_by_count` reveals a *fixed number of frames* within every group,
which is what the frames-per-video variant of X1 needs: annotation counts per video are not uniform,
so "reveal 5 frames per video" has no equivalent fraction that works for every video at once.

Run with:  python -m unittest tests.test_label_regime_by_count -v
'''

import unittest

import pandas as pd

from src.data.vfss_frame_dataset import apply_label_regime_by_count


def make_df(frames_per_video: dict, split: str = "train") -> pd.DataFrame:
    '''One row per (video_id, frame_id); `frames_per_video` maps video_id -> frame count.'''
    rows = []
    for video_id, n_frames in frames_per_video.items():
        for frame_id in range(1, n_frames + 1):
            rows.append({"video_id": video_id, "frame_id": frame_id, "split": split})
    return pd.DataFrame(rows)


class ApplyLabelRegimeByCountTestCase(unittest.TestCase):

    def test_reveals_exactly_frames_per_group_when_enough_exist(self):
        df = make_df({1: 10, 2: 10, 3: 10})
        result = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=3, seed=42)
        revealed = result[result.is_labeled].groupby("video_id").size()
        self.assertTrue((revealed == 3).all())

    def test_short_video_reveals_all_it_has_instead_of_erroring(self):
        df = make_df({1: 2, 2: 10})
        result = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=5, seed=42)
        revealed = result[result.is_labeled].groupby("video_id").size()
        self.assertEqual(revealed.loc[1], 2)   # only had 2 to give
        self.assertEqual(revealed.loc[2], 5)

    def test_no_rows_are_dropped(self):
        df = make_df({1: 4, 2: 6})
        result = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=2, seed=42)
        self.assertEqual(len(result), len(df))

    def test_regimes_are_nested_across_increasing_counts(self):
        df = make_df({v: 10 for v in range(1, 6)})
        revealed_by_k = {}
        for k in (2, 3, 4, 5):
            result = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=k, seed=42)
            revealed_by_k[k] = set(
                zip(result[result.is_labeled].video_id, result[result.is_labeled].frame_id)
            )
        for smaller, larger in zip((2, 3, 4, 5), (3, 4, 5, 5)):
            if smaller == larger:
                continue
            self.assertTrue(
                revealed_by_k[smaller] <= revealed_by_k[larger],
                f"Aninhamento quebrado entre k={smaller} e k={larger}",
            )

    def test_val_and_test_are_untouched_by_default(self):
        df = make_df({1: 10}, split="train")
        df_val = make_df({2: 10}, split="val")
        combined = pd.concat([df, df_val], ignore_index=True)
        result = apply_label_regime_by_count(combined, group_column="video_id", frames_per_group=2, seed=42)
        self.assertTrue(result[result.split == "val"].is_labeled.all())

    def test_same_seed_is_reproducible(self):
        df = make_df({1: 10, 2: 10})
        first = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=3, seed=7)
        second = apply_label_regime_by_count(df, group_column="video_id", frames_per_group=3, seed=7)
        pd.testing.assert_series_equal(first.is_labeled, second.is_labeled)

    def test_rejects_non_positive_frames_per_group(self):
        df = make_df({1: 5})
        with self.assertRaises(ValueError):
            apply_label_regime_by_count(df, group_column="video_id", frames_per_group=0, seed=42)


if __name__ == "__main__":
    unittest.main()
