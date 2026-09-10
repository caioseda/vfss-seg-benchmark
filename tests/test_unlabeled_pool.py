'''
The unlabeled pool: frames that were never annotated.

This is where this dataset departs from the benchmarks the semi-supervised literature is built on.
ACDC and friends are fully annotated and a low-label regime is *simulated* by hiding labels, so at
"100% of the labels" there is nothing unlabeled left and every method collapses onto its supervised
baseline. Here annotation is the scarce resource (~1k annotated frames against ~46k on disk), so the
unlabeled stream is real, and it must survive every label budget -- including 100%.

The tests below pin that property and the two failure modes that would silently break it: pool
frames leaking in from validation/test patients, and annotated frames being counted twice.

Run with:  python -m unittest tests.test_unlabeled_pool -v
'''

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from omegaconf import OmegaConf

from src.data.samplers import TwoStreamBatchSampler, labeled_unlabeled_indices
from src.data.vfss_frame_dataset import (
    VFSSWindowImageDataset,
    build_unlabeled_frame_pool,
    load_video_frame_dataframe,
)
from src.utils import instantiate_from_config

N_VIDEOS = 6
FRAMES_ON_DISK = 12       # per video
ANNOTATED_PER_VIDEO = 4   # the rest exist only as images -- the pool
LABELER = "VC"
VARIANT = "multiclass_c2_c4"


def make_dataset(root: Path) -> pd.DataFrame:
    '''Write a miniature dataset with the same on-disk layout as the real one.'''
    rows = []
    for video_id in range(1, N_VIDEOS + 1):
        frames_dir = root / "frames" / str(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in range(1, FRAMES_ON_DISK + 1):
            Image.fromarray(np.full((32, 32), frame_id * 3, dtype=np.uint8)).save(frames_dir / f"{frame_id}.png")

        target_dir = root / "targets" / "mask" / VARIANT / LABELER
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in range(1, ANNOTATED_PER_VIDEO + 1):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:12, 4:12] = 1
            mask[16:24, 16:24] = 3
            video_frame = f"v{video_id}_f{frame_id}"
            Image.fromarray(mask).save(target_dir / f"{video_frame}.tif")
            rows.append({
                "video_frame": video_frame,
                "video_id": video_id,
                "frame_id": frame_id,
                "selected_labeler": LABELER,
                "paciente_id": video_id,            # one patient per video keeps grouping trivial
                "momento": "pos",
                "procedimento": "total",
                "split_legado": "train",
                "image_path": f"frames/{video_id}/{frame_id}.png",
                "target_variant": VARIANT,
                "target_path": f"targets/mask/{VARIANT}/{LABELER}/{video_frame}.tif",
            })

    df = pd.DataFrame(rows)
    df.to_csv(root / "inca-video-frame-dataset.csv", index=False)
    return df


class UnlabeledPoolTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="vfss-pool-test-"))
        make_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def annotated_df(self, splits=None):
        '''The annotated table with an explicit `split`, so these tests do not depend on the splitter.'''
        df = load_video_frame_dataframe(self.root)
        splits = splits or {video_id: ("train" if video_id <= 4 else "test") for video_id in df.video_id.unique()}
        df["split"] = df.video_id.map(splits)
        return df

    def dataset(self, **kwargs):
        params = dict(
            dataset_path=self.root,
            size=16,
            target_variant=VARIANT,
            target_variants=(VARIANT,),
            mask_interpolation="nearest",
            split_mode="seed",
            split_seed=42,
            split_ratios=(0.7, 0.1, 0.2),
            split_group_column="paciente_id",
            label_group_column="paciente_id",
            label_seed=42,
            window_size=1,
            split="train",
        )
        params.update(kwargs)
        return VFSSWindowImageDataset(**params)


class TestPoolConstruction(UnlabeledPoolTestCase):

    def test_pool_holds_only_never_annotated_frames(self):
        annotated = self.annotated_df()
        pool = build_unlabeled_frame_pool(self.root, annotated, splits=("train",))

        annotated_keys = set(zip(annotated.video_id, annotated.frame_id))
        pool_keys = set(zip(pool.video_id, pool.frame_id))
        self.assertFalse(annotated_keys & pool_keys,
                         "an annotated frame leaked into the unlabeled pool (it would be trained on twice)")

        # 4 train videos * (12 on disk - 4 annotated)
        self.assertEqual(len(pool), 4 * (FRAMES_ON_DISK - ANNOTATED_PER_VIDEO))
        self.assertFalse(pool.is_labeled.any())
        self.assertFalse(pool.has_target.any(), "pool frames have no ground truth at all")
        self.assertTrue(pool.target_path.isna().all())

    def test_pool_never_touches_val_or_test_patients(self):
        '''The unlabeled pool is a leakage surface exactly like the labeled data is.'''
        annotated = self.annotated_df()
        pool = build_unlabeled_frame_pool(self.root, annotated, splits=("train",))

        test_videos = set(annotated[annotated.split == "test"].video_id)
        self.assertFalse(set(pool.video_id) & test_videos)

    def test_stride_and_size_are_respected(self):
        annotated = self.annotated_df()

        strided = build_unlabeled_frame_pool(self.root, annotated, splits=("train",), frame_stride=4)
        self.assertEqual(len(strided), 4 * 2)   # ceil(8 / 4) per video

        capped = build_unlabeled_frame_pool(self.root, annotated, splits=("train",), pool_size=10, seed=1)
        self.assertEqual(len(capped), 10)

    def test_sampling_is_reproducible_for_a_fixed_seed(self):
        annotated = self.annotated_df()
        first = build_unlabeled_frame_pool(self.root, annotated, pool_size=10, seed=7)
        second = build_unlabeled_frame_pool(self.root, annotated, pool_size=10, seed=7)
        other = build_unlabeled_frame_pool(self.root, annotated, pool_size=10, seed=8)

        self.assertEqual(list(first.video_frame), list(second.video_frame))
        self.assertNotEqual(list(first.video_frame), list(other.video_frame))


class TestPoolInDataset(UnlabeledPoolTestCase):

    def test_unlabeled_stream_survives_a_full_label_budget(self):
        '''
        The property the redesigned X1 rests on. Without the pool, `label_fraction=1.0` leaves zero
        unlabeled rows, `TwoStreamBatchSampler` cannot be built, batches become fully labelled, and
        the 100% cell stops being comparable to the others -- besides making the semi-supervised
        methods identical to the supervised baseline by construction.
        '''
        for label_fraction in (0.25, 1.0):
            with self.subTest(label_fraction=label_fraction):
                dataset = self.dataset(label_fraction=label_fraction, unlabeled_pool_size=20)
                labeled, unlabeled = labeled_unlabeled_indices(dataset)

                self.assertTrue(labeled)
                self.assertTrue(unlabeled, "no unlabeled rows: the pool did not reach the dataset")

                sampler = TwoStreamBatchSampler(labeled, unlabeled, batch_size=4,
                                                labeled_batch_size=2, seed=0)
                batch = next(iter(sampler))
                self.assertEqual(len(batch), 4)
                self.assertEqual(sum(index in set(labeled) for index in batch), 2,
                                 "batch composition must be the same at every label budget")

    def test_pool_is_off_by_default(self):
        '''Existing configs must keep seeing only annotated frames.'''
        dataset = self.dataset(label_fraction=1.0)
        self.assertTrue(dataset.video_frame_df.has_target.all())
        self.assertEqual(len(dataset), len(dataset.video_frame_df[dataset.video_frame_df.has_target]))

    def test_uncapped_pool_is_not_the_same_as_no_pool(self):
        '''
        `unlabeled_pool_size` is a cap, so None means "no cap" -- but None also used to be the only
        way to say "no pool". X1B needs an uncapped pool (temporal windows need dense neighbours),
        and without `use_unlabeled_pool` it would silently get no pool at all and fall back to a
        fully-labelled batch.
        '''
        uncapped = self.dataset(label_fraction=1.0, use_unlabeled_pool=True, unlabeled_pool_size=None)
        capped = self.dataset(label_fraction=1.0, unlabeled_pool_size=5)
        off = self.dataset(label_fraction=1.0)

        n_uncapped = int((~uncapped.video_frame_df.has_target).sum())
        self.assertGreater(n_uncapped, 5, "an uncapped pool must hold every eligible frame")
        self.assertEqual(int((~capped.video_frame_df.has_target).sum()), 5)
        self.assertEqual(int((~off.video_frame_df.has_target).sum()), 0)

        # And the explicit off switch still wins over a cap.
        disabled = self.dataset(label_fraction=1.0, use_unlabeled_pool=False, unlabeled_pool_size=5)
        self.assertEqual(int((~disabled.video_frame_df.has_target).sum()), 0)

    def test_hidden_targets_are_exposed_only_where_they_exist(self):
        '''
        `expose_hidden_targets` is what makes pseudo-label quality measurable during training: the
        label regime *hides* ground truth rather than deleting it. Pool frames have none, and must
        say so -- scoring a pseudo-label against their all-zeros placeholder would report a
        confidently wrong number.
        '''
        dataset = self.dataset(label_fraction=0.5, unlabeled_pool_size=20,
                               expose_hidden_targets=True)

        seen = {"hidden": False, "pool": False}
        for index in range(len(dataset)):
            sample = dataset[index]
            metadata = sample["metadata"]
            self.assertIn("hidden_segmentation", sample)

            if metadata["is_labeled"]:
                continue

            if metadata["has_target"]:            # annotated frame, label hidden by the regime
                seen["hidden"] = True
                self.assertTrue(metadata["has_hidden_target"])
                self.assertGreater(sample["hidden_segmentation"].sum().item(), 0,
                                   "hidden ground truth was not loaded")
                self.assertEqual(sample["segmentation"].sum().item(), 0,
                                 "the supervised target of a hidden-label row must stay a placeholder")
            else:                                  # frame from the pool: never annotated
                seen["pool"] = True
                self.assertFalse(metadata["has_hidden_target"])

        self.assertTrue(seen["hidden"], "no hidden-label annotated frame in the split; test is vacuous")
        self.assertTrue(seen["pool"], "no pool frame in the split; test is vacuous")


class TestDataModuleWiring(UnlabeledPoolTestCase):
    '''
    The path a real run takes: YAML -> OmegaConf -> DataModule -> sampler.

    Constructing the sampler directly (as the other suites do) misses everything that happens in
    between. It missed, for one, that a `DictConfig` validates every assigned value against its
    primitive types, so injecting the `frame_metadata` arrays the video-conditioned policies need
    raised `UnsupportedValueType` -- a failure no unit test could see and every real run would hit.
    '''

    def datamodule(self, **sampler_params):
        params = {"batch_size": 4, "labeled_batch_size": 2, "seed": 0}
        params.update(sampler_params)
        config = OmegaConf.create({
            "target": "src.data.base.DataModuleFromConfig",
            "params": {
                "batch_size": 4,
                "num_workers": 0,
                "shared_dataset_params": {
                    "dataset_path": str(self.root),
                    "size": 16,
                    "target_variant": VARIANT,
                    "target_variants": [VARIANT],
                    "mask_interpolation": "nearest",
                    "split_mode": "seed",
                    "split_seed": 42,
                    "split_ratios": [0.7, 0.1, 0.2],
                    "split_group_column": "paciente_id",
                    "label_group_column": "paciente_id",
                    "label_seed": 42,
                    "label_fraction": 1.0,
                    "window_size": 1,
                    "unlabeled_pool_size": 20,
                },
                "train": {"target": "src.data.vfss_frame_dataset.VFSSIncaTrain"},
                "train_batch_sampler": {
                    "target": "src.data.samplers.TwoStreamBatchSampler",
                    "params": params,
                },
            },
        })
        datamodule = instantiate_from_config(config)
        datamodule.setup()
        return datamodule

    def test_video_conditioned_policy_survives_the_omegaconf_round_trip(self):
        for policy, window in [("same_video", None), ("temporal_window", 5)]:
            with self.subTest(policy=policy):
                datamodule = self.datamodule(unlabeled_policy=policy, temporal_window=window)
                sampler = datamodule._train_batch_sampler

                self.assertIsInstance(sampler, TwoStreamBatchSampler)
                self.assertEqual(sampler.unlabeled_policy, policy)
                # The metadata actually arrived: without it the policy could not be served at all.
                self.assertTrue(sampler._unlabeled_by_video)

                videos = datamodule.datasets["train"].video_frame_df.video_id.to_numpy()
                for batch in sampler:
                    labeled, unlabeled = batch[:2], batch[2:]
                    for slot, index in enumerate(unlabeled):
                        self.assertEqual(videos[index], videos[labeled[slot % len(labeled)]])

    def test_global_policy_needs_no_metadata(self):
        sampler = self.datamodule(unlabeled_policy="global")._train_batch_sampler
        self.assertEqual(sampler.unlabeled_policy, "global")
        self.assertEqual(len(next(iter(sampler))), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
