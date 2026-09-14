'''
Correctness tests for the ACDC dataset (`src/data/acdc_dataset.py`).

The ACDC notebook is the acceptance test for the SSL port -- which only works if the dataset itself
is right. Every bug pinned below was either observed here or is one the surrounding code cannot
detect:

  - **No training augmentation.** Observed: the supervised baseline at the 10% budget reached train
    Dice 0.987 against volumetric validation Dice 0.70 *and falling*, ~15 points under the published
    baseline. Nothing raised; the loss curve looked healthy.
  - **The volume dataset silently disagreeing with the slice dataset** on channel layout or input
    range, so the model is validated on a distribution it was never trained on.
  - **`is_labeled` in a non-bool dtype**: `src/data/samplers.py::labeled_unlabeled_indices` does
    `~is_labeled`, and on an object array of Python bools that yields -2/-1, both truthy, so *both*
    index lists come out wrong with no error.
  - **A random labeled subset instead of the list prefix**, which is a different experiment at the
    same budget.

Builds a synthetic ACDC root, so the suite stays offline and CPU-fast.

Run with:  python -m unittest tests.test_acdc_dataset -v
'''

import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch

from src.data.acdc_dataset import (ACDCSliceDataset, ACDCTrain, ACDCVal, ACDCValVolumes,
                                   random_generator)
from src.data.samplers import frame_index_metadata, labeled_unlabeled_indices

DEPTH, HEIGHT, WIDTH = 4, 20, 24
N_TRAIN, N_VAL = 6, 2


class ACDCFixture(unittest.TestCase):
    '''A synthetic ACDC root with the same layout `scripts/prepare_acdc.py` writes.'''

    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(tempfile.mkdtemp(prefix="acdc-test-"))
        (cls.root / "volumes").mkdir(parents=True)
        (cls.root / "lists").mkdir(parents=True)

        rng = np.random.default_rng(0)
        cases = {"train": [], "val": []}
        for split, count in (("train", N_TRAIN), ("val", N_VAL)):
            for index in range(count):
                case = f"patient{index + 1:03d}_frame{1 if split == 'train' else 2:02d}"
                cases[split].append(case)
                image = rng.random((DEPTH, HEIGHT, WIDTH)).astype(np.float32)
                label = np.zeros((DEPTH, HEIGHT, WIDTH), dtype=np.uint8)
                label[:, 5:12, 6:14] = 1
                label[:, 8:10, 9:11] = 2
                np.savez_compressed(cls.root / "volumes" / f"{case}.npz", image=image, label=label)

        for split in ("train", "val", "test"):
            source = cases["train"] if split == "train" else cases["val"]
            (cls.root / "lists" / f"{split}.list").write_text("\n".join(source) + "\n")

        slices = [f"{case}_slice_{i}" for case in cases["train"] for i in range(DEPTH)]
        (cls.root / "lists" / "train_slices.list").write_text("\n".join(slices) + "\n")
        cls.total_slices = len(slices)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)


class TestAugmentation(ACDCFixture):
    '''The omission that put the whole reproduction 15 points low.'''

    def test_training_split_is_augmented_and_validation_is_not(self):
        train = ACDCTrain(root=self.root, augment=None)
        self.assertTrue(train.augment, "the train split must be augmented (reference RandomGenerator)")
        # Stochastic: two reads of the same row must differ. Without this the supervised baseline
        # memorises its handful of labeled slices.
        reads = {train[0]["image"].numpy().tobytes() for _ in range(12)}
        self.assertGreater(len(reads), 1, "training reads are identical -- augmentation is a no-op")

        validation = ACDCVal(root=self.root, augment=None)
        self.assertFalse(validation.augment, "validation must be deterministic")
        self.assertTrue(torch.equal(validation[0]["image"], validation[0]["image"]))

    def test_image_and_label_receive_the_same_transform(self):
        '''
        The bug that would make augmentation *worse* than none: a mask rotated differently from its
        image trains the model against a lie, and still produces a falling loss.

        Checked by feeding the label in as the image -- any transform applied consistently leaves
        the two identical, and any inconsistency separates them.
        '''
        label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        label[4:9, 3:15] = 1          # deliberately asymmetric, so a rotation is detectable
        label[10:12, 2:5] = 2

        for seed in range(25):
            rng = np.random.default_rng(seed)
            image_out, label_out = random_generator(label.astype(np.float64), label, rng)
            np.testing.assert_array_equal(
                image_out.astype(np.uint8), label_out,
                err_msg=f"seed {seed}: image and label were transformed differently",
            )

    def test_augmentation_can_be_disabled(self):
        train = ACDCTrain(root=self.root, augment=False)
        self.assertTrue(torch.equal(train[0]["image"], train[0]["image"]))


class TestLabelRegime(ACDCFixture):

    def test_labeled_rows_are_the_prefix_of_the_slice_list(self):
        # SSL4MIS takes `range(0, labeled_slice_num)`. A random draw of the same size is a
        # different experiment: another set of slices, another number.
        dataset = ACDCSliceDataset(root=self.root, split="train", labeled_slices=5)
        labeled, unlabeled = labeled_unlabeled_indices(dataset)
        self.assertEqual(labeled, [0, 1, 2, 3, 4])
        self.assertEqual(len(unlabeled), self.total_slices - 5)

    def test_is_labeled_is_numpy_bool(self):
        # `labeled_unlabeled_indices` does `~is_labeled`. On an object array of Python bools that
        # gives -2/-1 -- both truthy -- and both index lists come out wrong, silently.
        dataset = ACDCSliceDataset(root=self.root, split="train", labeled_slices=5)
        column = dataset.video_frame_df["is_labeled"]
        self.assertEqual(column.dtype, np.bool_)
        self.assertTrue(np.array_equal(~column.to_numpy(), np.logical_not(column.to_numpy())))

    def test_hidden_rows_carry_a_zero_placeholder_and_the_real_mask_separately(self):
        dataset = ACDCSliceDataset(root=self.root, split="train", labeled_slices=1,
                                   expose_hidden_targets=True, augment=False)
        hidden = dataset[self.total_slices - 1]
        self.assertFalse(hidden["metadata"]["is_labeled"])
        self.assertEqual(hidden["segmentation"].sum().item(), 0,
                         "a hidden row must expose the all-zeros placeholder to the loss")
        self.assertGreater(hidden["hidden_segmentation"].sum().item(), 0,
                           "the real mask must still be reachable for diagnostics")

    def test_sampler_metadata_is_present(self):
        # Injected by DataModuleFromConfig whenever the sampler declares it -- which
        # TwoStreamBatchSampler does, even under `unlabeled_policy: global`.
        dataset = ACDCSliceDataset(root=self.root, split="train", labeled_slices=5)
        metadata = frame_index_metadata(dataset)
        self.assertEqual(len(metadata["video_id"]), len(dataset))
        self.assertEqual(len(metadata["frame_id"]), len(dataset))


class TestSliceAndVolumeAgree(ACDCFixture):
    '''
    The network trains on slices and is validated on volumes. If the two disagree about layout or
    input range the model is evaluated on a distribution it never saw -- every volumetric number
    drops and nothing raises.
    '''

    def test_volume_layout_and_range_match_the_slice_dataset(self):
        for channels in (1, 3):
            with self.subTest(image_channels=channels):
                slices = ACDCVal(root=self.root, image_channels=channels, augment=False)
                volumes = ACDCValVolumes(root=self.root, image_channels=channels)

                slice_sample = slices[0]["image"]
                volume_sample = volumes[0]["image"]

                self.assertEqual(volume_sample.ndim, 4, "volumes must be [D, C, H, W]")
                self.assertEqual(volume_sample.shape[1], channels)
                self.assertEqual(slice_sample.shape[0], channels)
                # Both must land in [-1, 1]; the stored volumes are [0, 1].
                self.assertGreaterEqual(volume_sample.min().item(), -1.0001)
                self.assertLessEqual(volume_sample.max().item(), 1.0001)
                self.assertLess(volume_sample.min().item(), 0.0,
                                "the volume dataset skipped the [0,1] -> [-1,1] mapping")

    def test_volumes_are_native_resolution(self):
        volumes = ACDCValVolumes(root=self.root)
        sample = volumes[0]
        self.assertEqual(tuple(sample["image"].shape), (DEPTH, 1, HEIGHT, WIDTH))
        self.assertEqual(tuple(sample["segmentation"].shape), (DEPTH, HEIGHT, WIDTH))


if __name__ == "__main__":
    unittest.main()
