'''
ASSD/HD95 degenerate-case convention.

A class present in only one of prediction/target (a missed or hallucinated structure) is the worst
possible outcome, not an undefined one. `_max_distance` (the image/volume diagonal) is the sentinel
for that case -- not NaN, which `nanmean`-based aggregation would silently drop from the average
instead of penalizing (see `docs/x1-revisao-mudancas.md`, finding A2). These tests pin the three
degenerate cases and that a real prediction still gets a real (non-sentinel) distance.

Run with:  python -m unittest tests.test_metrics -v
'''

import math
import unittest

import torch

from src.metrics import (
    _max_distance,
    average_symmetric_surface_distance,
    hausdorff_distance_95,
    volumetric_asd,
    volumetric_hd95,
)


def _index_map(size: int, box=None) -> torch.Tensor:
    '''`[1, size, size]` index map, class 1 painted into `box=(top, left, side)` if given.'''
    out = torch.zeros(1, size, size, dtype=torch.long)
    if box is not None:
        top, left, side = box
        out[:, top:top + side, left:left + side] = 1
    return out


class TestPairwiseSurfaceDistances(unittest.TestCase):
    SIZE = 32

    def test_absent_from_both_scores_zero(self):
        pred = _index_map(self.SIZE)
        target = _index_map(self.SIZE)
        assd = average_symmetric_surface_distance(pred, target, num_classes=2)
        hd95 = hausdorff_distance_95(pred, target, num_classes=2)
        self.assertEqual(assd.item(), 0.0)
        self.assertEqual(hd95.item(), 0.0)

    def test_present_in_only_one_scores_the_diagonal_not_nan(self):
        pred = _index_map(self.SIZE)  # empty prediction
        target = _index_map(self.SIZE, box=(4, 4, 8))  # real structure the model missed

        assd = average_symmetric_surface_distance(pred, target, num_classes=2)
        hd95 = hausdorff_distance_95(pred, target, num_classes=2)

        expected = _max_distance((self.SIZE, self.SIZE), spacing=None)
        self.assertFalse(torch.isnan(assd).any())
        self.assertFalse(torch.isnan(hd95).any())
        self.assertAlmostEqual(assd.item(), expected, places=4)
        self.assertAlmostEqual(hd95.item(), expected, places=4)

    def test_hallucinated_structure_also_scores_the_diagonal(self):
        '''Same worst case, mirrored: prediction has it, target does not.'''
        pred = _index_map(self.SIZE, box=(4, 4, 8))
        target = _index_map(self.SIZE)

        assd = average_symmetric_surface_distance(pred, target, num_classes=2)
        expected = _max_distance((self.SIZE, self.SIZE), spacing=None)
        self.assertAlmostEqual(assd.item(), expected, places=4)

    def test_real_overlap_error_scores_a_real_distance_below_the_penalty(self):
        '''Both present, disjoint boxes: a real, finite distance -- not the penalty, not zero.'''
        pred = _index_map(self.SIZE, box=(2, 2, 6))
        target = _index_map(self.SIZE, box=(20, 20, 6))

        assd = average_symmetric_surface_distance(pred, target, num_classes=2)
        penalty = _max_distance((self.SIZE, self.SIZE), spacing=None)
        self.assertGreater(assd.item(), 0.0)
        self.assertLess(assd.item(), penalty)


class TestVolumetricSurfaceDistances(unittest.TestCase):
    SHAPE = (8, 16, 16)  # [D, H, W]

    def _volume(self, box=None) -> torch.Tensor:
        out = torch.zeros(self.SHAPE, dtype=torch.long)
        if box is not None:
            d0, d1, top, left, side = box
            out[d0:d1, top:top + side, left:left + side] = 1
        return out

    def test_present_in_only_one_scores_the_volume_diagonal_not_nan(self):
        pred = self._volume()
        target = self._volume(box=(2, 6, 4, 4, 6))

        asd = volumetric_asd(pred, target, num_classes=2)
        hd95 = volumetric_hd95(pred, target, num_classes=2)

        expected = _max_distance(self.SHAPE, spacing=None)
        self.assertFalse(torch.isnan(asd).any())
        self.assertAlmostEqual(asd.item(), expected, places=4)
        self.assertAlmostEqual(hd95.item(), expected, places=4)

    def test_absent_from_both_scores_zero(self):
        pred = self._volume()
        target = self._volume()
        asd = volumetric_asd(pred, target, num_classes=2)
        self.assertEqual(asd.item(), 0.0)


class TestMaxDistance(unittest.TestCase):
    def test_unit_spacing_is_the_euclidean_diagonal(self):
        self.assertAlmostEqual(_max_distance((3, 4), spacing=None), 5.0)

    def test_spacing_scales_each_dimension(self):
        # 3 rows at spacing 2 -> 6; 4 cols at spacing 1.5 -> 6; diagonal of a 6x6 square.
        self.assertAlmostEqual(_max_distance((3, 4), spacing=(2.0, 1.5)), math.hypot(6.0, 6.0))


if __name__ == "__main__":
    unittest.main()
