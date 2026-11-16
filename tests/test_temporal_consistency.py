'''
Consistência temporal (src/temporal_consistency.py): direção do warp, casos-limite, ponderação, teste
forward-backward, AUROC por histograma, agregação par -> vídeo -> paciente e o bloco de vídeo com Farneback.

Roda com:  python -m unittest tests.test_temporal_consistency -v
'''

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

from src.temporal_consistency import (
    VideoChunks,
    aggregate_video_patient,
    auroc_from_counts,
    boundary,
    confidence_counts,
    fb_consistency,
    holm,
    overlap_metrics,
    pair_confidence,
    paired_patient_differences,
    tc_pair_metrics,
)

SIZE = 64


def disk(cx: float, cy: float, r: float = 10.0, size: int = SIZE) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2)[None, None]


def uniform_flow(dx: float, dy: float, n: int = 1, size: int = SIZE) -> torch.Tensor:
    flow = torch.zeros(n, 2, size, size)
    flow[:, 0], flow[:, 1] = dx, dy
    return flow


class TestPairMetrics(unittest.TestCase):
    def setUp(self):
        self.P_t, self.P_t1 = disk(30, 32), disk(34, 32)        # a estrutura anda +4 px em x
        self.ones = torch.ones(1, 1, SIZE, SIZE)
        self.valid = torch.ones(1, 1, SIZE, SIZE, dtype=torch.bool)

    def test_correct_forward_flow_aligns_masks(self):
        '''F_{t->t+1} = (+4, 0) com a convenção dtc_trus leva P_t exatamente sobre P_{t+1}.'''
        out, _ = tc_pair_metrics(self.P_t, self.P_t1, uniform_flow(4, 0), self.ones, self.valid)
        self.assertAlmostEqual(out["tc_iou"].item(), 1.0, places=5)
        self.assertAlmostEqual(out["tc_dice"].item(), 1.0, places=5)
        self.assertLess(out["tc_iou_identity"].item(), 0.8)

    def test_wrong_direction_is_worse_than_identity(self):
        out, _ = tc_pair_metrics(self.P_t, self.P_t1, uniform_flow(-4, 0), self.ones, self.valid)
        self.assertLess(out["tc_iou"].item(), out["tc_iou_identity"].item())

    def test_empty_cases(self):
        empty = torch.zeros(1, 1, SIZE, SIZE, dtype=torch.bool)
        both = overlap_metrics(empty, empty)
        self.assertTrue(torch.isnan(both["iou"]).all() and torch.isnan(both["dice"]).all())
        one = overlap_metrics(empty, self.P_t1)
        self.assertEqual(one["iou"].item(), 0.0)
        self.assertEqual(one["dice"].item(), 0.0)
        out, _ = tc_pair_metrics(empty, empty, uniform_flow(0, 0), self.ones, self.valid)
        self.assertTrue(out["both_empty"].item())

    def test_unit_weight_equals_unweighted(self):
        plain = overlap_metrics(self.P_t, self.P_t1)
        weighted = overlap_metrics(self.P_t, self.P_t1, torch.ones(1, 1, SIZE, SIZE))
        for key in ("iou", "dice"):
            self.assertAlmostEqual(plain[key].item(), weighted[key].item(), places=6)

    def test_weight_zero_on_disagreement_gives_one(self):
        disagreement = (self.P_t ^ self.P_t1).float()
        weighted = overlap_metrics(self.P_t, self.P_t1, 1.0 - disagreement)
        self.assertAlmostEqual(weighted["iou"].item(), 1.0, places=6)


class TestConfidence(unittest.TestCase):
    def test_fb_consistent_and_inconsistent(self):
        self.assertTrue(fb_consistency(uniform_flow(3, 1), uniform_flow(-3, -1)).all())
        self.assertFalse(fb_consistency(uniform_flow(3, 1), uniform_flow(3, 1)).any())

    def test_mnoc_is_one_when_warp_explains_image(self):
        I_t = disk(30, 32).float() * 0.8
        I_t1 = disk(34, 32).float() * 0.8
        conf = pair_confidence(I_t, I_t1, uniform_flow(4, 0), uniform_flow(-4, 0))
        self.assertGreater(conf["mnoc"].min().item(), 0.999)
        self.assertTrue(conf["fb_valid"].all())

    def test_boundary_of_disk_is_a_ring(self):
        ring = boundary(disk(32, 32))
        self.assertTrue(ring.any())
        self.assertFalse(ring[0, 0, 32, 32].item())
        self.assertTrue((ring <= disk(32, 32)).all())

    def test_auroc_from_counts(self):
        rng = np.random.default_rng(0)
        conf = torch.from_numpy(rng.random((1, 1, SIZE, SIZE))).float()
        region = torch.ones_like(conf, dtype=torch.bool)
        perfect = confidence_counts(conf, conf < 0.3, region)   # erro só onde a confiança é baixa
        self.assertGreater(auroc_from_counts(perfect), 0.99)
        reversed_ = confidence_counts(conf, conf > 0.7, region)
        self.assertLess(auroc_from_counts(reversed_), 0.01)
        noise = torch.from_numpy(rng.random((1, 1, SIZE, SIZE)) < 0.5)
        self.assertAlmostEqual(auroc_from_counts(confidence_counts(conf, noise, region)), 0.5, delta=0.05)
        self.assertTrue(np.isnan(auroc_from_counts(confidence_counts(conf, conf > 2, region))))


class TestAggregation(unittest.TestCase):
    def test_patient_with_one_long_video_weighs_like_two_short(self):
        rows = ([{"paciente_id": 1, "video_id": 10, "tc_iou": 1.0}] * 100
                + [{"paciente_id": 2, "video_id": 20, "tc_iou": 0.0}] * 2
                + [{"paciente_id": 2, "video_id": 21, "tc_iou": 0.5}] * 10)
        per_patient = aggregate_video_patient(pd.DataFrame(rows), ["tc_iou"]).set_index("paciente_id")
        self.assertAlmostEqual(per_patient.loc[1, "tc_iou"], 1.0)
        self.assertAlmostEqual(per_patient.loc[2, "tc_iou"], 0.25)   # média dos vídeos, não dos pares
        self.assertEqual(per_patient.loc[2, "n_videos"], 2)
        self.assertEqual(per_patient.loc[2, "n_pairs"], 12)
        self.assertAlmostEqual(per_patient.tc_iou.mean(), 0.625)

    def test_nan_pairs_are_skipped_and_quantile(self):
        rows = [{"paciente_id": 1, "video_id": 1, "tc_iou": v} for v in (np.nan, 0.2, 0.4, 0.6, 0.8, 1.0)]
        mean = aggregate_video_patient(pd.DataFrame(rows), ["tc_iou"])
        self.assertAlmostEqual(mean.tc_iou.item(), 0.6)
        p10 = aggregate_video_patient(pd.DataFrame(rows), ["tc_iou"], video_stat=0.1)
        self.assertAlmostEqual(p10.tc_iou.item(), 0.28)

    def test_paired_differences_and_holm(self):
        per_patient = pd.DataFrame(
            [{"paciente_id": p, "method": "a", "tc_iou": 0.9 + 0.001 * p} for p in range(20)]
            + [{"paciente_id": p, "method": "b", "tc_iou": 0.8} for p in range(20)])
        table = paired_patient_differences(per_patient, "tc_iou", [("a", "b")], n_boot=500)
        row = table.iloc[0]
        self.assertEqual(row.n_patients, 20)
        self.assertGreater(row.ci_lo, 0.09)
        self.assertLess(row.p_holm, 0.001)
        np.testing.assert_allclose(holm([0.01, 0.04, np.nan, 0.03]), [0.03, 0.06, np.nan, 0.06])


class _StubDataset:
    '''O mínimo de `VFSSFrameDatasetBase` que `VideoChunks` usa.'''

    def __init__(self, size):
        self.image_transform = T.Resize((size, size))

    def _preprocess_image(self, image):
        image = image.float() / 255.0
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 2 - 1
        return self.image_transform(image)


class TestVideoChunks(unittest.TestCase):
    '''Um vídeo sintético com textura que anda 2 px/frame: blocos cobrem cada par uma vez e o Farneback transporta.'''

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        texture = (rng.random((96, 96 + 40)) * 255).astype(np.uint8)
        texture = np.array(Image.fromarray(texture).resize((96 + 40, 96)).filter(ImageFilter.GaussianBlur(2)))
        self.n_frames = 7
        for f in range(1, self.n_frames + 1):
            frame = texture[:, 2 * f: 2 * f + 96].copy()
            frame[30:60, 30:60] = 240          # a "estrutura", parada no quadro
            Image.fromarray(frame).save(self.root / f"{f}.png")

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_chunks_cover_each_pair_once(self):
        video = {"video_id": 1, "paciente_id": 1, "frames_dir": str(self.root), "n_frames": self.n_frames}
        chunks = VideoChunks(_StubDataset(64), [video], {"method": "farneback", "preprocess": "vfss"}, chunk=4)
        pairs = []
        for i in range(len(chunks)):
            item = chunks[i]
            ids = item["frame_ids"].tolist()
            pairs += list(zip(ids[:-1], ids[1:]))
            self.assertEqual(item["model"].shape, (len(ids), 1, 64, 64))
            self.assertEqual(item["photo"].shape, (len(ids), 1, 64, 64))
            self.assertEqual(item["flow_fw"].shape, (len(ids) - 1, 2, 64, 64))
            self.assertGreaterEqual(item["photo"].min().item(), 0.0)
        self.assertEqual(pairs, [(t, t + 1) for t in range(1, self.n_frames)])

    def test_farneback_reduces_reconstruction_error(self):
        video = {"video_id": 1, "paciente_id": 1, "frames_dir": str(self.root), "n_frames": 3}
        item = VideoChunks(_StubDataset(96), [video], {"method": "farneback", "preprocess": "vfss"}, chunk=4)[0]
        photo = item["photo"]
        conf = pair_confidence(photo[:-1], photo[1:], item["flow_fw"], item["flow_bw"])
        border = slice(10, -10)
        err_warp = (photo[1:] - conf["I_warp"]).abs()[..., border, border].mean()
        err_id = (photo[1:] - photo[:-1]).abs()[..., border, border].mean()
        self.assertLess(err_warp.item(), err_id.item())


if __name__ == "__main__":
    unittest.main()
