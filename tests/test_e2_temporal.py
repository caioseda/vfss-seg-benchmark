'''
E2: vizinhos temporais, transporte por fluxo, Mean Teacher temporal, teste INCA por `holdout_test`, a
composição 8+8 do sampler e a grade do sweep.

Roda com:  python -m unittest tests.test_e2_temporal -v
'''

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.optical_flow import compute_flow, flow_and_confidence, warp
from src.data.samplers import TwoStreamBatchSampler
from src.data.splits import assign_protocol_split, load_protocol_split, make_protocol_split
from src.data.vfss_frame_dataset import VFSSTemporalNeighborsDataset, VFSSWindowImageDataset
from src.models.ssl import MeanTeacherLitWrapper, TemporalMeanTeacherLitWrapper
from tests.helpers import OPTIMIZER_CFG, TINY_MODEL_CFG, make_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_VIDEOS, FRAMES_ON_DISK, ANNOTATED, SIZE = 3, 6, 2, 32
VARIANT, LABELER = "binary_all", "VC"


def make_dataset(root: Path) -> None:
    '''Mini dataset no layout real. Cada frame tem um quadrado claro que anda 2 px por frame em x.'''
    rows = []
    for video_id in range(1, N_VIDEOS + 1):
        frames_dir = root / "frames" / str(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in range(1, FRAMES_ON_DISK + 1):
            image = np.full((SIZE, SIZE), 30, dtype=np.uint8)
            x0 = 4 + 2 * frame_id
            image[10:20, x0:x0 + 8] = 220
            Image.fromarray(image).save(frames_dir / f"{frame_id}.png")
        target_dir = root / "targets" / "mask" / VARIANT / LABELER
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in range(1, ANNOTATED + 1):
            mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            mask[10:20, 4:12] = 1
            video_frame = f"v{video_id}_f{frame_id}"
            Image.fromarray(mask).save(target_dir / f"{video_frame}.tif")
            rows.append({
                "video_frame": video_frame, "video_id": video_id, "frame_id": frame_id,
                "selected_labeler": LABELER, "paciente_id": video_id, "momento": "pos",
                "procedimento": "total", "split_legado": "train",
                "image_path": f"frames/{video_id}/{frame_id}.png", "target_variant": VARIANT,
                "target_path": f"targets/mask/{VARIANT}/{LABELER}/{video_frame}.tif",
            })
    pd.DataFrame(rows).to_csv(root / "inca-video-frame-dataset.csv", index=False)


class TemporalNeighborsDatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="vfss-e2-test-"))
        make_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def build(self, cls=VFSSTemporalNeighborsDataset, **kwargs):
        params = dict(dataset_path=self.root, size=SIZE, repeat_channels=True, target_variant=VARIANT,
                      split="train", split_mode="legado", use_unlabeled_pool=True)
        params.update(kwargs)
        return cls(**params)

    def row_of(self, dataset, video_id, frame_id):
        df = dataset.video_frame_df
        return int(np.flatnonzero((df.video_id == video_id).to_numpy() & (df.frame_id == frame_id).to_numpy())[0])

    def test_bidirectional_neighbors_and_video_borders(self):
        dataset = self.build(neighbor_offsets=[-1, 1], flow_method="identity")
        middle = dataset[self.row_of(dataset, 1, 4)]
        self.assertEqual(tuple(middle["neighbors"].shape), (2, 3, SIZE, SIZE))
        self.assertEqual(tuple(middle["neighbor_flow"].shape), (2, 2, SIZE, SIZE))
        self.assertEqual(tuple(middle["neighbor_confidence"].shape), (2, 1, SIZE, SIZE))
        self.assertTrue(middle["neighbor_valid"].all())
        self.assertEqual(middle["metadata"]["neighbor_frame_ids"], [3, 5])

        last = dataset[self.row_of(dataset, 1, FRAMES_ON_DISK)]
        self.assertEqual(last["neighbor_valid"].tolist(), [True, False])
        self.assertEqual(last["metadata"]["neighbor_frame_ids"], [FRAMES_ON_DISK - 1, -1])
        self.assertEqual(float(last["neighbors"][1].abs().sum()), 0.0)

    def test_neighbor_is_the_real_previous_frame(self):
        dataset = self.build(neighbor_offsets=[-1], flow_method="identity")
        sample = dataset[self.row_of(dataset, 2, 5)]
        previous = dataset[self.row_of(dataset, 2, 4)]
        torch.testing.assert_close(sample["neighbors"][0], previous["image"])

    def test_labeled_rows_carry_no_neighbors(self):
        dataset = self.build(neighbor_offsets=[-1, 1], flow_method="identity")
        sample = dataset[self.row_of(dataset, 1, 2)]   # anotado e revelado
        self.assertTrue(sample["metadata"]["is_labeled"])
        self.assertFalse(sample["neighbor_valid"].any())
        both = self.build(neighbor_offsets=[-1, 1], flow_method="identity", neighbors_for="all")
        self.assertTrue(both[self.row_of(both, 1, 2)]["neighbor_valid"].all())

    def test_empty_offsets_equal_base_dataset(self):
        temporal = self.build(neighbor_offsets=[])
        base = self.build(cls=VFSSWindowImageDataset)
        self.assertEqual(len(temporal), len(base))
        a, b = temporal[3], base[3]
        self.assertEqual(set(a), set(b))
        torch.testing.assert_close(a["image"], b["image"])

    def test_farneback_transport_beats_no_transport(self):
        dataset = self.build(neighbor_offsets=[-1], flow_method="farneback", flow_preprocess="none")
        sample = dataset[self.row_of(dataset, 1, 4)]
        center = (sample["image"][:1] + 1) / 2
        neighbor = (sample["neighbors"][0, :1] + 1) / 2
        moved = warp(neighbor[None], sample["neighbor_flow"][:1], "dtc_trus")[0]
        self.assertLess(float((moved - center).abs().mean()), float((neighbor - center).abs().mean()))

    def test_invalid_configuration_is_refused(self):
        for kwargs in ({"neighbor_offsets": [0]}, {"neighbor_offsets": [-1, -1]},
                       {"flow_method": "raft"}, {"warp_convention": "x"}, {"neighbors_for": "x"}):
            with self.assertRaises(ValueError, msg=kwargs):
                self.build(**kwargs)


class WarpTests(unittest.TestCase):

    def test_integer_translation_both_conventions(self):
        image = torch.zeros(1, 1, 16, 16)
        image[..., 4:8, 2:6] = 1.0
        shift = 3.0
        expected = torch.zeros_like(image)
        expected[..., 4:8, 5:9] = 1.0
        # dtc_trus: F_{n->t} = +3 em x, amostra em x - F. backward: F_{t->n} = -3, amostra em x + F.
        flow = torch.zeros(1, 2, 16, 16)
        flow[:, 0] = shift
        torch.testing.assert_close(warp(image, flow, "dtc_trus"), expected)
        torch.testing.assert_close(warp(image, -flow, "backward"), expected)

    def test_identity_flow_is_a_no_op(self):
        image = torch.rand(2, 3, 8, 8)
        torch.testing.assert_close(warp(image, torch.zeros(2, 2, 8, 8)), image)
        self.assertEqual(compute_flow(np.zeros((8, 8)), np.ones((8, 8)), method="identity").sum(), 0)

    def test_confidence_is_one_where_warp_explains_image(self):
        frame = np.random.RandomState(0).rand(16, 16).astype(np.float32)
        _, confidence = flow_and_confidence(frame, frame, method="identity")
        torch.testing.assert_close(confidence, torch.ones(1, 16, 16), atol=1e-4, rtol=0)  # grid_sample arredonda


def build(cls=TemporalMeanTeacherLitWrapper, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("total_steps", 1000)
    return cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
               report_class_ids={1: "C2", 3: "C4"}, **kwargs)


def at_step(module, step):
    return mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=step)


def temporal_batch(n_labeled=2, n_unlabeled=2, k=2, size=16, valid=None):
    batch = make_batch(n_labeled=n_labeled, n_unlabeled=n_unlabeled, size=size)
    rows = n_labeled + n_unlabeled
    generator = torch.Generator().manual_seed(1)
    batch["neighbors"] = torch.randn(rows, k, 3, size, size, generator=generator)
    batch["neighbor_flow"] = torch.zeros(rows, k, 2, size, size)
    batch["neighbor_confidence"] = torch.ones(rows, k, 1, size, size)
    if valid is None:
        valid = torch.zeros(rows, k, dtype=torch.bool)
        valid[n_labeled:] = True
    batch["neighbor_valid"] = valid
    return batch


class TemporalScheduleTests(unittest.TestCase):

    def test_zero_in_warmup_then_ramps_from_end_of_warmup(self):
        module = build(temporal_weight=0.5, temporal_warmup_fraction=0.1, temporal_rampup_fraction=None)
        weights = []
        for step in (0, 50, 99, 100, 400, 700, 1000):
            with at_step(module, step):
                weights.append(module.current_temporal_weight())
        self.assertEqual(weights[:3], [0.0, 0.0, 0.0])
        self.assertGreater(weights[3], 0.0)
        self.assertAlmostEqual(weights[3], 0.5 * np.exp(-5.0), places=6)   # rampa começa em 0
        self.assertTrue(all(a <= b for a, b in zip(weights, weights[1:])))
        self.assertAlmostEqual(weights[-1], 0.5, places=6)


class TemporalMeanTeacherTests(unittest.TestCase):

    def test_zero_temporal_weight_reduces_to_mean_teacher(self):
        batch = temporal_batch()
        losses = []
        for cls, extra in ((MeanTeacherLitWrapper, {}), (TemporalMeanTeacherLitWrapper, {"temporal_weight": 0.0})):
            module = build(cls, consistency_warmup_fraction=0.0, **extra)
            module.log = module.log_dict = lambda *a, **k: None
            torch.manual_seed(5)
            with at_step(module, 500):
                losses.append(module.training_step(batch, 0).detach())
        torch.testing.assert_close(losses[0], losses[1])

    def test_temporal_gradient_reaches_student_and_not_teacher(self):
        module = build(temporal_weight=1.0, temporal_warmup_fraction=0.0, consistency_weight=0.0)
        module.log = module.log_dict = lambda *a, **k: None
        with at_step(module, 1000):
            loss = module.training_step(temporal_batch(), 0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.parameters()))
        self.assertTrue(all(p.grad is None for p in module.teacher.parameters()))
        self.assertEqual(module.forward_counts["teacher"], 2 + 4)   # 2 centrais + 2x2 vizinhos

    def test_invalid_neighbors_do_not_contribute(self):
        module = build(temporal_weight=1.0, temporal_warmup_fraction=0.0)
        a = temporal_batch()
        b = temporal_batch()
        b["neighbor_valid"][:, 1] = False
        b["neighbors"][:, 1] = 1e3   # lixo que não pode entrar na perda
        student = torch.softmax(torch.randn(4, 4, 16, 16), 1)
        transported = torch.softmax(torch.randn(4, 4, 16, 16), 1)
        conf = torch.ones(4, 1, 16, 16)
        full = module.temporal_loss(student, transported, conf)
        # Pares com confiança zero em parte da imagem: média ponderada só sobre os pixels confiáveis.
        conf_half = conf.clone()
        conf_half[..., :8] = 0
        partial = module.temporal_loss(student, transported, conf_half)
        manual = (((student - transported) ** 2).mean(1, keepdim=True)[..., 8:]).flatten(1).mean(1).mean()
        torch.testing.assert_close(partial, manual)
        self.assertNotEqual(float(full), float(partial))

        module.log = module.log_dict = lambda *x, **k: None
        with at_step(module, 1000):
            loss_b = module.training_step(b, 0)
        self.assertTrue(torch.isfinite(loss_b))
        self.assertEqual(module.forward_counts["teacher"], 2 + 2)


class HoldoutTestSplitTests(unittest.TestCase):

    def setUp(self):
        rows, video = [], 0
        for patient in range(100):
            video += 1
            rows += [{"paciente_id": patient, "video_id": video, "frame_id": f} for f in range(2)]
        self.frames = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            make_protocol_split(self.frames, seed=42).to_csv(path, index=False)
            self.table = load_protocol_split(path)
        self.partition = self.table.set_index("paciente_id").partition

    def test_holdout_test_assigns_exactly_the_inca_test_partition(self):
        df = assign_protocol_split(self.frames, self.table, holdout_test=True)
        test = set(df[df.split == "test"].paciente_id.astype(str))
        self.assertEqual(test, set(self.partition[self.partition == "test"].index))
        train = set(df[df.split == "train"].paciente_id.astype(str))
        self.assertEqual(train, set(self.partition[self.partition == "dev"].index))
        self.assertFalse(df.split.isna().any())

    def test_holdout_test_and_cv_fold_are_exclusive(self):
        with self.assertRaises(ValueError):
            assign_protocol_split(self.frames, self.table, holdout_test=True, cv_fold=0)


class SamplerCompositionTests(unittest.TestCase):

    def test_unlabeled_batch_size(self):
        sampler = TwoStreamBatchSampler(list(range(40)), list(range(40, 200)),
                                        labeled_batch_size=8, unlabeled_batch_size=8, seed=0)
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 16)
        self.assertTrue(all(i < 40 for i in batch[:8]) and all(i >= 40 for i in batch[8:]))
        with self.assertRaises(ValueError):
            TwoStreamBatchSampler([0], [1], batch_size=10, labeled_batch_size=8, unlabeled_batch_size=8)


class E2SweepTests(unittest.TestCase):

    def setUp(self):
        from src.sweep import Sweep
        self.tmp = tempfile.mkdtemp(prefix="e2-sweep-")
        self.sweep = Sweep("configs/sweep/e2.yaml", log_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grid_and_overrides(self):
        plan = self.sweep.plan
        cells = plan.cells()
        self.assertEqual([c["method"] for c in cells],
                         ["supervised", "fixmatch", "meanteacher", "mt-temporal-prev", "mt-temporal-bi", "mt-temporal-prev3"])
        overrides = plan.overrides(cells[0])
        frozen = plan.frozen_hparams["deeplabv3plus"]
        self.assertEqual(overrides["optimizer"]["params"]["lr"], frozen["lr"])
        shared = overrides["data"]["params"]["shared_dataset_params"]
        self.assertTrue(shared["holdout_test"])
        self.assertIsNone(shared["cv_fold"])
        self.assertIsNone(shared["train_folds"])
        sampler = overrides["data"]["params"]["train_batch_sampler"]["params"]
        self.assertEqual((sampler["labeled_batch_size"], sampler["unlabeled_batch_size"]), (8, 8))

    def test_arch_override_replaces_model_block(self):
        from src.sweep import Sweep
        sweep = Sweep("configs/sweep/e2.yaml", log_dir=self.tmp, overrides={"axes.arch": ["swinunet_tiny"]})
        cell = [c for c in sweep.plan.cells() if c["method"] == "mt-temporal-bi"][0]
        self.assertTrue(sweep.plan.run_id(cell).startswith("swinunet_tiny-mt-temporal-bi"))
        composed = OmegaConf.load(sweep.plan.experiment_config(cell))
        e1 = OmegaConf.load(str(PROJECT_ROOT / "configs/experiment/e1/e1-swinunet-tiny.yaml"))
        self.assertEqual(OmegaConf.to_container(composed.model), OmegaConf.to_container(e1.model))
        self.assertEqual(composed.lit_module.target,
                         "src.models.ssl.temporal_mean_teacher.TemporalMeanTeacherLitWrapper")
        self.assertEqual(list(composed.data.params.train.params.neighbor_offsets), [-1, 1])


class E2TemporalT3Tests(unittest.TestCase):
    '''`mt-temporal-prev3` = `mt-temporal-prev` com o vizinho a 3 frames (T=3), e nada mais.'''

    def test_prev3_differs_from_prev_only_in_gap_and_name(self):
        prev = OmegaConf.to_container(OmegaConf.load(str(PROJECT_ROOT / "configs/experiment/e2/e2-mt-temporal-prev.yaml")))
        prev3 = OmegaConf.to_container(OmegaConf.load(str(PROJECT_ROOT / "configs/experiment/e2/e2-mt-temporal-prev3.yaml")))
        self.assertEqual(prev3["data"]["params"]["train"]["params"]["neighbor_gap"], 3)
        self.assertEqual(prev3["data"]["params"]["train"]["params"]["neighbor_offsets"], [-1])
        prev3["data"]["params"]["train"]["params"]["neighbor_gap"] = 1
        prev3["name"] = prev["name"]
        self.assertEqual(prev3, prev)

    def test_neighbor_is_the_frame_three_steps_back(self):
        ds = VFSSTemporalNeighborsDataset.__new__(VFSSTemporalNeighborsDataset)
        ds.neighbor_offsets, ds.neighbor_gap = [-1], 3
        self.assertEqual(ds.neighbor_frame_ids(10), [7])


class E2CVSweepTests(unittest.TestCase):

    def setUp(self):
        from src.sweep import Sweep
        self.tmp = tempfile.mkdtemp(prefix="e2-cv-sweep-")
        self.sweep = Sweep("configs/sweep/e2-cv.yaml", log_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grid_is_folds_by_methods_with_method_innermost(self):
        cells = self.sweep.plan.cells()
        self.assertEqual(len(cells), 30)
        self.assertEqual([c["fold"] for c in cells[:6]], [0] * 6)
        self.assertEqual([c["method"] for c in cells[:6]],
                         ["supervised", "fixmatch", "meanteacher", "mt-temporal-prev", "mt-temporal-bi", "mt-temporal-prev3"])
        run_ids = [self.sweep.plan.run_id(c) for c in cells]
        self.assertEqual(len(set(run_ids)), 30)
        self.assertEqual(run_ids[0], "deeplabv3plus-supervised-global-f0-s42")

    def test_overrides_hold_out_the_fold_and_never_the_inca_test(self):
        plan = self.sweep.plan
        for cell in plan.cells():
            overrides = plan.overrides(cell)
            shared = overrides["data"]["params"]["shared_dataset_params"]
            self.assertEqual(shared["cv_fold"], cell["fold"])
            self.assertIsNone(shared["train_folds"])
            self.assertFalse(shared["holdout_test"])
            self.assertEqual(overrides["optimizer"]["params"]["lr"], plan.frozen_hparams[cell["arch"]]["lr"])
            sampler = overrides["data"]["params"]["train_batch_sampler"]["params"]
            self.assertEqual((sampler["labeled_batch_size"], sampler["unlabeled_batch_size"]), (8, 8))

    def test_merged_config_is_a_valid_cv_split(self):
        '''YAML do treino final (`holdout_test: true`) + overrides do CV -> split aceito, fold k em 'test'.'''
        plan = self.sweep.plan
        cell = plan.cells()[-1]
        merged = OmegaConf.merge(OmegaConf.load(plan.experiment_config(cell)), plan.overrides(cell))
        shared = merged.data.params.shared_dataset_params
        self.assertFalse(shared.holdout_test)

        table = load_protocol_split(PROJECT_ROOT / str(shared.split_file))
        dev = table[table.partition == "dev"]
        frames = pd.DataFrame([{"paciente_id": p, "video_id": i, "frame_id": 0}
                               for i, p in enumerate(table.paciente_id)])
        df = assign_protocol_split(frames, table, train_folds=shared.train_folds,
                                   cv_fold=shared.cv_fold, holdout_test=shared.holdout_test)
        held_out = set(dev[dev.fold.astype(int) == cell["fold"]].paciente_id.astype(str))
        self.assertEqual(set(df[df.split == "test"].paciente_id.astype(str)), held_out)
        test_inca = set(table[table.partition == "test"].paciente_id.astype(str))
        self.assertTrue(df[df.paciente_id.astype(str).isin(test_inca)].split.isna().all())
        # O pool nao rotulado so le videos de 'train': o fold retido fica fora dele.
        self.assertEqual(list(merged.data.params.train.params.unlabeled_pool_splits), ["train"])
        self.assertTrue(held_out.isdisjoint(set(df[df.split == "train"].paciente_id.astype(str))))


if __name__ == "__main__":
    unittest.main()
