'''
E2: DTC-TRUS -- a etapa 2 (`DTCTRUSLitWrapper`) e a ligação com a etapa 1 pelo sweep (`teacher_from`).

Roda com:  python -m unittest tests.test_e2_dtc_trus -v
'''

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.ssl import DTCTRUSLitWrapper, SupervisedLitWrapper
from src.models.ssl.dtc_trus import adaptive_threshold, masked_avg_pool, resize_flow
from tests.helpers import OPTIMIZER_CFG, TINY_FEATURE_MODEL_CFG, TINY_MODEL_CFG, make_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIZE = 16


def save_teacher(directory: Path, model_cfg=TINY_FEATURE_MODEL_CFG, seed: int = 123) -> Path:
    '''Um checkpoint no formato Lightning de um `SupervisedLitWrapper` (state_dict + hiperparâmetros).'''
    torch.manual_seed(seed)
    supervised = SupervisedLitWrapper(model_cfg=model_cfg, optimizer_cfg=OPTIMIZER_CFG, report_class_ids={1: "FG"})
    path = directory / f"teacher-{seed}.ckpt"
    torch.save({"state_dict": supervised.state_dict(), "hyper_parameters": dict(supervised.hparams),
                "global_step": 0}, path)
    return path


def build(teacher_checkpoint, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("total_steps", 1000)
    module = DTCTRUSLitWrapper(model_cfg=TINY_FEATURE_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                               report_class_ids={1: "FG"}, teacher_checkpoint=str(teacher_checkpoint), **kwargs)
    module.log = module.log_dict = lambda *a, **k: None
    return module


def dtc_batch(n_labeled=2, n_unlabeled=2, k=1, size=SIZE, flow_px=0.0):
    '''Lote 2+2 binário com UM vizinho (t-1) válido para TODOS os centrais (`neighbors_for: all`).'''
    batch = make_batch(n_labeled=n_labeled, n_unlabeled=n_unlabeled, size=size)
    batch["segmentation"] = (batch["segmentation"] > 0).long()
    rows = n_labeled + n_unlabeled
    generator = torch.Generator().manual_seed(1)
    batch["neighbors"] = torch.randn(rows, k, 3, size, size, generator=generator)
    batch["neighbor_flow"] = torch.zeros(rows, k, 2, size, size)
    batch["neighbor_flow"][:, :, 0] = flow_px
    batch["neighbor_confidence"] = torch.ones(rows, k, 1, size, size)
    batch["neighbor_valid"] = torch.ones(rows, k, dtype=torch.bool)
    return batch


class DTCTRUSTeacherTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dtc-trus-"))
        self.ckpt = save_teacher(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_teacher_is_the_stage1_network_frozen_and_out_of_the_optimizer(self):
        module = build(self.ckpt)
        stage1 = torch.load(self.ckpt, weights_only=False)["state_dict"]
        for name, value in module.teacher.state_dict().items():
            torch.testing.assert_close(value, stage1[f"model.{name}"])
        self.assertTrue(all(not p.requires_grad for p in module.teacher.parameters()))
        module.train()
        self.assertFalse(module.teacher.training)
        self.assertTrue(module.model.training)
        optimizer_params = {id(p) for p in module.optimizer_param_groups()}
        self.assertTrue(optimizer_params.isdisjoint({id(p) for p in module.teacher.parameters()}))

    def test_student_does_not_inherit_the_teacher_weights(self):
        module = build(self.ckpt)
        differs = [not torch.equal(s, t) for s, t in zip(module.model.parameters(), module.teacher.parameters())]
        self.assertTrue(any(differs))

    def test_missing_or_mismatched_teacher_is_refused(self):
        with self.assertRaisesRegex(ValueError, "teacher_checkpoint"):
            DTCTRUSLitWrapper(model_cfg=TINY_FEATURE_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                              report_class_ids={1: "FG"}, teacher_checkpoint=None)
        other = {"target": TINY_FEATURE_MODEL_CFG["target"], "params": {**TINY_FEATURE_MODEL_CFG["params"], "hidden": 4}}
        with self.assertRaisesRegex(ValueError, "model_cfg"):
            build(save_teacher(self.tmp, model_cfg=other, seed=7))

    def test_prototype_needs_forward_features(self):
        tiny_2c = {"target": TINY_MODEL_CFG["target"], "params": {**TINY_MODEL_CFG["params"], "n_classes": 2}}
        ckpt = save_teacher(self.tmp, model_cfg=tiny_2c, seed=8)
        with self.assertRaisesRegex(ValueError, "forward_features"):
            DTCTRUSLitWrapper(model_cfg=tiny_2c, optimizer_cfg=OPTIMIZER_CFG, report_class_ids={1: "FG"},
                              teacher_checkpoint=str(ckpt))
        module = DTCTRUSLitWrapper(model_cfg=tiny_2c, optimizer_cfg=OPTIMIZER_CFG, report_class_ids={1: "FG"},
                                   teacher_checkpoint=str(ckpt), use_prototype=False, total_steps=10)
        module.log = module.log_dict = lambda *a, **k: None
        self.assertTrue(torch.isfinite(module.training_step(dtc_batch(), 0)))


class DTCTRUSLossTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="dtc-trus-"))
        cls.ckpt = save_teacher(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_stage2_ignores_ground_truth(self):
        '''Fiel ao original: trocar a GT dos rotulados não muda a perda.'''
        # Um módulo novo por chamada: o passo atualiza o BN, e o pseudo-rótulo (em eval) o lê.
        a, b = dtc_batch(), dtc_batch()
        b["segmentation"][:2] = 1 - b["segmentation"][:2]
        b["metadata"]["is_labeled"][:] = False
        torch.testing.assert_close(build(self.ckpt).training_step(a, 0).detach(),
                                   build(self.ckpt).training_step(b, 0).detach())

    def test_flip_pseudo_label_is_the_unflipped_eval_prediction(self):
        module = build(self.ckpt)
        module.train()
        images = torch.randn(3, 3, SIZE, SIZE)
        target = module.flip_pseudo_label(images)
        module.model.eval()
        with torch.no_grad():
            expected = torch.flip(torch.softmax(module.model(torch.flip(images, dims=[-1])), 1), dims=[-1])
        torch.testing.assert_close(target, expected)
        module.train()
        self.assertTrue(module.model.training)     # o flip devolve o aluno ao modo de treino

    def test_kd_is_zero_for_identical_logits_and_scales_with_temperature(self):
        module = build(self.ckpt)
        logits = torch.randn(2, 2, SIZE, SIZE)
        self.assertEqual(float(module.kd_loss(logits, logits)), 0.0)
        other = torch.randn(2, 2, SIZE, SIZE)
        T = module.kd_temperature
        expected = F.mse_loss(torch.softmax(logits / T, 1), torch.softmax(other / T, 1)) * T * T
        torch.testing.assert_close(module.kd_loss(logits, other), expected)

    def test_temporal_consistency_formula(self):
        probs = torch.softmax(torch.randn(2, 2, SIZE, SIZE), 1)
        ones = torch.ones(2, 1, SIZE, SIZE)
        self.assertAlmostEqual(float(DTCTRUSLitWrapper.temporal_consistency_loss(probs, probs, ones)), 0.0, places=6)
        # Manual: P_t = 1 no FG, P~ = 0.5 -> |Δ| = 0.5 nas duas classes; M_noc = 0.5 -> 1 - 0.5*0.5 = 0.75.
        current = torch.stack([torch.zeros(1, SIZE, SIZE), torch.ones(1, SIZE, SIZE)], 1)
        transported = torch.full_like(current, 0.5)
        loss = DTCTRUSLitWrapper.temporal_consistency_loss(current, transported, torch.full((1, 1, SIZE, SIZE), 0.5))
        self.assertAlmostEqual(float(loss), 0.75, places=6)

    def test_soft_segmentation_loss_matches_supervised_loss_on_one_hot_targets(self):
        module = build(self.ckpt)
        logits = torch.randn(2, 2, SIZE, SIZE)
        target = torch.randint(0, 2, (2, SIZE, SIZE))
        one_hot = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        torch.testing.assert_close(module.soft_segmentation_loss(logits, one_hot),
                                   module.supervised_loss(logits, target))

    def test_helpers(self):
        flow = torch.full((1, 2, SIZE, SIZE), 3.0)
        resized = resize_flow(flow, (SIZE // 2, SIZE // 2))
        torch.testing.assert_close(resized, torch.full((1, 2, SIZE // 2, SIZE // 2), 3.0 * (SIZE // 2 - 1) / (SIZE - 1)))
        probs = torch.arange(10.0).reshape(1, 1, 2, 5)
        self.assertEqual(adaptive_threshold(probs, 0.7), 7.0)       # índice int(10*0.7)=7 do ordenado
        features = torch.ones(2, 4, 3, 3)
        self.assertTrue(torch.equal(masked_avg_pool(features, torch.zeros(2, 1, 3, 3)), torch.zeros(2, 4)))

    def test_step_gradients_and_forward_counts(self):
        module = build(self.ckpt)
        module.train()
        loss = module.training_step(dtc_batch(flow_px=1.0), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.parameters()))
        self.assertTrue(all(p.grad is None for p in module.teacher.parameters()))
        self.assertEqual(module.forward_counts["student"], 4 + 4)        # 4 centrais + 4 vizinhos, um forward
        self.assertEqual(module.forward_counts["student_nograd"], 4)     # flip
        self.assertEqual(module.forward_counts["teacher"], 4)

    def test_temporal_terms_backpropagate_through_the_neighbor_too(self):
        '''Como no original, nenhuma das duas predições do par é destacada.'''
        module = build(self.ckpt, seg_weight=0.0, kd_weight=0.0, proto_weight=0.0)
        batch = dtc_batch(flow_px=1.0)
        batch["neighbors"].requires_grad_(True)
        module.training_step(batch, 0).backward()
        self.assertIsNotNone(batch["neighbors"].grad)
        self.assertGreater(float(batch["neighbors"].grad.abs().sum()), 0.0)

    def test_invalid_neighbors_do_not_contribute(self):
        a, b = dtc_batch(k=2), dtc_batch(k=2)
        a["neighbor_valid"][:, 1] = False
        b["neighbor_valid"][:, 1] = False
        b["neighbors"][:, 1] = 1e3
        torch.testing.assert_close(build(self.ckpt).training_step(a, 0).detach(),
                                   build(self.ckpt).training_step(b, 0).detach())

    def test_weak_augment_reaches_neighbors(self):
        module = build(self.ckpt, student_augment="weak")
        module.train()
        self.assertTrue(torch.isfinite(module.training_step(dtc_batch(flow_px=1.0), 0)))
        keys = module._augmented_keys(dtc_batch())
        self.assertTrue({"neighbors", "neighbor_flow", "neighbor_confidence"} <= set(keys))


class DTCTRUSConfigTests(unittest.TestCase):
    '''O YAML do DTC-TRUS segue o molde comum do E2: só `lit_module`, `name`, `neighbors_for` e o callback
    `val_student` (que não existe sem teacher EMA) diferem do `mt-temporal-prev`.'''

    def test_common_protocol(self):
        load = lambda name: OmegaConf.to_container(OmegaConf.load(str(PROJECT_ROOT / f"configs/experiment/e2/{name}.yaml")))
        dtc, prev = load("e2-dtc-trus"), load("e2-mt-temporal-prev")
        self.assertEqual(dtc["lit_module"]["target"], "src.models.ssl.dtc_trus.DTCTRUSLitWrapper")
        self.assertIsNone(dtc["lit_module"]["params"]["teacher_checkpoint"])
        self.assertEqual(dtc["data"]["params"]["train"]["params"]["neighbors_for"], "all")
        dtc["data"]["params"]["train"]["params"]["neighbors_for"] = "unlabeled"
        for key in ("model", "optimizer", "lr_scheduler", "data"):
            self.assertEqual(dtc[key], prev[key], msg=key)
        self.assertEqual(dtc["trainer"]["params"], prev["trainer"]["params"])
        self.assertEqual(dtc["trainer"]["checkpoint"], prev["trainer"]["checkpoint"])
        prev_callbacks = [c for c in prev["trainer"]["callbacks"] if c.get("params", {}).get("prefix") != "val_student"]
        self.assertEqual(dtc["trainer"]["callbacks"], prev_callbacks)


class DTCTRUSSweepTests(unittest.TestCase):
    '''`teacher_from`: a etapa 1 de cada célula `dtc-trus` é o best.ckpt da `supervised` pareada.'''

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dtc-sweep-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def sweep(self, name):
        from src.sweep import Sweep
        return Sweep(f"configs/sweep/{name}.yaml", log_dir=self.tmp)

    def write_state(self, sweep, cells):
        '''`cells`: {run_id: status}; cada `done` ganha um checkpoint (existente) próprio.'''
        state = {"cells": {}}
        for run_id, status in cells.items():
            checkpoint = self.tmp / f"{run_id}.ckpt"
            checkpoint.write_text("x")
            state["cells"][run_id] = {"status": status, "best_checkpoint_path": str(checkpoint)}
        sweep.sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep.state_path.write_text(json.dumps(state))

    def test_final_training_reads_the_paired_supervised_checkpoint(self):
        sweep = self.sweep("e2")
        cells = sweep.plan.cells()
        self.assertEqual([c["method"] for c in cells][-1], "dtc-trus")
        dtc = cells[-1]
        sup_run_id = "deeplabv3plus-supervised-global-s42"
        self.write_state(sweep, {sup_run_id: "done"})
        params = sweep.plan.overrides(dtc)["lit_module"]["params"]
        self.assertEqual(params["teacher_run_id"], sup_run_id)
        self.assertEqual(params["teacher_checkpoint"], str(self.tmp / f"{sup_run_id}.ckpt"))
        self.assertNotIn("lit_module", sweep.plan.overrides(cells[0]))   # métodos de uma etapa: nada muda

    def test_cv_fold_k_gets_the_teacher_of_fold_k(self):
        sweep = self.sweep("e2-cv")
        cells = sweep.plan.cells()
        self.assertEqual(len(cells), 35)
        self.write_state(sweep, {f"deeplabv3plus-supervised-global-f{k}-s42": "done" for k in range(5)})
        for cell in cells:
            if cell["method"] != "dtc-trus":
                continue
            params = sweep.plan.overrides(cell)["lit_module"]["params"]
            self.assertEqual(params["teacher_run_id"], f"deeplabv3plus-supervised-global-f{cell['fold']}-s42")
            self.assertTrue(params["teacher_checkpoint"].endswith(f"-f{cell['fold']}-s42.ckpt"))

    def test_missing_or_unfinished_teacher_is_an_error(self):
        sweep = self.sweep("e2-cv")
        dtc = [c for c in sweep.plan.cells() if c["method"] == "dtc-trus" and c["fold"] == 1][0]
        with self.assertRaisesRegex(RuntimeError, "supervised-global-f1"):
            sweep.plan.overrides(dtc)
        self.write_state(sweep, {"deeplabv3plus-supervised-global-f0-s42": "done",
                                 "deeplabv3plus-supervised-global-f1-s42": "running"})
        with self.assertRaisesRegex(RuntimeError, "status=running"):
            sweep.plan.overrides(dtc)

    def test_cost_row_adds_stage1(self):
        sweep = self.sweep("e2")
        dtc = sweep.plan.cells()[-1]
        resources = pd.DataFrame([{"run_id": "deeplabv3plus-supervised-global-s42",
                                   "fit_wall_s": 100.0, "train_s": 80.0, "steps": 8000}])
        costs = sweep.plan._stage1_costs(dtc, resources)
        self.assertEqual(costs, {"teacher_run_id": "deeplabv3plus-supervised-global-s42",
                                 "stage1_fit_wall_s": 100.0, "stage1_train_s": 80.0, "stage1_steps": 8000.0})
        self.assertEqual(sweep.plan._stage1_costs(sweep.plan.cells()[0], resources), {})

        experiment = mock.Mock(resources={"fit_wall_s": 50.0, "train_s": 40.0}, experiment_run_dir=self.tmp)
        sweep.plan.resources_csv.parent.mkdir(parents=True, exist_ok=True)
        resources.to_csv(sweep.plan.resources_csv, index=False)
        sweep.plan.finish_cell(dtc, experiment, pd.DataFrame())
        row = pd.read_csv(sweep.plan.resources_csv).set_index("run_id").loc[sweep.plan.run_id(dtc)]
        self.assertEqual((row.total_fit_wall_s, row.total_train_s), (150.0, 120.0))


if __name__ == "__main__":
    unittest.main()
