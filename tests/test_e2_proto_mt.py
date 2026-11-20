'''
E2: MT temporal + protótipos (`PrototypeTemporalMeanTeacherLitWrapper`) e os YAMLs dos 3 métodos novos
(`mt-temporal-proto-teacher`, `mt-temporal-proto-student`, `mt-temporal-nomt`).

Roda com:  python -m unittest tests.test_e2_proto_mt -v
'''

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.ssl import PrototypeTemporalMeanTeacherLitWrapper, TemporalMeanTeacherLitWrapper
from tests.helpers import OPTIMIZER_CFG, TINY_FEATURE_MODEL_CFG, TINY_MODEL_CFG
from tests.test_e2_temporal import at_step, temporal_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIZE = 16


def build(cls=PrototypeTemporalMeanTeacherLitWrapper, model_cfg=TINY_FEATURE_MODEL_CFG, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("total_steps", 1000)
    module = cls(model_cfg=model_cfg, optimizer_cfg=OPTIMIZER_CFG, report_class_ids={1: "FG"}, **kwargs)
    module.logged = {}
    module.log = lambda name, value, **k: module.logged.__setitem__(name, value)
    module.log_dict = lambda values, **k: module.logged.update(values)
    return module


def batch(k=1):
    out = temporal_batch(k=k, size=SIZE)
    out["segmentation"] = (out["segmentation"] > 0).long()
    # Fluxo não nulo: o transporte tem de mexer em algo.
    out["neighbor_flow"][:, :, 0] = 1.0
    return out


def step(module, global_step=500):
    '''Um `training_step` num passo com o termo temporal ligado; devolve (loss, student.grads, teacher.grads).'''
    module.train()
    with at_step(module, global_step):
        loss = module.training_step(batch(), 0)
        loss.backward()
    student = {n: p.grad for n, p in module.model.named_parameters()}
    teacher = {n: p.grad for n, p in module.teacher.named_parameters()}
    return loss, student, teacher


class PrototypeMeanTeacherTests(unittest.TestCase):

    def test_proto_weight_follows_the_temporal_ramp(self):
        module = build(proto_weight=0.1, temporal_warmup_fraction=0.1)
        weights = []
        for global_step in (0, 99, 100, 500, 1000):
            with at_step(module, global_step):
                weights.append((module.current_proto_weight(), module.current_temporal_weight()))
        self.assertEqual(weights[0][0], 0.0)
        self.assertEqual(weights[1][0], 0.0)
        for proto, temporal in weights:
            self.assertAlmostEqual(proto, temporal * 0.1 / module.temporal_weight, places=9)
        self.assertAlmostEqual(weights[-1][0], 0.1, places=6)

    def test_no_proto_loss_in_warmup_and_the_step_matches_the_plain_temporal_mt(self):
        for source in ("teacher", "student"):
            module = build(proto_source=source, temporal_warmup_fraction=0.1)
            plain = build(TemporalMeanTeacherLitWrapper, model_cfg=TINY_FEATURE_MODEL_CFG, temporal_warmup_fraction=0.1)
            with self.subTest(source=source), at_step(module, 50), at_step(plain, 50):
                module.train(), plain.train()
                loss, expected = module.training_step(batch(), 0), plain.training_step(batch(), 0)
                self.assertAlmostEqual(loss.item(), expected.item(), places=6)
                self.assertNotIn("train/proto_loss", module.logged)

    def test_teacher_source_gradient_reaches_the_student_only(self):
        module = build(proto_source="teacher")
        loss, student, teacher = step(module)
        self.assertIn("train/proto_loss", module.logged)
        self.assertTrue(torch.isfinite(module.logged["train/proto_loss"]))
        self.assertTrue(all(g is None for g in teacher.values()))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in student.values()))

    def test_proto_term_adds_gradient_through_the_center_features(self):
        '''Com proto_weight=0 só sobra o MT temporal; ligar o termo tem de mudar o gradiente do student.'''
        with_proto, without = build(proto_weight=0.1), build(proto_weight=0.0)
        _, grads, _ = step(with_proto)
        _, base, _ = step(without)
        differing = [n for n in grads if not torch.allclose(grads[n], base[n])]
        self.assertTrue(differing)

    def test_student_source_stacks_the_neighbors_and_backpropagates_through_them(self):
        module = build(proto_source="student")
        module.train()
        with at_step(module, 500):
            module.training_step(batch(), 0)
        self.assertEqual(module.forward_counts["student"], 4 + 2)   # 4 centrais (2+2) + 2 vizinhos válidos
        self.assertEqual(module.forward_counts["teacher"], 2 + 2)
        teacher_source = build(proto_source="teacher")
        teacher_source.train()
        with at_step(teacher_source, 500):
            teacher_source.training_step(batch(), 0)
        self.assertEqual(teacher_source.forward_counts["student"], 4)

        # O vizinho do student leva gradiente: com `student` o gradiente difere do `teacher`.
        _, grads_student, teacher_grads = step(build(proto_source="student"))
        _, grads_teacher, _ = step(build(proto_source="teacher"))
        self.assertTrue(all(g is None for g in teacher_grads.values()))
        self.assertTrue(any(not torch.allclose(grads_student[n], grads_teacher[n]) for n in grads_student))

    def test_requires_forward_features(self):
        with self.assertRaisesRegex(ValueError, "forward_features"):
            build(model_cfg=TINY_MODEL_CFG)
        with self.assertRaisesRegex(ValueError, "proto_source"):
            build(proto_source="both")


class NewMethodConfigTests(unittest.TestCase):

    def load(self, name):
        return OmegaConf.to_container(OmegaConf.load(str(PROJECT_ROOT / f"configs/experiment/e2/{name}.yaml")))

    def test_proto_variants_differ_from_prev_only_in_the_prototype_block(self):
        prev = self.load("e2-mt-temporal-prev")
        for source in ("teacher", "student"):
            with self.subTest(source=source):
                cfg = self.load(f"e2-mt-temporal-proto-{source}")
                self.assertEqual(cfg["lit_module"]["target"],
                                 "src.models.ssl.temporal_mean_teacher.PrototypeTemporalMeanTeacherLitWrapper")
                params = cfg["lit_module"]["params"]
                self.assertEqual((params["proto_source"], params["proto_weight"], params["proto_percentile"]),
                                 (source, 0.1, 0.7))
                for key in ("proto_source", "proto_weight", "proto_percentile"):
                    del params[key]
                cfg["lit_module"]["target"] = prev["lit_module"]["target"]
                cfg["name"] = prev["name"]
                self.assertEqual(cfg, prev)


if __name__ == "__main__":
    unittest.main()
