'''
Testes da instrumentação do diagnóstico do colapso do MT no E2:

  - `student_augment` (src/models/ssl/base.py): `none` não toca o lote; `weak` mantém imagem e máscara
    alinhadas e entrega ao teacher a MESMA geometria do student;
  - `bn_buffer_mode` e `recalibrate_batchnorm` (src/models/ssl/mean_teacher.py);
  - `MTEvalProbe` (src/callbacks.py): observacional, reproduz a métrica do protocolo e é no-op sem teacher.

    python -m unittest tests.test_e2_diag -v
'''

import types
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

from src.callbacks import MTEvalProbe, PatientValidation
from src.models.ssl import FixMatchLitWrapper, MeanTeacherLitWrapper, SupervisedLitWrapper
from src.models.ssl.diffrect import DiffRectLitWrapper
from src.models.ssl import base as ssl_base
from src.models.ssl import TemporalMeanTeacherLitWrapper
from src.models.ssl.mean_teacher import recalibrate_batchnorm
from tests.helpers import OPTIMIZER_CFG, TINY_BN_MODEL_CFG, ToyPatientFrames


def build(wrapper_cls, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("total_steps", 30_000)
    return wrapper_cls(model_cfg=TINY_BN_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
                       report_class_ids={1: "FG"}, **kwargs)


def at_step(module, step: int):
    return mock.patch.object(type(module), "global_step",
                             new_callable=mock.PropertyMock, return_value=step)


def silence(module):
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None


def binary_batch(n_labeled=2, n_unlabeled=2, size=16, seed=0):
    '''Imagem cujo canal 0 > 0 É a máscara: dá para checar o alinhamento depois de uma geometria.'''
    generator = torch.Generator().manual_seed(seed)
    n = n_labeled + n_unlabeled
    images = torch.rand(n, 3, size, size, generator=generator) * 2 - 1
    masks = (images[:, 0] > 0).long()
    is_labeled = torch.zeros(n, dtype=torch.bool)
    is_labeled[:n_labeled] = True
    masks[~is_labeled] = 0
    return {"image": images, "segmentation": masks, "metadata": {"is_labeled": is_labeled}}


class TestStudentAugment(unittest.TestCase):

    def test_none_is_identity(self):
        module = build(SupervisedLitWrapper)
        batch = binary_batch()
        self.assertIs(module.augment_batch(batch), batch)

    def test_weak_keeps_image_and_mask_aligned(self):
        module = build(SupervisedLitWrapper, student_augment="weak")
        # Imagem com blocos grandes, para que a interpolação bilinear não borre a borda inteira.
        batch = binary_batch(n_labeled=4, n_unlabeled=0, size=32)
        blocks = torch.zeros(4, 32, 32)
        blocks[:, 4:20, 6:22] = 1.0
        batch["image"] = (blocks * 2 - 1)[:, None].repeat(1, 3, 1, 1)
        batch["segmentation"] = blocks.long()
        torch.manual_seed(3)
        out = module.augment_batch(batch)
        self.assertIsNot(out, batch)
        self.assertFalse(torch.equal(out["image"], batch["image"]), "weak não alterou a imagem")
        agreement = ((out["image"][:, 0] > 0).long() == out["segmentation"]).float().mean()
        self.assertGreater(float(agreement), 0.97, "máscara desalinhada da imagem após a geometria")
        self.assertEqual(out["segmentation"].dtype, batch["segmentation"].dtype)
        self.assertIs(out["metadata"], batch["metadata"])

    def test_teacher_sees_the_augmented_unlabeled_rows(self):
        module = build(MeanTeacherLitWrapper, student_augment="weak", noise_std=0.0, noise_clamp=0.0,
                       consistency_warmup_steps=0)
        silence(module)
        batch = binary_batch()
        flipped = {"image": batch["image"].flip(-1), "seg": batch["segmentation"].flip(-1)}

        def fake_aug(image, masks, generator=None):
            return image.flip(-1), [m.flip(-1) if m is not None else None for m in masks]

        seen = {}
        real_forward = module.teacher.forward

        def spy(x):
            seen["x"] = x.clone()
            return real_forward(x)

        with mock.patch.object(ssl_base, "weak_augment_multi", side_effect=fake_aug), \
                mock.patch.object(module.teacher, "forward", side_effect=spy), at_step(module, 10_000):
            module.training_step(batch, 0)
        unlabeled = ~batch["metadata"]["is_labeled"]
        self.assertTrue(torch.allclose(seen["x"], flipped["image"][unlabeled]),
                        "o teacher não recebeu a mesma geometria do student")

    def test_unsupported_methods_refuse_the_flag(self):
        # DiffRect tem passo de treino e augmentation próprios; FixMatch e o MT temporal suportam
        # (tests/test_e2_aug.py).
        with self.assertRaises(ValueError):
            build(DiffRectLitWrapper, student_augment="weak")
        for cls in (FixMatchLitWrapper, TemporalMeanTeacherLitWrapper):
            self.assertEqual(build(cls, student_augment="weak").student_augment, "weak")
        with self.assertRaises(ValueError):
            build(SupervisedLitWrapper, student_augment="strong")


class TestBNBufferMode(unittest.TestCase):

    def _diverge(self, module):
        with torch.no_grad():
            for m in module.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.fill_(1.0)
                    m.running_var.fill_(3.0)
            for m in module.teacher.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.fill_(0.0)
                    m.running_var.fill_(1.0)

    def _teacher_bn(self, module):
        return next(m for m in module.teacher.modules() if isinstance(m, torch.nn.BatchNorm2d))

    def test_copy_is_the_reference_behaviour(self):
        module = build(MeanTeacherLitWrapper)
        self._diverge(module)
        with at_step(module, 10_000):
            module.update_teacher()
        bn = self._teacher_bn(module)
        self.assertTrue(torch.allclose(bn.running_mean, torch.ones_like(bn.running_mean)))

    def test_ema_averages_float_buffers(self):
        module = build(MeanTeacherLitWrapper, bn_buffer_mode="ema", ema_decay=0.99)
        self._diverge(module)
        student_bn = next(m for m in module.model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        student_bn.num_batches_tracked.fill_(7)
        with at_step(module, 10_000):
            module.update_teacher()
        bn = self._teacher_bn(module)
        self.assertTrue(torch.allclose(bn.running_mean, torch.full_like(bn.running_mean, 0.01)))
        self.assertTrue(torch.allclose(bn.running_var, torch.full_like(bn.running_var, 0.99 + 0.03)))
        self.assertEqual(int(bn.num_batches_tracked), 7, "buffer inteiro deve ser copiado")

    def test_ema_copies_at_step_zero(self):
        module = build(MeanTeacherLitWrapper, bn_buffer_mode="ema")
        self._diverge(module)
        with at_step(module, 0):
            module.update_teacher()
        bn = self._teacher_bn(module)
        self.assertTrue(torch.allclose(bn.running_mean, torch.ones_like(bn.running_mean)))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build(MeanTeacherLitWrapper, bn_buffer_mode="average")

    def test_recalibration_touches_only_buffers(self):
        module = build(MeanTeacherLitWrapper)
        net = module.teacher
        net.eval()
        params_before = [p.clone() for p in net.parameters()]
        bn = self._teacher_bn(module)
        momentum = bn.momentum
        images = [torch.randn(4, 3, 8, 8) * 5 + 2 for _ in range(3)]
        used = recalibrate_batchnorm(net, images)
        self.assertEqual(used, 3)
        for before, after in zip(params_before, net.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertFalse(net.training, "modo eval não foi restaurado")
        self.assertEqual(bn.momentum, momentum)
        # Média cumulativa (momentum=None) == média exata das médias por lote da 1a camada.
        with torch.no_grad():
            feats = [net.net[0](x).mean(dim=(0, 2, 3)) for x in images]
        self.assertTrue(torch.allclose(bn.running_mean, torch.stack(feats).mean(0), atol=1e-5))

    def test_recalibrate_mode_buffers_batches_and_runs_before_eval(self):
        module = build(MeanTeacherLitWrapper, bn_buffer_mode="recalibrate", bn_recalibration_batches=2)
        for i in range(3):
            with at_step(module, i):
                module.on_train_batch_end(None, binary_batch(seed=i), i)
        self.assertEqual(len(module.recent_train_images), 2)
        with mock.patch.object(module, "recalibrate_teacher_bn", wraps=module.recalibrate_teacher_bn) as spy:
            module.on_validation_epoch_start()
        spy.assert_called_once()
        self.assertFalse(module.teacher.training)


class TestMTEvalProbe(unittest.TestCase):

    def _trainer(self):
        loader = DataLoader(ToyPatientFrames(n_patients=3, frames_per_patient=4, size=16), batch_size=4)
        return types.SimpleNamespace(
            sanity_checking=False,
            strategy=types.SimpleNamespace(root_device=torch.device("cpu")),
            datamodule=types.SimpleNamespace(val_dataloader=lambda: loader),
        )

    def _feed(self, probe, module, n=3):
        for i in range(n):
            probe.on_train_batch_end(None, module, None, binary_batch(seed=i), i)

    def test_noop_without_teacher(self):
        module = build(SupervisedLitWrapper)
        logged = []
        module.log_dict = lambda d, **k: logged.append(d)
        probe = MTEvalProbe(num_classes=2)
        self._feed(probe, module)
        probe.on_validation_epoch_end(self._trainer(), module)
        self.assertEqual(logged, [])
        self.assertEqual(len(probe.recent), 0)

    def test_logs_keys_matches_protocol_and_is_observational(self):
        module = build(MeanTeacherLitWrapper)
        with torch.no_grad():   # teacher != student, para que as duas leituras difiram
            for p in module.teacher.parameters():
                p.add_(0.3 * torch.randn_like(p))
        module.train()
        state_before = {k: v.clone() for k, v in module.state_dict().items()}

        logged = {}
        module.log_dict = lambda d, **k: logged.update(d)
        probe = MTEvalProbe(num_classes=2, recal_batches=2, train_eval_batches=2)
        self._feed(probe, module, n=3)
        self.assertEqual(len(probe.recent), 2)
        self.assertEqual(len(probe.fixed_train), 2)
        trainer = self._trainer()
        probe.on_validation_epoch_end(trainer, module)

        for key in ("dice_teacher_eval", "dice_student_eval", "dice_teacher_bnrecal",
                    "dice_teacher_trainbn", "fg_frac_gt", "fg_frac_teacher", "fg_frac_student",
                    "fg_frac_unl_teacher", "param_dist", "bn_var_max_teacher", "bn_nonfinite_student",
                    "train_dice_eval_student", "train_dice_eval_teacher"):
            self.assertIn(f"probe/{key}", logged)
        self.assertGreater(logged["probe/param_dist"], 0.0)

        # Mesmo número que o PatientValidation (o critério de checkpoint) mede com o teacher.
        reference = {}
        module.log_dict = lambda d, **k: reference.update(d)
        PatientValidation(num_classes=2).on_validation_epoch_end(trainer, module)
        self.assertAlmostEqual(logged["probe/dice_teacher_eval"], reference["val/dice_patient"], places=6)

        for k, v in module.state_dict().items():
            self.assertTrue(torch.equal(v, state_before[k]), f"a sonda alterou {k}")
        self.assertTrue(module.training, "modo train não foi restaurado")


if __name__ == "__main__":
    unittest.main(verbosity=2)
