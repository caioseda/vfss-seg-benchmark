'''
`student_augment: weak` em todos os métodos do E2 (correção do colapso do MT):

  - a refatoração de `weak_augment_multi` não mudou o resultado para um mesmo `generator`;
  - `weak_augment_temporal`: a geometria comuta com o transporte por fluxo,
        warp(aug(vizinho), aug(fluxo)) ≈ aug(warp(vizinho, fluxo)),
    nas duas convenções, com e sem flip; e o M_noc fica zero fora do quadro;
  - FixMatch aumenta só os rotulados; o MT temporal aumenta central, vizinhos, fluxo e M_noc juntos.

    python -m unittest tests.test_e2_aug -v
'''

import math
import unittest
from unittest import mock

import torch
from torchvision.transforms import InterpolationMode

from src.data import augmentations as aug
from src.data.optical_flow import warp
from src.models.ssl import FixMatchLitWrapper, TemporalMeanTeacherLitWrapper
from tests.helpers import OPTIMIZER_CFG, TINY_MODEL_CFG, make_batch

SIZE = 64


def smooth_image(channels=1, size=SIZE):
    '''Soma de gaussianas largas: bilinear quase exato, então o erro medido é o da geometria.'''
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    image = torch.zeros(size, size)
    for cx, cy, s in ((20, 24, 7.0), (42, 36, 9.0), (30, 48, 6.0)):
        image += torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s * s))
    return (image / image.max() * 2 - 1)[None].repeat(channels, 1, 1)


def smooth_flow(size=SIZE):
    '''Translação + leve rotação + componente suave: um campo genérico, não só constante.'''
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    c = (size - 1) / 2
    dx = 2.0 - 0.05 * (ys - c) + 1.0 * torch.sin(xs / 12)
    dy = -1.5 + 0.05 * (xs - c) + 0.8 * torch.cos(ys / 10)
    return torch.stack([dx, dy])


def interior(size=SIZE, margin=12):
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[margin:-margin, margin:-margin] = True
    return mask


class TestWeakAugmentRefactor(unittest.TestCase):

    def test_same_output_as_the_inline_reference(self):
        '''Reimplementa o laço antigo (sorteios inline) e compara bit a bit.'''
        image = torch.rand(3, 3, 24, 24) * 2 - 1
        mask = (torch.rand(3, 24, 24) > 0.5).long()

        def reference(generator):
            device = image.device
            out_i, out_m = [], []
            for i in range(image.shape[0]):
                img_i, m_i = image[i], mask[i].unsqueeze(0).float()
                if aug._rand(generator, device) < aug.WEAK_FLIP_PROB:
                    img_i, m_i = aug.TF.hflip(img_i), aug.TF.hflip(m_i)
                angle = aug._uniform(-aug.WEAK_MAX_ROTATION_DEG, aug.WEAK_MAX_ROTATION_DEG, generator, device)
                mdx, mdy = aug.WEAK_MAX_TRANSLATE * 24, aug.WEAK_MAX_TRANSLATE * 24
                t = [int(round(aug._uniform(-mdx, mdx, generator, device))),
                     int(round(aug._uniform(-mdy, mdy, generator, device)))]
                scale = aug._uniform(*aug.WEAK_SCALE_RANGE, generator=generator, device=device)
                out_i.append(aug.TF.affine(img_i, angle=angle, translate=t, scale=scale, shear=[0.0, 0.0],
                                           interpolation=InterpolationMode.BILINEAR, fill=[-1.0]))
                out_m.append(aug.TF.affine(m_i, angle=angle, translate=t, scale=scale, shear=[0.0, 0.0],
                                           interpolation=InterpolationMode.NEAREST, fill=[0.0]).squeeze(0).long())
            return torch.stack(out_i), torch.stack(out_m)

        ref_img, ref_mask = reference(torch.Generator().manual_seed(7))
        new_img, (new_mask,) = aug.weak_augment_multi(image, [mask], generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(new_img, ref_img, rtol=0, atol=0)
        self.assertTrue(torch.equal(new_mask, ref_mask))


class TestFlowEquivariance(unittest.TestCase):

    def check(self, params, convention):
        neighbor = smooth_image()[None]                    # [1,1,H,W]
        flow = smooth_flow()[None]                          # [1,2,H,W]
        reference = aug._apply_weak_geometry(warp(neighbor, flow, convention)[0], params,
                                             InterpolationMode.BILINEAR, fill=-1.0)
        aug_neighbor = aug._apply_weak_geometry(neighbor[0], params, InterpolationMode.BILINEAR, fill=-1.0)
        aug_flow = aug._warp_flow(flow, params)
        transported = warp(aug_neighbor[None], aug_flow, convention)[0]

        region = interior()
        error = (transported - reference).abs()[0][region]
        self.assertLess(float(error.max()), 0.06, f"{params} {convention}: max {float(error.max()):.3f}")

        # Controle: sem girar os vetores (só reamostrar), o transporte erra -- o teste tem poder.
        naive = aug._apply_weak_geometry(flow[0], params, InterpolationMode.BILINEAR, fill=0.0)[None]
        wrong = warp(aug_neighbor[None], naive, convention)[0]
        return float(error.mean()), float((wrong - reference).abs()[0][region].mean())

    def test_transport_commutes_with_the_geometry(self):
        for convention in ("dtc_trus", "backward"):
            for params in ((False, 9.0, [2, -3], 1.08), (True, -7.0, [-1, 2], 0.92), (True, 0.0, [0, 0], 1.0)):
                with self.subTest(convention=convention, params=params):
                    good, naive = self.check(params, convention)
                    self.assertLess(good, 0.01)
                    self.assertGreater(naive, 3 * good)

    def test_vector_matrix(self):
        torch.testing.assert_close(aug.weak_vector_matrix((True, 0.0, [0, 0], 1.0)),
                                   torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64)))
        linear = aug.weak_vector_matrix((False, 10.0, [3, 3], 1.1))
        self.assertAlmostEqual(float(torch.linalg.det(linear)), 1.1 ** 2, places=6)
        self.assertAlmostEqual(math.degrees(abs(math.atan2(float(linear[1, 0]), float(linear[0, 0])))), 10.0,
                               places=4)


class TestWeakAugmentTemporal(unittest.TestCase):

    def batch(self, b=3, k=2, size=32):
        g = torch.Generator().manual_seed(0)
        return (torch.rand(b, 3, size, size, generator=g) * 2 - 1,
                (torch.rand(b, size, size, generator=g) > 0.5).long(),
                torch.rand(b, k, 3, size, size, generator=g) * 2 - 1,
                torch.randn(b, k, 2, size, size, generator=g),
                torch.ones(b, k, 1, size, size))

    def test_center_matches_weak_augment_multi(self):
        image, mask, neighbors, flow, conf = self.batch()
        ref_img, (ref_mask,) = aug.weak_augment_multi(image, [mask], generator=torch.Generator().manual_seed(4))
        out = aug.weak_augment_temporal(image, [mask], neighbors, flow, conf,
                                        generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(out[0], ref_img, rtol=0, atol=0)
        self.assertTrue(torch.equal(out[1][0], ref_mask))
        self.assertEqual(out[2].shape, neighbors.shape)
        self.assertEqual(out[3].shape, flow.shape)
        self.assertEqual(out[4].shape, conf.shape)

    def test_confidence_is_zero_outside_the_frame(self):
        image, mask, neighbors, flow, conf = self.batch(b=1)
        params = (False, 10.0, [3, 3], 0.9)
        with mock.patch.object(aug, "_draw_weak_params", return_value=params):
            *_, out_conf = aug.weak_augment_temporal(image, [mask], neighbors, flow, conf)
        self.assertEqual(float(out_conf[0, :, 0, 0, 0].max()), 0.0)        # canto vem de fora
        self.assertGreater(float(out_conf[0, :, 0, 16, 16].min()), 0.99)   # centro segue válido


def build(cls, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("total_steps", 1000)
    return cls(model_cfg=TINY_MODEL_CFG, optimizer_cfg=OPTIMIZER_CFG,
               report_class_ids={1: "C2", 3: "C4"}, **kwargs)


class TestMethodsWiring(unittest.TestCase):

    def test_fixmatch_augments_only_labeled_rows(self):
        module = build(FixMatchLitWrapper, student_augment="weak")
        batch = make_batch(n_labeled=2, n_unlabeled=3, size=16)
        torch.manual_seed(1)
        out = module.augment_batch(batch)
        labeled = batch["metadata"]["is_labeled"].bool()
        self.assertTrue(torch.equal(out["image"][~labeled], batch["image"][~labeled]))
        self.assertFalse(torch.equal(out["image"][labeled], batch["image"][labeled]))

    def _record_forward_rows(self, module):
        '''Envolve `module.forward` e devolve a lista de tamanhos de lote de cada chamada, em ordem.'''
        calls = []
        original = module.forward

        def spy(images):
            calls.append(images.shape[0])
            return original(images)
        module.forward = spy
        return calls

    def test_fixmatch_main_forward_sees_only_labeled_rows_with_augmentation(self):
        '''
        A regressão do E2 (2026-11-17): rotulados aumentados + não rotulados CRUS no mesmo forward em modo
        train fazem a rede usar o grupo cru como dreno do BN, e em eval toda imagem crua vira "tudo FG".
        Com `student_augment` ligado o forward principal é só dos rotulados (weight 0: nenhum outro forward).
        '''
        module = build(FixMatchLitWrapper, student_augment="weak", consistency_warmup_fraction=0.5)
        module.log = module.log_dict = lambda *a, **k: None
        self.assertEqual(module.main_forward_rows, "labeled")
        batch = make_batch(n_labeled=2, n_unlabeled=3, size=16)
        calls = self._record_forward_rows(module)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=0):
            loss = module.training_step(batch, 0)
        self.assertEqual(calls, [2])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.parameters()))

    def test_fixmatch_unlabeled_rows_only_in_weak_and_strong_forwards(self):
        '''Depois do warm-up: rotulados (2), weak dos não rotulados (3, sem grad), strong (3). Nada cru.'''
        module = build(FixMatchLitWrapper, student_augment="weak", consistency_warmup_fraction=0.0,
                       confidence_threshold=0.0)
        module.log = module.log_dict = lambda *a, **k: None
        batch = make_batch(n_labeled=2, n_unlabeled=3, size=16)
        calls = self._record_forward_rows(module)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=900):
            loss = module.training_step(batch, 0)
        self.assertEqual(calls, [2, 3, 3])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_fixmatch_without_augmentation_keeps_the_whole_batch_forward(self):
        '''Sem aug (X1) as metades têm a mesma distribuição: o comportamento histórico fica.'''
        module = build(FixMatchLitWrapper, student_augment="none", consistency_warmup_fraction=0.5)
        module.log = module.log_dict = lambda *a, **k: None
        self.assertEqual(module.main_forward_rows, "all")
        batch = make_batch(n_labeled=2, n_unlabeled=3, size=16)
        calls = self._record_forward_rows(module)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=0):
            module.training_step(batch, 0)
        self.assertEqual(calls, [5])

    def test_fixmatch_batch_without_labeled_rows_stays_differentiable(self):
        module = build(FixMatchLitWrapper, student_augment="weak", consistency_warmup_fraction=0.5)
        module.log = module.log_dict = lambda *a, **k: None
        batch = make_batch(n_labeled=0, n_unlabeled=3, size=16)
        calls = self._record_forward_rows(module)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=0):
            loss = module.training_step(batch, 0)
        self.assertEqual(calls, [])
        self.assertEqual(loss.item(), 0.0)
        loss.backward()

    def test_temporal_augments_neighbors_flow_and_confidence(self):
        module = build(TemporalMeanTeacherLitWrapper, student_augment="weak")
        batch = make_batch(n_labeled=2, n_unlabeled=2, size=16)
        g = torch.Generator().manual_seed(1)
        batch["neighbors"] = torch.randn(4, 2, 3, 16, 16, generator=g)
        batch["neighbor_flow"] = torch.full((4, 2, 2, 16, 16), 1.0)
        batch["neighbor_confidence"] = torch.ones(4, 2, 1, 16, 16)
        batch["neighbor_valid"] = torch.ones(4, 2, dtype=torch.bool)
        params = (True, 0.0, [0, 0], 1.0)   # só flip: vizinho espelhado, dx troca de sinal
        with mock.patch.object(aug, "_draw_weak_params", return_value=params):
            out = module.augment_batch(batch)
        torch.testing.assert_close(out["image"], batch["image"].flip(-1))
        torch.testing.assert_close(out["neighbors"], batch["neighbors"].flip(-1))
        torch.testing.assert_close(out["neighbor_flow"][:, :, 0], torch.full((4, 2, 16, 16), -1.0))
        torch.testing.assert_close(out["neighbor_flow"][:, :, 1], torch.full((4, 2, 16, 16), 1.0))
        self.assertTrue(torch.equal(out["neighbor_valid"], batch["neighbor_valid"]))

    def test_temporal_training_step_runs_with_augmentation(self):
        module = build(TemporalMeanTeacherLitWrapper, student_augment="weak", temporal_warmup_fraction=0.0,
                       consistency_warmup_fraction=0.0)
        module.log = module.log_dict = lambda *a, **k: None
        batch = make_batch(n_labeled=2, n_unlabeled=2, size=16)
        batch["neighbors"] = torch.randn(4, 2, 3, 16, 16)
        batch["neighbor_flow"] = torch.randn(4, 2, 2, 16, 16)
        batch["neighbor_confidence"] = torch.ones(4, 2, 1, 16, 16)
        batch["neighbor_valid"] = torch.ones(4, 2, dtype=torch.bool)
        with mock.patch.object(type(module), "global_step", new_callable=mock.PropertyMock, return_value=900):
            loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
