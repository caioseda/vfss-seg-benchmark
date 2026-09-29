'''
Retomada fiel: um run interrompido e retomado segue a MESMA trajetoria que teria seguido sem a
interrupcao.

Por que isto merece um teste proprio. Retomar "funciona" de forma enganosa: o Lightning restaura
modelo, otimizador, scheduler e `global_step`, entao o treino continua e a curva de loss parece
inteiramente normal -- mas o que NAO e restaurado (o estado dos RNGs e a posicao do sampler) faz o
run divergir em silencio. Nenhum teste de fumaca pega isso; so a comparacao com o run ininterrupto.

Duas garantias, com escopos diferentes e deliberadamente separadas:

  test_schedule_*   Os hiperparametros voltam e progridem identicos -- LR, peso de consistencia e
                    decaimento EMA no passo N sao os mesmos com ou sem interrupcao. Valem sempre,
                    porque os tres sao funcao de `global_step`.
  test_trajectory_* A trajetoria inteira (incluindo os sorteios de augmentation e o pareamento
                    rotulado/nao-rotulado) coincide. Exige `num_workers=0`: com processos de
                    trabalho, a semente de cada worker deriva de um sorteio feito na criacao do
                    iterador, e o numero de sorteios anteriores a esse ponto difere atravessando a
                    fronteira da retomada.

Roda com:  python -m unittest tests.test_resume -v
'''

import tempfile
import unittest
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.callbacks import ResumeFidelity
from src.data.samplers import TwoStreamBatchSampler
from src.models.ssl import FixMatchLitWrapper, MeanTeacherLitWrapper, SupervisedLitWrapper
from tests.helpers import TINY_MODEL_CFG

SIZE = 16
N_LABELED = 16
N_UNLABELED = 32
BATCH_SIZE = 4
LABELED_BATCH_SIZE = 2
MAX_STEPS = 12
CUT = 6          # onde a interrupcao acontece


class ToyFrames(Dataset):
    '''
    Dataset com a forma que `VFSSWindowImageDataset` produz. As linhas nao rotuladas levam
    `segmentation` toda zero, como o dataset real faz para um frame cujo rotulo foi escondido.

    `__getitem__` sorteia do RNG global de proposito: e assim que uma augmentation se comporta, e e
    o que faz a retomada ser observavel neste teste.
    '''

    def __init__(self):
        generator = torch.Generator().manual_seed(1234)
        self.images = torch.randn(N_LABELED + N_UNLABELED, 3, SIZE, SIZE, generator=generator)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        labeled = index < N_LABELED
        target = torch.zeros(SIZE, SIZE, dtype=torch.long)
        if labeled:
            target[SIZE // 8:SIZE // 2, SIZE // 8:SIZE // 2] = 1
            target[9 * SIZE // 16:14 * SIZE // 16, 9 * SIZE // 16:14 * SIZE // 16] = 3
        jitter = torch.randn(3, SIZE, SIZE) * 0.01          # <- consome o RNG global
        return {
            "image": self.images[index] + jitter,
            "segmentation": target,
            # `is_labeled` vive dentro de `metadata`, como no dataset real -- e de la que
            # `SemiSupervisedLitWrapper.labeled_mask` o le.
            "metadata": {"index": index, "is_labeled": torch.tensor(labeled)},
        }


class ToyDataModule(pl.LightningDataModule):
    '''Expoe o TwoStreamBatchSampler e delega o `state_dict` a ele, como `DataModuleFromConfig`.'''

    def __init__(self, num_workers: int = 0):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = ToyFrames()
        # Gerador proprio do loader, como em `DataModuleFromConfig`: mantem o sorteio de
        # `_base_seed` da criacao do iterador fora do fluxo global de RNG.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(0)
        self.sampler = TwoStreamBatchSampler(
            labeled_indices=list(range(N_LABELED)),
            unlabeled_indices=list(range(N_LABELED, N_LABELED + N_UNLABELED)),
            batch_size=BATCH_SIZE, labeled_batch_size=LABELED_BATCH_SIZE,
            seed=42, unlabeled_policy="global",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_sampler=self.sampler, num_workers=self.num_workers,
                          generator=self.loader_generator)

    def state_dict(self):
        return {"train_batch_sampler": self.sampler.state_dict(),
                "loader_generator": self.loader_generator.get_state()}

    def load_state_dict(self, state_dict):
        generator_state = (state_dict or {}).get("loader_generator")
        if generator_state is not None:
            self.loader_generator.set_state(generator_state)
        sampler_state = (state_dict or {}).get("train_batch_sampler")
        if sampler_state:
            self.sampler.load_state_dict(sampler_state)


class TraceRecorder(pl.Callback):
    '''Um registro por passo do que a retomada tem de preservar.'''

    def __init__(self):
        self.rows = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        weight = getattr(pl_module, "current_consistency_weight", None)
        self.rows.append({
            "step": int(trainer.global_step),
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "consistency_weight": float(weight()) if callable(weight) else 0.0,
            "loss": round(float(loss.detach()), 8),
            "indices": [int(i) for i in batch["metadata"]["index"]],
        })


def build_module(wrapper_cls):
    torch.manual_seed(0)
    return wrapper_cls(
        model_cfg=TINY_MODEL_CFG,
        optimizer_cfg={"target": "torch.optim.AdamW", "params": {"lr": 1e-2}},
        lr_scheduler_cfg={"target": "torch.optim.lr_scheduler.CosineAnnealingLR",
                          "params": {"T_max": MAX_STEPS}, "interval": "step"},
        report_class_ids={1: "C2", 3: "C4"},
        surface_metric_stages=[],
    )


def train(wrapper_cls, run_dir: Path, max_steps: int, ckpt_path=None, num_workers: int = 0,
          with_fidelity: bool = True, with_sampler_state: bool = True):
    '''
    Um fit de verdade, com o mesmo maquinario de checkpoint que a producao usa.

    `with_fidelity` / `with_sampler_state` existem para a ablacao: desligar cada peca e a unica
    forma de mostrar que ela carrega peso, em vez de supor.
    '''
    pl.seed_everything(42, workers=True)
    recorder = TraceRecorder()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"), monitor=None, save_last=True, save_top_k=0,
        every_n_train_steps=1, save_on_train_epoch_end=False,
    )
    callbacks = [recorder, checkpoint]
    if with_fidelity:
        callbacks.insert(0, ResumeFidelity())
    data_module = ToyDataModule(num_workers=num_workers)
    if not with_sampler_state:
        data_module.load_state_dict = lambda *args, **kwargs: None
    trainer = pl.Trainer(
        max_steps=max_steps, max_epochs=-1, accelerator="cpu", devices=1,
        logger=False, enable_checkpointing=True, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=callbacks,
    )
    trainer.fit(build_module(wrapper_cls), data_module, ckpt_path=ckpt_path)
    return recorder.rows


class ResumeFidelityTests(unittest.TestCase):

    def _uninterrupted_and_resumed(self, wrapper_cls, num_workers: int = 0, **ablation):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = train(wrapper_cls, root / "ref", MAX_STEPS, num_workers=num_workers)
            # Interrompe em CUT, depois retoma do last.ckpt e vai ate o fim.
            train(wrapper_cls, root / "run", CUT, num_workers=num_workers)
            resumed = train(wrapper_cls, root / "run", MAX_STEPS,
                            ckpt_path=str(root / "run" / "checkpoints" / "last.ckpt"),
                            num_workers=num_workers, **ablation)
        return reference, resumed

    def _divergences(self, **ablation):
        reference, resumed = self._uninterrupted_and_resumed(FixMatchLitWrapper, **ablation)
        by_step = {row["step"]: row for row in reference}
        batches = sum(1 for row in resumed if row["indices"] != by_step[row["step"]]["indices"])
        losses = sum(1 for row in resumed
                     if abs(row["loss"] - by_step[row["step"]]["loss"]) > 1e-6)
        return batches, losses

    def assert_schedule_matches(self, wrapper_cls):
        reference, resumed = self._uninterrupted_and_resumed(wrapper_cls)
        self.assertTrue(resumed, "a retomada nao produziu nenhum passo")
        by_step = {row["step"]: row for row in reference}
        for row in resumed:
            expected = by_step.get(row["step"])
            self.assertIsNotNone(expected, f"passo {row['step']} ausente na referencia")
            for key in ("lr", "consistency_weight"):
                self.assertAlmostEqual(
                    row[key], expected[key], places=10,
                    msg=(f"{wrapper_cls.__name__}: `{key}` divergiu no passo {row['step']} -- "
                         f"retomado={row[key]!r} referencia={expected[key]!r}"),
                )

    def test_schedule_supervised(self):
        self.assert_schedule_matches(SupervisedLitWrapper)

    def test_schedule_fixmatch(self):
        self.assert_schedule_matches(FixMatchLitWrapper)

    def test_schedule_meanteacher(self):
        self.assert_schedule_matches(MeanTeacherLitWrapper)

    def test_trajectory_matches_uninterrupted_run(self):
        '''
        A trajetoria completa, e nao so o cronograma: mesmos indices servidos e mesma loss a cada
        passo. E o que so vale porque `ResumeFidelity` devolve o estado dos RNGs e o sampler devolve
        a posicao dentro da epoca.
        '''
        reference, resumed = self._uninterrupted_and_resumed(FixMatchLitWrapper)
        by_step = {row["step"]: row for row in reference}
        for row in resumed:
            expected = by_step[row["step"]]
            self.assertEqual(row["indices"], expected["indices"],
                             f"batch diferente no passo {row['step']}")
            self.assertAlmostEqual(row["loss"], expected["loss"], places=5,
                                   msg=f"loss divergiu no passo {row['step']}")

    def test_without_resume_fidelity_the_trajectory_diverges(self):
        '''
        Ablacao. Sem o callback, o cronograma continua exato (o teste acima cobre isso) mas as
        losses passam a divergir, porque as augmentations, o ruido do teacher e o cutout do FixMatch
        voltam a sortear de um RNG que reiniciou. Se este teste passasse, `ResumeFidelity` nao
        estaria fazendo nada.
        '''
        batches, losses = self._divergences(with_fidelity=False)
        self.assertEqual(batches, 0, "o sampler ainda deveria servir os mesmos batches")
        self.assertGreater(losses, 0,
                           "sem ResumeFidelity as losses deveriam divergir -- o callback nao morde")

    def test_without_sampler_state_the_batches_diverge(self):
        '''Ablacao: sem o estado do sampler a retomada reinicia a epoca e serve outros batches.'''
        batches, losses = self._divergences(with_sampler_state=False)
        self.assertGreater(batches, 0,
                           "sem estado do sampler os batches deveriam divergir")
        self.assertGreater(losses, 0)

    def test_sampler_position_is_what_makes_it_work(self):
        '''
        Controle: sem o estado do sampler a retomada reinicia a epoca. Se este teste passasse, os
        outros nao estariam medindo nada.
        '''
        def make():
            return TwoStreamBatchSampler(
                labeled_indices=list(range(N_LABELED)),
                unlabeled_indices=list(range(N_LABELED, N_LABELED + N_UNLABELED)),
                batch_size=BATCH_SIZE, labeled_batch_size=LABELED_BATCH_SIZE,
                seed=42, unlabeled_policy="global")

        reference = [list(b) for b in make()]
        partial, state = [], None
        sampler = make()
        for i, batch in enumerate(sampler):
            partial.append(list(batch))
            if i == 2:
                state = sampler.state_dict()
                break

        with_state = make(); with_state.load_state_dict(state)
        self.assertEqual(partial + [list(b) for b in with_state], reference,
                         "com estado, a retomada deveria reproduzir a sequencia ininterrupta")

        without_state = make()
        self.assertNotEqual(partial + [list(b) for b in without_state][:len(reference) - len(partial)],
                            reference,
                            "sem estado a sequencia deveria divergir -- se nao diverge, o teste nao morde")


if __name__ == "__main__":
    unittest.main()
