'''
ACDC -- teste de aceitacao do port de SSL, e o gate por tras da grade X1.

Nada no pipeline do VFSS distingue "o port esta quebrado" de "o metodo nao ajuda no VFSS". No ACDC
os numeros tem gabarito publicado, e por isso esta grade roda antes de qualquer celula do X1 sobre
DiffRect ser interpretada.

Difere das grades do VFSS em tres coisas, todas deliberadas:

  - a avaliacao e VOLUMETRICA (`src/evaluation_volume.py`), em val E teste, porque Dice por fatia
    sobre uma pilha e outra quantidade da que a literatura de ACDC reporta;
  - ha um eixo `backbone`. Todo numero publicado de ACDC 2-D e medido na UNet do SSL4MIS (1.81M
    parametros, com dropout); a UNet deste repositorio tem 17.26M e nenhum. Rodar so a nossa
    deixaria qualquer diferenca ambigua entre "o port esta errado" e "a arquitetura e outra";
  - o scheduler e `PolynomialLR` (`total_iters`), nao o cosseno do X1, porque aqui o objetivo e
    comparabilidade com um numero publicado e nao entre metodos.
'''
from typing import Any, Dict

import pandas as pd
from omegaconf import OmegaConf

from src.data.acdc_dataset import ACDC_LABEL_RATIOS, ACDC_PATIENTS_TO_SLICES
from src.sweeps.base import SweepPlan


class ACDCSweep(SweepPlan):

    def run_id(self, cell: Dict[str, Any]) -> str:
        ratio = str(cell["ratio"]).replace("%", "pct")
        return f"{cell['method']}-{cell['backbone']}-{ratio}-s{cell['seed']}"

    def run_name(self, cell: Dict[str, Any]) -> str:
        return f"acdc-{self.run_id(cell)}"

    def _check_val_every_n_epoch(self, ratio: str) -> int:
        '''
        O TwoStreamBatchSampler define a epoca como uma passada pelos ROTULADOS, entao ela encolhe
        com o orcamento: em 1% sao 32//3 = 10 batches, ou seja 3.000 epocas ate os 30k passos. Um
        `check_val_every_n_epoch` constante validaria ordens de magnitude mais em 1% do que em 10%.
        '''
        labeled_slices = ACDC_PATIENTS_TO_SLICES[ACDC_LABEL_RATIOS[ratio]]
        steps_per_epoch = max(1, labeled_slices // int(self.config.training.labeled_batch_size))
        return max(1, round(int(self.config.training.val_every_n_steps) / steps_per_epoch))

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        training = self.config.training
        ratio = str(cell["ratio"])
        overrides: Dict[str, Any] = {
            "name": self.run_name(cell),
            "trainer": {"params": {
                "max_steps": int(training.max_steps),
                "check_val_every_n_epoch": self._check_val_every_n_epoch(ratio),
            }},
            "lr_scheduler": {"params": {"total_iters": int(training.max_steps)}},
            "data": {"params": {
                "batch_size": int(training.batch_size),
                "shared_dataset_params": {"root": str(self.config.data.root),
                                          "expose_hidden_targets": True},
                "train": {"params": {"labeled_patients": ACDC_LABEL_RATIOS[ratio]}},
                "train_batch_sampler": {"params": {
                    "batch_size": int(training.batch_size),
                    "labeled_batch_size": int(training.labeled_batch_size),
                    "seed": int(cell["seed"]),
                }},
            }},
        }
        # `ssl4mis` e o backbone que o YAML ja traz (o da referencia); `repo` e uma sobrescrita.
        backbone = self.config.backbones[str(cell["backbone"])]
        if backbone is not None:
            overrides["model"] = OmegaConf.to_container(backbone, resolve=True)
        return overrides

    def evaluate(self, experiment) -> pd.DataFrame:
        '''
        Metricas volumetricas por volume, em val e teste.

        `surface_metrics=True` so aqui: durante o treino o callback as mantem desligadas, porque sao
        uma transformada de distancia por classe por volume a cada validacao.
        '''
        from src.data.acdc_dataset import ACDCTestVolumes, ACDCValVolumes
        from src.evaluation_volume import evaluate_volumes

        evaluation = self.config.evaluation
        class_ids = {int(k): v for k, v in
                     OmegaConf.to_container(evaluation.class_ids, resolve=True).items()}
        forward_fn = getattr(experiment.lit_module, "_eval_forward", None)

        frames = []
        for split, dataset_cls in (("val", ACDCValVolumes), ("test", ACDCTestVolumes)):
            per_volume = evaluate_volumes(
                experiment.lit_module,
                dataset_cls(root=str(self.config.data.root), image_channels=1),
                num_classes=int(evaluation.num_classes), class_ids=class_ids,
                forward_fn=forward_fn, surface_metrics=True,
            )
            per_volume.insert(0, "split", split)
            frames.append(per_volume)
        return pd.concat(frames, ignore_index=True)
