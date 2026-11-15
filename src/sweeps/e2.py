'''
E2 -- semi-supervisão (supervisionado, FixMatch, Mean Teacher, MT + temporal) sobre a arquitetura
vencedora do E1, em regime de TREINO FINAL.

O que difere do E1 e não pode virar YAML estático:

  - a arquitetura é um eixo da grade. O bloco `model` vem do YAML do E1 daquela arquitetura e
    SUBSTITUI o do YAML do método -- não é mesclado, porque `OmegaConf.merge` deixaria chaves de uma
    arquitetura (ex. `encoder_name`) no bloco de outra. Por isso `experiment_config` compõe um YAML por
    célula em `logs/e2/_sweep/configs/`;
  - lr/wd vêm de `configs/frozen/e1-hparams.yaml`, por arquitetura;
  - o split é o do protocolo com TODOS os folds em treino e `holdout_test=True`: o 'test' do run é o
    teste INCA, alcançável só por esta flag;
  - a avaliação final roda em dois conjuntos (teste INCA e UNM inteiro) e o custo do run
    (`ResourceMonitor`) vai para uma CSV própria, que a tabela comparativa cruza com a de acurácia.

`E2CVSweep` é a etapa anterior, no molde do `E1CVSweep`: CV 5-fold por paciente no desenvolvimento,
para estimar média ± desvio entre folds de cada método. O teste INCA e o UNM não são tocados.
'''
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.splits import resolve_project_path
from src.evaluation import evaluate_per_class
from src.sweeps.e1 import E1CVSweep
from src.utils import instantiate_from_config


class E2Sweep(E1CVSweep):
    '''Célula = (seed, arch, unlabeled_policy, method). Herda `frozen_hparams` e `_protocol_overrides` do E1.'''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['arch']}-{cell['method']}-{cell['unlabeled_policy']}-s{cell['seed']}"

    # ------------------------------------------------------------------ config

    def experiment_config(self, cell: Dict[str, Any]) -> str:
        '''YAML do método com o bloco `model` da arquitetura da célula, gravado no `_sweep/`.'''
        method, arch = str(cell["method"]), str(cell["arch"])
        if method not in self.config.base_configs:
            raise KeyError(f"`base_configs` nao tem entrada para method='{method}'")
        if arch not in self.config.arch_configs:
            raise KeyError(f"`arch_configs` nao tem entrada para arch='{arch}'")
        method_cfg = OmegaConf.load(str(resolve_project_path(str(self.config.base_configs[method]))))
        arch_cfg = OmegaConf.load(str(resolve_project_path(str(self.config.arch_configs[arch]))))
        method_cfg.model = arch_cfg.model

        path = self.sweep_dir / "configs" / f"{self.run_id(cell)}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = OmegaConf.to_yaml(method_cfg)
        # Reescreve só se mudou: o arquivo é a identidade do run (o fingerprint sai do conteúdo).
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        return str(path)

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        arch = str(cell["arch"])
        if arch not in self.frozen_hparams:
            raise KeyError(f"{self.config.hparams_file} nao tem hiperparametros para arch='{arch}'.")
        hparams = self.frozen_hparams[arch]
        training = self.config.training
        overrides = self._protocol_overrides(cell, hparams["lr"], hparams["weight_decay"],
                                             train_folds=None, cv_fold=None)
        params = overrides["data"]["params"]
        params["shared_dataset_params"]["holdout_test"] = True
        params["train_batch_sampler"] = {"params": {
            "labeled_batch_size": int(training.labeled_batch_size),
            "unlabeled_batch_size": int(training.unlabeled_batch_size),
            "unlabeled_policy": str(cell["unlabeled_policy"]),
            "seed": int(cell["seed"]),
        }}
        return overrides

    # ------------------------------------------------------------------ lifecycle

    def begin(self, fast_dev_run: bool | int = False) -> None:
        self._fast_dev_run = bool(fast_dev_run)

    # ------------------------------------------------------------------ evaluation

    def _unm_loader(self) -> DataLoader:
        dataset = instantiate_from_config(self.config.evaluation.unm)
        return DataLoader(dataset, batch_size=int(self.config.training.batch_size),
                          shuffle=False, num_workers=int(self.config.evaluation.get("num_workers", 8)),
                          pin_memory=True)

    def _timed_eval(self, module, loader, eval_set: str) -> pd.DataFrame:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        per_sample = evaluate_per_class(
            module, loader, num_classes=int(self.config.evaluation.num_classes),
            class_ids=self._class_ids(), forward_fn=getattr(module, "_eval_forward", None),
        )
        self._eval_costs[f"eval_{eval_set}_s"] = time.perf_counter() - start
        self._eval_costs[f"eval_{eval_set}_frames"] = int(per_sample["video_frame"].nunique()) if len(per_sample) else 0
        if torch.cuda.is_available():
            self._eval_costs[f"vram_eval_{eval_set}_peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        per_sample["eval_set"] = eval_set
        return per_sample

    def evaluate(self, experiment) -> pd.DataFrame:
        '''
        Melhor checkpoint (val/dice_patient) carregado EXPLICITAMENTE, depois teste INCA e UNM inteiro,
        pelo caminho de avaliação do método (`_eval_forward`: o teacher no MT).
        '''
        module = experiment.lit_module
        if experiment.best_checkpoint_path:
            checkpoint = torch.load(experiment.best_checkpoint_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        self._eval_costs: Dict[str, Any] = {}
        frames = [
            self._timed_eval(module, experiment.data_module.test_dataloader(), "inca_test"),
            self._timed_eval(module, self._unm_loader(), "unm"),
        ]
        per_sample = pd.concat(frames, ignore_index=True)
        optimizer = experiment.config.optimizer.params
        per_sample["lr"] = float(optimizer.lr)
        per_sample["weight_decay"] = float(optimizer.weight_decay)
        return per_sample

    # ------------------------------------------------------------------ resources

    @property
    def resources_csv(self) -> Path:
        return self.log_dir / self.family / str(self.config.get("resources_csv", "results_resources.csv"))

    def load_resources(self) -> pd.DataFrame:
        return pd.read_csv(self.resources_csv) if self.resources_csv.exists() else pd.DataFrame()

    def finish_cell(self, cell: Dict[str, Any], experiment, per_sample: pd.DataFrame) -> None:
        '''Uma linha de custo por célula (tempo, VRAM, tamanho do modelo, avaliação), chave `run_id`.'''
        if getattr(self, "_fast_dev_run", False):
            return
        row = {"run_id": self.run_id(cell), **cell,
               **(experiment.resources or {}), **getattr(self, "_eval_costs", {}),
               "output_dir": str(experiment.experiment_run_dir)}
        existing = self.load_resources()
        if len(existing) and "run_id" in existing.columns:
            existing = existing[existing.run_id != row["run_id"]]
        table = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        self.resources_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.resources_csv, index=False)


class E2CVSweep(E2Sweep):
    '''
    Célula = (seed, arch, unlabeled_policy, fold, method). Rodada k: treina nos folds != k (rótulos +
    frames nunca anotados SÓ desses vídeos), escolhe o checkpoint pela validação fixa e avalia no fold k,
    que é o split 'test' do run (`eval_set="cv"`). O teste INCA e o UNM não são carregados.

    O pool não rotulado não vaza: `assign_protocol_split` marca o fold k como 'test' ANTES de
    `build_unlabeled_frame_pool`, que só aceita vídeos de `unlabeled_pool_splits=[train]`; os vizinhos
    temporais vêm do mesmo vídeo do central, logo do mesmo paciente de treino.
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return (f"{cell['arch']}-{cell['method']}-{cell['unlabeled_policy']}"
                f"-f{int(cell['fold'])}-s{cell['seed']}")

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        overrides = super().overrides(cell)
        shared = overrides["data"]["params"]["shared_dataset_params"]
        shared["train_folds"] = None          # todos os folds exceto `cv_fold`
        shared["cv_fold"] = int(cell["fold"])
        # Os YAMLs do E2 trazem `holdout_test: true` (treino final); no CV ele é proibido junto de `cv_fold`.
        shared["holdout_test"] = False
        return overrides

    def evaluate(self, experiment) -> pd.DataFrame:
        '''Melhor checkpoint (val/dice_patient), avaliado só no fold retido -- nunca no teste INCA nem no UNM.'''
        module = experiment.lit_module
        if experiment.best_checkpoint_path:
            checkpoint = torch.load(experiment.best_checkpoint_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        self._eval_costs = {}
        per_sample = self._timed_eval(module, experiment.data_module.test_dataloader(), "cv")
        optimizer = experiment.config.optimizer.params
        per_sample["lr"] = float(optimizer.lr)
        per_sample["weight_decay"] = float(optimizer.weight_decay)
        return per_sample
