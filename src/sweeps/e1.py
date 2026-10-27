'''
E1 -- arquiteturas supervisionadas sob o protocolo do artigo.

Duas grades com a mesma base (`configs/experiment/e1/`):

  E1TuningSweep  tuning A.1: treina so no fold 0 do desenvolvimento, seleciona pela validacao fixa e
                 NAO toca em teste nenhum. Cada celula e um trial (lr, weight_decay) de `src/tuning.py`.
  E1CVSweep      Etapa B: CV 5-fold por paciente com os hiperparametros congelados; o fold retido e o
                 split 'test' do run, a validacao fixa segue escolhendo o checkpoint.

O que difere do X1 e nao pode virar YAML estatico: o split vem da tabela congelada
(`split_mode: protocol`), e a validacao e por PASSOS (`val_check_interval`) em vez de por epocas. A
epoca muda de tamanho de fold para fold (os folds tem 117 a 127 frames, o tuning treina com um fold so),
entao um `check_val_every_n_epoch` fixo validaria em passos diferentes por celula.
'''
from functools import cached_property
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.data.splits import resolve_project_path
from src.evaluation import evaluate_per_class
from src.sweeps.base import SweepPlan
from src.tuning import sample_random_trials


class _E1Sweep(SweepPlan):

    def experiment_config(self, cell: Dict[str, Any]) -> str:
        '''A arquitetura e o eixo que escolhe o YAML base, nao `method` (o metodo e sempre supervisionado).'''
        base = self.config.base_configs
        arch = cell["arch"]
        if arch not in base:
            raise KeyError(f"`base_configs` nao tem entrada para arch='{arch}'")
        return str(base[arch])

    def _protocol_overrides(self, cell: Dict[str, Any], lr: float, weight_decay: float,
                            train_folds: Optional[List[int]], cv_fold: Optional[int]) -> Dict[str, Any]:
        training = self.config.training
        return {
            "name": self.run_name(cell),
            "trainer": {"params": {
                "max_steps": int(training.max_steps),
                "val_check_interval": int(training.val_every_n_steps),
                "check_val_every_n_epoch": None,   # exigido pelo Lightning para validar por passos
            }},
            "lr_scheduler": {"params": {"T_max": int(training.max_steps)}},
            "optimizer": {"params": {"lr": float(lr), "weight_decay": float(weight_decay)}},
            "data": {"params": {
                "batch_size": int(training.batch_size),
                "shared_dataset_params": {
                    "split_mode": "protocol",
                    "split_file": str(self.config.data.split_file),
                    "train_folds": train_folds,
                    "cv_fold": cv_fold,
                },
            }},
        }

    def _class_ids(self) -> Dict[int, str]:
        return {int(k): v for k, v in
                OmegaConf.to_container(self.config.evaluation.class_ids, resolve=True).items()}


class E1TuningSweep(_E1Sweep):
    '''
    Celula = (trial, arch, seed). Todas as arquiteturas recebem os MESMOS pontos -- a mesma seed do sampler
    gera a mesma sequencia --, o que deixa o tuning pareado entre arquiteturas.

    `candidates: {arch: [trial, ...]}` (opcional) restringe a grade, para a confirmacao das melhores
    configuracoes com mais folds de treino (`data.train_folds: [1, 2, 3, 4]`).
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['arch']}-t{int(cell['trial']):02d}-s{cell['seed']}"

    def cells(self) -> List[Dict[str, Any]]:
        cells = super().cells()
        candidates = self.config.get("candidates")
        if candidates:
            allowed = {str(arch): {int(t) for t in trials} for arch, trials in
                       OmegaConf.to_container(candidates, resolve=True).items()}
            cells = [cell for cell in cells if int(cell["trial"]) in allowed.get(str(cell["arch"]), set())]
        return cells

    @cached_property
    def trial_points(self) -> List[Dict[str, float]]:
        '''`fixed_trials` primeiro (pontos escolhidos a mao, ex.: a referencia 1e-3/1e-4), depois os sorteados.'''
        space = self.config.search_space
        params = OmegaConf.to_container(space.params, resolve=True)
        fixed = space.get("fixed_trials")
        fixed = OmegaConf.to_container(fixed, resolve=True) if fixed else []
        for point in fixed:
            if set(point) != set(params):
                raise ValueError(f"fixed_trials {point} deve definir exatamente {sorted(params)}")
        sampled = sample_random_trials(params, int(space.n_trials), int(space.sampler_seed))
        return [{k: float(v) for k, v in point.items()} for point in fixed] + sampled

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        trial = int(cell["trial"])
        if not 0 <= trial < len(self.trial_points):
            raise IndexError(f"trial={trial} fora dos {len(self.trial_points)} pontos de `search_space`.")
        point = self.trial_points[trial]
        overrides = self._protocol_overrides(
            cell, point["lr"], point["weight_decay"],
            train_folds=[int(f) for f in self.config.data.train_folds], cv_fold=None,
        )
        # Sem split de teste: o DataModule descarta splits None e `Experiment.run` pula `trainer.test`.
        overrides["data"]["params"]["test"] = None
        return overrides

    def evaluate(self, experiment) -> pd.DataFrame:
        '''
        Por frame, na VALIDACAO FIXA, com os pesos do melhor checkpoint. Sem `trainer.test`, nada mais
        carrega esses pesos no modulo -- no fim do fit ele esta com os pesos do ultimo passo.
        '''
        module = experiment.lit_module
        if experiment.best_checkpoint_path:
            checkpoint = torch.load(experiment.best_checkpoint_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint["state_dict"])
        per_sample = evaluate_per_class(
            module, experiment.data_module.val_dataloader(),
            num_classes=int(self.config.evaluation.num_classes), class_ids=self._class_ids(),
            forward_fn=getattr(module, "_eval_forward", None),
        )
        optimizer = experiment.config.optimizer.params
        per_sample["eval_set"] = "val"
        per_sample["lr"] = float(optimizer.lr)
        per_sample["weight_decay"] = float(optimizer.weight_decay)
        return per_sample


class E1CVSweep(_E1Sweep):
    '''
    Celula = (fold, arch, seed). O fold e o eixo mais externo: se a grade for cortada, as arquiteturas
    concluidas continuam pareadas fold a fold.
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['arch']}-f{int(cell['fold'])}-s{cell['seed']}"

    @cached_property
    def frozen_hparams(self) -> Dict[str, Dict[str, Any]]:
        '''Lido sob demanda: a grade pode ser listada antes do congelamento, so nao pode ser treinada.'''
        path = resolve_project_path(str(self.config.hparams_file))
        if not path.exists():
            raise FileNotFoundError(
                f"{path} nao existe: os hiperparametros ainda nao foram congelados. Rode o tuning "
                "(configs/sweep/e1-tuning.yaml) e `src.tuning.write_frozen_hparams` antes do CV."
            )
        return yaml.safe_load(path.read_text())["hparams"]

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        arch = str(cell["arch"])
        if arch not in self.frozen_hparams:
            raise KeyError(f"{self.config.hparams_file} nao tem hiperparametros para arch='{arch}'.")
        hparams = self.frozen_hparams[arch]
        return self._protocol_overrides(cell, hparams["lr"], hparams["weight_decay"],
                                        train_folds=None, cv_fold=int(cell["fold"]))

    def evaluate(self, experiment) -> pd.DataFrame:
        '''O 'test' deste run e o fold retido do CV -- nunca o teste INCA.'''
        per_sample = super().evaluate(experiment)
        per_sample["eval_set"] = "cv"
        return per_sample
