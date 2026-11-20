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

Métodos de duas etapas (DTC-TRUS): `teacher_from: {dtc-trus: supervised}` no YAML do sweep diz que a etapa
1 do método é a célula `supervised` com as MESMAS demais coordenadas (seed, arch, política e, no CV, fold).
O best.ckpt dela -- registrado pelo `Sweep` em `_sweep/state.json` -- vira `teacher_checkpoint`. Assim a
diferença DTC - supervisionado é pareada pelo mesmo professor, e a etapa 1 não é re-treinada (nem re-sorteada
pelo não-determinismo). A ordem da grade (`method` mais interno, `supervised` antes) garante que a irmã já
rodou; se não rodou, `overrides` falha dizendo qual célula falta. O custo da etapa 1 é somado na CSV de custo.
'''
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.splits import resolve_project_path
from src.evaluation import evaluate_per_class
from src.sweeps.e1 import E1CVSweep
from src.utils import instantiate_from_config


class E2Sweep(E1CVSweep):
    '''Célula = (seed, arch, unlabeled_policy, method). Herda `frozen_hparams` e `_protocol_overrides` do E1.'''

    # O E2 compara métodos; a arquitetura é fixa por unidade (diferente do E1, de quem herda).
    compared_axes = ("method",)

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
        lit_params = dict(self.ssl_hparams(cell))
        teacher = self.teacher_for(cell)
        if teacher is not None:
            teacher_run_id, checkpoint = teacher
            lit_params.update({"teacher_checkpoint": checkpoint, "teacher_run_id": teacher_run_id})
        if lit_params:
            overrides["lit_module"] = {"params": lit_params}
        return overrides

    # ------------------------------------------------------------------ frozen SSL hyperparameters

    def _frozen_ssl_document(self) -> Dict[str, Any]:
        path = resolve_project_path(str(self.config.ssl_hparams_file))
        if not path.exists():
            raise FileNotFoundError(
                f"{path} não existe: os pesos dos métodos SSL ainda não foram congelados. Rode o tuning do E2 "
                "(configs/sweep/e2-ssl-tuning-search.yaml e e2-ssl-tuning-confirm.yaml, ver "
                "docs/e2-tuning-ssl.md) e scripts/freeze_e2_ssl_hparams.py antes."
            )
        return yaml.safe_load(path.read_text())

    def ssl_hparams(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Pesos congelados pelo tuning do E2 para o método da célula (`ssl_hparams_file`), ou {} quando o sweep
        não declara o arquivo -- o caso do `e2.yaml`/`e2-cv.yaml`, que seguem com os pesos dos YAMLs base.

        `ssl_hparams_from: {variante: {method: base, params: [...]}}` faz uma variante herdar os pesos do
        método base (só os listados em `params`, se houver); `ssl_hparams_untuned` lista os métodos sem peso
        não supervisionado (o supervisionado). Qualquer outro método sem entrada é recusado, em vez de rodar
        silenciosamente com o peso padrão.
        '''
        if not self.config.get("ssl_hparams_file"):
            return {}
        method, arch = str(cell["method"]), str(cell["arch"])
        untuned = [str(m) for m in (self.config.get("ssl_hparams_untuned") or [])]
        if method in untuned:
            return {}
        mappings = self.config.get("ssl_hparams_from")
        mappings = OmegaConf.to_container(mappings, resolve=True) if OmegaConf.is_config(mappings) else (mappings or {})
        mapping = mappings.get(method) or {}
        source = str(mapping.get("method", method))
        frozen = self._frozen_ssl_document()["hparams"]
        if arch not in frozen or source not in frozen[arch]:
            raise KeyError(
                f"{self.config.ssl_hparams_file} não tem pesos para arch='{arch}', method='{source}'"
                + (f" (herdados por '{method}')" if source != method else "")
                + ". Declare o método em `ssl_hparams_untuned` ou `ssl_hparams_from` se for o caso."
            )
        params = dict(frozen[arch][source]["params"])
        keys = mapping.get("params")
        if keys is not None:
            missing = set(keys) - set(params)
            if missing:
                raise KeyError(f"'{source}' não tem os pesos {sorted(missing)} pedidos por '{method}'.")
            params = {key: params[key] for key in keys}
        return params

    # ------------------------------------------------------------------ two-stage methods

    def _teacher_methods(self) -> Dict[str, str]:
        return {str(k): str(v) for k, v in (self.config.get("teacher_from") or {}).items()}

    def _sweep_state(self) -> Dict[str, Any]:
        path = self.sweep_dir / "state.json"
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {"cells": {}}

    def teacher_sibling(self, cell: Dict[str, Any], teacher_method: str) -> Dict[str, Any]:
        '''A célula que é a etapa 1 de `cell`: a mesma, com `method` trocado. O tuning (e2_tuning.py) troca.'''
        return {**cell, "method": teacher_method}

    def teacher_state(self) -> Dict[str, Any]:
        '''O `state.json` onde a etapa 1 está registrada: o do próprio sweep (ou shard).'''
        return self._sweep_state()

    def teacher_for(self, cell: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        '''
        `(run_id, best_checkpoint_path)` da célula que é a etapa 1 desta, ou None se o método tem uma etapa
        só. A irmã é a mesma célula com `method` trocado -- então, no CV, o fold k só recebe o professor do
        fold k (treinado sem o fold k).
        '''
        teacher_method = self._teacher_methods().get(str(cell["method"]))
        if teacher_method is None:
            return None
        run_id = self.run_id(self.teacher_sibling(cell, teacher_method))
        entry = self.teacher_state().get("cells", {}).get(run_id, {})
        checkpoint = entry.get("best_checkpoint_path")
        if entry.get("status") != "done" or not checkpoint or not Path(checkpoint).exists():
            raise RuntimeError(
                f"A etapa 1 de '{self.run_id(cell)}' é a célula '{run_id}', que ainda não está concluída "
                f"com um best checkpoint em {self.sweep_dir / 'state.json'} (status="
                f"{entry.get('status', 'ausente')}, checkpoint={checkpoint}). Rode-a antes -- ela vem antes "
                f"na grade se `{teacher_method}` estiver em `axes.method`."
            )
        return run_id, str(checkpoint)

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
        return self.tables_dir / str(self.config.get("resources_csv", "results_resources.csv"))

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
        row.update(self._stage1_costs(cell, existing))
        row["total_fit_wall_s"] = float(row.get("fit_wall_s", float("nan"))) + float(row.get("stage1_fit_wall_s", 0.0))
        row["total_train_s"] = float(row.get("train_s", float("nan"))) + float(row.get("stage1_train_s", 0.0))
        if len(existing) and "run_id" in existing.columns:
            existing = existing[existing.run_id != row["run_id"]]
        table = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        self.resources_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.resources_csv, index=False)

    STAGE1_COST_KEYS = ("fit_wall_s", "train_s", "steps")

    def _stage1_costs(self, cell: Dict[str, Any], resources: pd.DataFrame) -> Dict[str, Any]:
        '''Custo da etapa 1 (a célula professora), lido da própria CSV de custo, como `stage1_*`.'''
        teacher_method = self._teacher_methods().get(str(cell["method"]))
        if teacher_method is None:
            return {}
        teacher_run_id = self.run_id(self.teacher_sibling(cell, teacher_method))
        costs: Dict[str, Any] = {"teacher_run_id": teacher_run_id}
        rows = resources[resources.run_id == teacher_run_id] if "run_id" in resources.columns else resources.iloc[:0]
        for key in self.STAGE1_COST_KEYS:
            costs[f"stage1_{key}"] = float(rows.iloc[-1][key]) if len(rows) and key in rows.columns else float("nan")
        return costs


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
