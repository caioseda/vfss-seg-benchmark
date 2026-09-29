'''
X1 -- gate de margem para semi-supervisao: quanto os frames NAO anotados somam sobre a anotacao
que ja temos, em cada orcamento de rotulo.

Este modulo e a grade que vivia em `notebooks/hyphotesis_testing/experiment_1_does_ssl_helps.ipynb`
(constantes + `build_overrides` + `run_id`). Move-la para ca nao muda nada do experimento: os
`run_id` e as sobrescritas sao reproduzidos ao pe da letra, e `tests/test_sweeps.py` afirma isso
contra as linhas ja gravadas em `logs/x1/results_per_sample.csv`. A grade so passa a ser
reproduzivel sem o notebook, e a rodar fora do kernel dele.

A variante por frames-por-video (`X1FramesSweep`) mora aqui tambem porque difere do X1 em
exatamente um eixo -- o orcamento e um numero de frames anotados POR VIDEO, agrupado por
`video_id`, em vez de uma fracao dos pacientes.
'''
from functools import lru_cache
from typing import Any, Dict

from src.data.vfss_frame_dataset import (
    apply_label_regime,
    apply_label_regime_by_count,
    load_video_frame_dataframe,
    split_by_group,
)
from src.sweeps.base import SweepPlan


class _VFSSLabelBudgetSweep(SweepPlan):
    '''
    Comum as grades do VFSS: a frequencia de validacao e derivada do orcamento.

    Isto e a parte que NAO pode virar YAML estatico. O orcamento de compute e fixo em PASSOS
    (`max_steps`), mas o Lightning valida a cada N EPOCAS -- e com o `TwoStreamBatchSampler` uma
    epoca e `len(rotulados) // labeled_batch_size`, que encolhe conforme o orcamento de anotacao.
    Um `check_val_every_n_epoch` constante validaria 4x mais no orcamento de 25% do que no de 100%.
    Por isso o valor e calculado por celula, a partir da contagem real de rotulos revelados.
    '''

    def _dataframe(self):
        return self._load_dataframe(
            str(self.config.data.dataset_path),
            int(self.config.data.split_seed),
            str(self.config.data.split_group_column),
            tuple(self.config.data.split_ratios),
        )

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_dataframe(dataset_path: str, split_seed: int, group_column: str, ratios: tuple):
        '''Cacheado: indexar ~46k frames custa segundos, e a grade inteira usa o mesmo fold.'''
        dataframe = load_video_frame_dataframe(dataset_path)
        return split_by_group(dataframe, group_column, seed=split_seed, ratios=list(ratios))

    def _revealed_labels(self, regime_fn, **regime_kwargs) -> int:
        '''Quantos rotulos o regime revela no treino -- a contagem de que sai `steps_per_epoch`.'''
        regime = regime_fn(
            self._dataframe(),
            seed=int(self.config.data.label_seed),
            label_regime_splits=("train",),
            **regime_kwargs,
        )
        train = regime[regime.split == "train"]
        return int(train.is_labeled.sum())

    def _check_val_every_n_epoch(self, revealed: int) -> int:
        steps_per_epoch = max(1, revealed // int(self.config.training.labeled_batch_size))
        return max(1, round(int(self.config.training.val_every_n_steps) / steps_per_epoch))

    def _pool_params(self) -> Dict[str, Any]:
        '''
        O pool nao rotulado, identico em todas as celulas de um sweep -- e a variavel que NAO varia.

        `unlabeled_pool_size: null` e TETO ausente, nao interruptor: e assim que o X1B roda um pool
        DENSO (sem teto e com `frame_stride: 1`), porque uma janela temporal precisa que os vizinhos
        do frame ancora estejam no pool. O X1 usa teto de 8000 e stride 3, ja que ali a proximidade
        temporal nao e a variavel e frames consecutivos de VFSS sao quase identicos.
        '''
        data = self.config.data
        size = data.get("unlabeled_pool_size")
        params: Dict[str, Any] = {
            "unlabeled_pool_size": None if size is None else int(size),
            "unlabeled_pool_frame_stride": int(data.unlabeled_pool_frame_stride),
            "unlabeled_pool_seed": int(data.unlabeled_pool_seed),
            "unlabeled_pool_splits": list(data.unlabeled_pool_splits),
        }
        if data.get("use_unlabeled_pool") is not None:
            params["use_unlabeled_pool"] = bool(data.use_unlabeled_pool)
        return params

    def _shared_overrides(self, cell: Dict[str, Any], check_val_every_n_epoch: int,
                          regime_params: Dict[str, Any]) -> Dict[str, Any]:
        training = self.config.training
        data = self.config.data
        return {
            "name": self.run_name(cell),
            "trainer": {"params": {
                "max_steps": int(training.max_steps),
                "check_val_every_n_epoch": check_val_every_n_epoch,
            }},
            "lr_scheduler": {"params": {"T_max": int(training.max_steps)}},
            "data": {"params": {
                "batch_size": int(training.batch_size),
                "shared_dataset_params": {
                    **regime_params,
                    "label_seed": int(data.label_seed),
                    "split_seed": int(data.split_seed),
                    "split_group_column": str(data.split_group_column),
                    **self._pool_params(),
                    "expose_hidden_targets": True,   # diagnostico; nenhuma loss le
                },
                "train_batch_sampler": {"params": {
                    "batch_size": int(training.batch_size),
                    "labeled_batch_size": int(training.labeled_batch_size),
                    "seed": int(cell["seed"]),
                    "unlabeled_policy": str(training.unlabeled_policy),
                }},
            }},
        }


class X1Sweep(_VFSSLabelBudgetSweep):
    '''Orcamento = fracao dos PACIENTES cujos rotulos sao revelados.'''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['method']}-lf{int(round(cell['label_fraction'] * 100)):03d}-s{cell['seed']}"

    def run_name(self, cell: Dict[str, Any]) -> str:
        # `run_name_prefix` existe porque a variante binaria grava `x1-binario-...`: prefixos
        # distintos por variante sao o que impede duas grades de dividirem diretorio de run.
        prefix = self.config.get("run_name_prefix", "x1")
        return f"{prefix}-{self.run_id(cell)}"

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        fraction = float(cell["label_fraction"])
        group_column = str(self.config.data.label_group_column)
        revealed = self._revealed_labels(apply_label_regime, group_column=group_column,
                                         label_fraction=fraction)
        return self._shared_overrides(
            cell,
            self._check_val_every_n_epoch(revealed),
            {"label_fraction": fraction, "label_group_column": group_column},
        )


class X1FramesSweep(_VFSSLabelBudgetSweep):
    '''Orcamento = N frames anotados POR VIDEO. Agrupa por `video_id`, nao por paciente.'''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return f"{cell['method']}-fpv{int(cell['frames_per_video']):02d}-s{cell['seed']}"

    def run_name(self, cell: Dict[str, Any]) -> str:
        return f"{self.config.run_name_prefix}-{self.run_id(cell)}"

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        frames_per_video = int(cell["frames_per_video"])
        group_column = str(self.config.data.label_group_column)
        # `apply_label_regime_by_count` revela ate N frames por grupo, em vez de uma fracao de
        # grupos inteiros -- e por isso o parametro do dataset e `label_frames_per_group`.
        revealed = self._revealed_labels(apply_label_regime_by_count, group_column=group_column,
                                         frames_per_group=frames_per_video)
        return self._shared_overrides(
            cell,
            self._check_val_every_n_epoch(revealed),
            {"label_frames_per_group": frames_per_video, "label_group_column": group_column},
        )


class X1BSweep(_VFSSLabelBudgetSweep):
    '''
    X1B -- varia so a politica de amostragem do fluxo nao rotulado, com o orcamento fixo em 100%.
    Reusa `configs/experiment/x1/` de proposito: a unica diferenca em relacao ao X1 e a politica.
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        # O nome EXIBIDO da politica, com o "=" removido: "W=5" -> "W5". E o que os run_id ja
        # gravados em logs/x1b/results_per_sample.csv usam, e mudar isso faria o sweep retreinar
        # celulas que ja estao prontas.
        return f"{cell['method']}-{str(cell['policy']).replace('=', '')}-s{cell['seed']}"

    def run_name(self, cell: Dict[str, Any]) -> str:
        return f"x1b-{self.run_id(cell)}"

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        label_fraction = float(self.config.training.label_fraction)
        group_column = str(self.config.data.label_group_column)
        revealed = self._revealed_labels(apply_label_regime, group_column=group_column,
                                         label_fraction=label_fraction)
        overrides = self._shared_overrides(
            cell,
            self._check_val_every_n_epoch(revealed),
            {"label_fraction": label_fraction, "label_group_column": group_column},
        )
        # `policies` mapeia o nome exibido para (unlabeled_policy, temporal_window). `temporal_window`
        # e sempre escrito, inclusive como None, para que a config registrada seja identica a das
        # celulas que o notebook ja rodou.
        policy_cfg = self.config.policies[str(cell["policy"])]
        window = policy_cfg.get("temporal_window")
        sampler_params = overrides["data"]["params"]["train_batch_sampler"]["params"]
        sampler_params["unlabeled_policy"] = str(policy_cfg.unlabeled_policy)
        sampler_params["temporal_window"] = None if window is None else int(window)
        return overrides


class X1PoolAblationSweep(X1FramesSweep):
    '''
    X1 por frames-por-video com um eixo a mais, `pool`: a configuracao do pool nao rotulado
    (teto + stride) passa a variar por celula, lida de `pools` no YAML. Serve para medir se os
    resultados ja gravados com teto 8000 / stride 3 mudariam com o pool denso do X1B.
    '''

    def run_id(self, cell: Dict[str, Any]) -> str:
        return (f"{cell['method']}-fpv{int(cell['frames_per_video']):02d}"
                f"-pool{cell['pool']}-s{cell['seed']}")

    def overrides(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        overrides = super().overrides(cell)
        pool = self.config.pools[str(cell["pool"])]
        size = pool.get("unlabeled_pool_size")
        overrides["data"]["params"]["shared_dataset_params"].update({
            "use_unlabeled_pool": True,
            "unlabeled_pool_size": None if size is None else int(size),
            "unlabeled_pool_frame_stride": int(pool.unlabeled_pool_frame_stride),
        })
        return overrides
