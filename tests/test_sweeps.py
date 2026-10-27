'''
Os sweeps reproduzem, ao pe da letra, as grades que os notebooks rodaram.

Por que isto merece um teste. Mover a grade de `build_overrides` (constantes de notebook) para
`src/sweeps/` e `configs/sweep/` so e seguro se a configuracao efetiva nao mudar -- caso contrario
os resultados novos deixam de ser comparaveis com os que ja estao na CSV, silenciosamente, porque um
`label_fraction` ou um `check_val_every_n_epoch` diferente produz um run perfeitamente plausivel.

A referencia nao e uma copia do codigo do notebook (que poderia derivar junto): e o `hparams.yaml`
que cada run JA CONCLUIDO gravou em disco. O teste afirma que a config que o sweep geraria hoje para
aquela celula e a mesma que produziu aquela linha da CSV.

Por isso este teste depende de `logs/` -- e pulado quando os runs nao estao presentes (num clone
limpo, por exemplo), em vez de falhar.

Roda com:  python -m unittest tests.test_sweeps -v
'''

import csv
import unittest
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from src.sweep import Sweep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = PROJECT_ROOT / "configs" / "sweep"
# Descoberto do diretorio, e nao listado a mao: uma grade nova entra na cobertura sem editar este
# arquivo -- que e o que o `acdc` e o `x1-binario` precisavam para deixar de ficar de fora.
SWEEPS = sorted(path.stem for path in SWEEP_DIR.glob("*.yaml"))

# As chaves que o sweep e responsavel por definir. Nao se compara a config inteira de proposito:
# o YAML base pode ganhar comentarios ou campos novos sem que a grade mude.
PINNED_KEYS = [
    "name",
    "trainer/params/max_steps",
    "trainer/params/check_val_every_n_epoch",
    "lr_scheduler/params/T_max",
    "data/params/batch_size",
    "data/params/shared_dataset_params/label_fraction",
    "data/params/shared_dataset_params/label_frames_per_group",
    "data/params/shared_dataset_params/label_seed",
    "data/params/shared_dataset_params/label_group_column",
    "data/params/shared_dataset_params/split_seed",
    "data/params/shared_dataset_params/split_group_column",
    "data/params/shared_dataset_params/split_mode",
    "data/params/shared_dataset_params/split_file",
    "data/params/shared_dataset_params/train_folds",
    "data/params/shared_dataset_params/cv_fold",
    "optimizer/params/lr",
    "optimizer/params/weight_decay",
    "trainer/params/val_check_interval",
    "data/params/shared_dataset_params/use_unlabeled_pool",
    "data/params/shared_dataset_params/unlabeled_pool_size",
    "data/params/shared_dataset_params/unlabeled_pool_frame_stride",
    "data/params/shared_dataset_params/unlabeled_pool_seed",
    "data/params/train_batch_sampler/params/batch_size",
    "data/params/train_batch_sampler/params/labeled_batch_size",
    "data/params/train_batch_sampler/params/seed",
    "data/params/train_batch_sampler/params/unlabeled_policy",
    "data/params/train_batch_sampler/params/temporal_window",
]


# `name` e a etiqueta do run (e o nome do diretorio), nao um hiperparametro -- e para esta grade ele
# foi mudado DE PROPOSITO. Os dois sweeps de frames-por-video usavam o mesmo prefixo `x1-frames-`, e
# por isso gravaram runs de 4 e de 8 classes na MESMA pasta, distinguiveis so pelo timestamp (ver o
# cabecalho de `scripts/migrate_logs.py`). Renomear o prefixo e a correcao. As celulas ja concluidas
# nao sao afetadas: elas sao puladas pelo `run_id`, que nao mudou.
# Um unico run, gravado antes de o template de `name` do ACDC ganhar o segmento do backbone: o
# run_id na CSV e `supervised-repo-10pct-s1337` mas o diretorio e `acdc-supervised-10pct-s1337`. Os
# outros 15 runs do ACDC casam exatamente. Nao e uma divergencia cientifica -- o `backbone: repo` da
# celula esta correto no hparams -- e a migracao ja preservou o vinculo via a coluna `output_dir`.
# Isento por (grade, run_id), e nao pela grade inteira, para que qualquer OUTRA deriva ainda falhe.
KNOWN_NAME_DRIFT = {
    ("acdc", "supervised-repo-10pct-s1337"):
        "gravado antes de `name` do ACDC incluir o backbone; o diretorio nao tem o segmento -repo",
}

NAME_EXEMPT = {
    "x1-frames-per-video-multiclass-all":
        "prefixo renomeado para x1-frames-mc-all, desfazendo a colisao com x1-frames-per-video",
}


def flatten(config, prefix: str = "") -> dict:
    '''Mesmo achatamento que `Experiment._flatten_config`, que e o que gerou os hparams.yaml.'''
    flat = {}
    if isinstance(config, dict):
        for key, value in config.items():
            flat.update(flatten(value, f"{prefix}{key}/"))
    elif isinstance(config, (list, tuple)):
        flat[prefix.rstrip("/")] = str(list(config))
    else:
        flat[prefix.rstrip("/")] = config
    return flat


class SweepReproducesNotebookGrid(unittest.TestCase):

    def _run_dirs_by_run_id(self, sweep: Sweep) -> dict:
        if not sweep.results_csv.exists():
            return {}
        with sweep.results_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_run_id = {}
        for row in rows:
            by_run_id.setdefault(row["run_id"], row.get("output_dir"))
        return by_run_id

    def _check(self, sweep_name: str):
        sweep = Sweep(PROJECT_ROOT / "configs" / "sweep" / f"{sweep_name}.yaml")
        run_dirs = self._run_dirs_by_run_id(sweep)
        if not run_dirs:
            return 0   # grade ainda nao rodada: nada gravado para comparar

        checked = 0
        for cell in sweep.plan.cells():
            run_id = sweep.plan.run_id(cell)
            run_dir = run_dirs.get(run_id)
            if not run_dir:
                continue
            hparams_path = Path(run_dir) / "csv" / "hparams.yaml"
            if not hparams_path.exists():
                continue

            recorded = yaml.safe_load(hparams_path.read_text()) or {}
            generated = flatten(OmegaConf.to_container(OmegaConf.merge(
                OmegaConf.load(sweep.plan.experiment_config(cell)),
                OmegaConf.create(sweep.plan.overrides(cell)),
            ), resolve=True))

            for key in PINNED_KEYS:
                if key not in recorded and key not in generated:
                    continue
                if key == "name" and sweep_name in NAME_EXEMPT:
                    continue
                if key == "name" and (sweep_name, run_id) in KNOWN_NAME_DRIFT:
                    continue
                self.assertEqual(
                    str(recorded.get(key, "<ausente>")), str(generated.get(key, "<ausente>")),
                    msg=(f"{sweep_name} / {run_id}: `{key}` divergiu do run ja gravado.\n"
                         f"  registrado em {hparams_path}\n"
                         f"  O sweep produziria um run NAO comparavel com a linha que ja esta na CSV."),
                )
            checked += 1

        return checked

    def test_every_sweep_reproduces_its_recorded_runs(self):
        '''
        Uma subasserção por grade, para que a falha diga QUAL grade divergiu. Grades ainda nao
        rodadas contam como 0 celulas conferidas, em vez de virarem um sucesso silencioso -- o
        resumo impresso deixa isso visivel.
        '''
        self.assertTrue(SWEEPS, f"nenhum config de sweep em {SWEEP_DIR}")
        compared = {name: self._check(name) for name in SWEEPS}
        print(f"\n  celulas conferidas por grade: {compared}")
        self.assertTrue(any(compared.values()),
                        "nenhuma grade tinha celula concluida para comparar -- o teste nao mediu nada")

    def test_deliberate_renames_are_only_names(self):
        '''
        Uma grade isenta do `name` tem de divergir em NADA ALEM dele. Se a isencao passasse a cobrir
        uma mudanca cientifica, os resultados novos deixariam de ser comparaveis com os antigos.
        '''
        for sweep_name, reason in NAME_EXEMPT.items():
            sweep = Sweep(PROJECT_ROOT / "configs" / "sweep" / f"{sweep_name}.yaml")
            run_dirs = self._run_dirs_by_run_id(sweep)
            if not run_dirs:
                continue
            for cell in sweep.plan.cells():
                run_dir = run_dirs.get(sweep.plan.run_id(cell))
                if not run_dir:
                    continue
                hparams_path = Path(run_dir) / "csv" / "hparams.yaml"
                if not hparams_path.exists():
                    continue
                recorded = yaml.safe_load(hparams_path.read_text()) or {}
                generated = flatten(OmegaConf.to_container(OmegaConf.merge(
                    OmegaConf.load(sweep.plan.experiment_config(cell)),
                    OmegaConf.create(sweep.plan.overrides(cell)),
                ), resolve=True))
                differing = {key for key in PINNED_KEYS
                             if str(recorded.get(key, "<ausente>")) != str(generated.get(key, "<ausente>"))
                             and not (key not in recorded and key not in generated)}
                self.assertEqual(
                    differing, {"name"},
                    msg=(f"{sweep_name} ({reason}): a isencao vale so para `name`, mas estas chaves "
                         f"tambem divergem: {sorted(differing - {'name'})}"),
                )

    def test_relative_config_path_resolves_from_any_cwd(self):
        '''
        Um notebook roda a partir do proprio diretorio. Se `Sweep` resolvesse o caminho do config
        contra o cwd, a celula `sweep_status("configs/sweep/x1.yaml")` falharia em
        `notebooks/hyphotesis_testing/` -- que e exatamente onde ela e executada.
        '''
        import os

        original = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT / "notebooks" / "hyphotesis_testing")
            sweep = Sweep("configs/sweep/x1.yaml")
            self.assertTrue(sweep.config_path.is_file(), sweep.config_path)
            self.assertEqual(sweep.plan.family, "x1")
            # E os logs tambem: relativos ao repo, nao ao diretorio do notebook.
            self.assertEqual(sweep.results_csv,
                             (PROJECT_ROOT / "logs" / "x1" / "results_per_sample.csv"))
        finally:
            os.chdir(original)

    def test_run_ids_match_what_is_already_recorded(self):
        '''
        Um `run_id` que derivasse faria o sweep retreinar celulas prontas -- caro e, pior, silencioso.
        Toda linha ja gravada tem de corresponder a uma celula da grade declarada.
        '''
        for sweep_name in SWEEPS:
            sweep = Sweep(PROJECT_ROOT / "configs" / "sweep" / f"{sweep_name}.yaml")
            recorded = set(self._run_dirs_by_run_id(sweep))
            if not recorded:
                continue
            declared = {sweep.plan.run_id(cell) for cell in sweep.plan.cells()}
            self.assertEqual(
                recorded - declared, set(),
                msg=(f"{sweep_name}: ha run_id na CSV que a grade declarada nao produz. "
                     "O sweep trataria essas celulas como pendentes e as retreinaria."),
            )


if __name__ == "__main__":
    unittest.main()
