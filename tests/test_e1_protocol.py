'''
As pecas do E1 fora do split: agregacao paciente -> fold, normalizacao de entrada, o adapter do smp,
os planos de tuning/CV e um `Experiment` que treina sem split de teste.

Roda com:  python -m unittest tests.test_e1_protocol -v
'''

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.evaluation import (
    aggregate_per_class,
    aggregate_per_patient,
    paired_fold_differences,
    summarize_over_folds,
)
from src.models.modules import IMAGENET_MEAN, IMAGENET_STD, InputNormalization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_SWEEP = PROJECT_ROOT / "configs" / "sweep" / "e1-tuning-search.yaml"
CONFIRM_SWEEP = PROJECT_ROOT / "configs" / "sweep" / "e1-tuning-confirm.yaml"
CV_SWEEP = PROJECT_ROOT / "configs" / "sweep" / "e1-cv.yaml"


def per_sample_rows(values_by_patient: dict, **extra) -> pd.DataFrame:
    '''`{paciente_id: [dice do frame 1, dice do frame 2, ...]}` -> tabela por frame de `evaluate_per_class`.'''
    rows = []
    for patient, values in values_by_patient.items():
        for index, value in enumerate(values):
            rows.append({"video_frame": f"v{patient}_f{index}", "video_id": patient, "paciente_id": patient,
                         "class": "FG", "dice": value, "iou": value, "assd": 1.0 - value, "hd95": 2.0, **extra})
    return pd.DataFrame(rows)


class PatientAggregationTests(unittest.TestCase):

    def test_every_patient_counts_once(self):
        # Paciente 1: 9 frames com Dice 1.0; paciente 2: 1 frame com Dice 0.0.
        rows = per_sample_rows({1: [1.0] * 9, 2: [0.0]})
        self.assertAlmostEqual(aggregate_per_class(rows).dice.item(), 0.9)
        per_patient = aggregate_per_patient(rows)
        self.assertAlmostEqual(per_patient.dice.item(), 0.5)
        self.assertEqual(per_patient.n_patients.item(), 2)
        self.assertEqual(per_patient.n_samples.item(), 10)

    def test_nan_frames_drop_inside_the_patient(self):
        rows = per_sample_rows({1: [1.0, float("nan")], 2: [0.5]})
        self.assertAlmostEqual(aggregate_per_patient(rows).dice.item(), 0.75)

    def test_groups_are_kept_apart(self):
        rows = pd.concat([per_sample_rows({1: [1.0]}, arch="a"), per_sample_rows({1: [0.0]}, arch="b")])
        result = aggregate_per_patient(rows, group_columns=["arch"]).set_index("arch").dice
        self.assertEqual((result["a"], result["b"]), (1.0, 0.0))

    def test_fold_summary_and_paired_differences(self):
        per_fold = pd.DataFrame([
            {"arch": arch, "fold": fold, "class": "FG", "dice": dice, "iou": dice, "assd": 0.0, "hd95": 0.0}
            for arch, dices in {"unet": [0.8, 0.9, 0.7], "fpn": [0.85, 0.95, 0.70]}.items()
            for fold, dice in enumerate(dices)
        ])
        summary = summarize_over_folds(per_fold, ["arch"]).set_index("arch")
        self.assertAlmostEqual(summary.loc["unet", "dice_mean"], 0.8)
        self.assertAlmostEqual(summary.loc["unet", "dice_std"], 0.1)
        self.assertEqual(summary.loc["unet", "n_folds"], 3)

        diffs = paired_fold_differences(per_fold, by="arch", reference="unet")
        self.assertEqual(set(diffs.arch), {"fpn"})
        self.assertEqual(sorted(diffs.dice.round(6).tolist()), [0.0, 0.05, 0.05])
        with self.assertRaises(ValueError):
            paired_fold_differences(per_fold, by="arch", reference="missing")


class InputNormalizationTests(unittest.TestCase):

    def test_none_is_identity(self):
        x = torch.rand(2, 3, 4, 4) * 2 - 1
        self.assertTrue(torch.equal(InputNormalization(None)(x), x))

    def test_imagenet_maps_dataset_range(self):
        x = torch.full((1, 3, 2, 2), -1.0)   # preto no dataset -> 0 em [0, 1]
        out = InputNormalization("imagenet")(x)
        expected = -torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD)
        self.assertTrue(torch.allclose(out[0, :, 0, 0], expected))

    def test_buffers_do_not_enter_the_state_dict(self):
        self.assertEqual(dict(InputNormalization("imagenet").state_dict()), {})

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            InputNormalization("cifar")


class SMPModelTests(unittest.TestCase):

    def test_every_e1_architecture_builds_and_keeps_resolution(self):
        from src.models.smp.adapters import SMPModel
        for arch in ("Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus"):
            with self.subTest(arch=arch):
                model = SMPModel(arch, n_channels=3, n_classes=2, encoder_weights=None,
                                 input_normalization="imagenet").eval()
                self.assertEqual(model.n_classes, 2)
                with torch.no_grad():
                    out = model(torch.rand(2, 3, 64, 64) * 2 - 1)
                self.assertEqual(tuple(out.shape), (2, 2, 64, 64))

    def test_e1_configs_point_to_buildable_models(self):
        for path in sorted((PROJECT_ROOT / "configs" / "experiment" / "e1").glob("*.yaml")):
            config = OmegaConf.load(path)
            with self.subTest(config=path.name):
                self.assertEqual(config.model.params.input_normalization, "imagenet")
                self.assertEqual(config.trainer.checkpoint.monitor, "val/dice_patient")
                self.assertEqual(config.data.params.shared_dataset_params.split_mode, "protocol")
                self.assertEqual(config.data.params.shared_dataset_params.target_variant, "binary_all")


class E1PlanTests(unittest.TestCase):
    '''A busca TPE, a confirmacao das top-k e o CV. Nenhum treina: so os planos e o study.'''

    def setUp(self):
        from src.sweeps.e1 import E1ConfirmSweep, E1CVSweep, E1TuningSweep
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name)
        self.tuning_cls, self.confirm_cls, self.cv_cls = E1TuningSweep, E1ConfirmSweep, E1CVSweep

    def tearDown(self):
        self.tmp.cleanup()

    def _tuning(self, config=None):
        return self.tuning_cls(config or OmegaConf.load(TUNING_SWEEP), log_dir=self.log_dir)

    def _finish(self, plan, cell, values_by_patient, pruned=False):
        '''Fecha a celula como o `Sweep` faz: linhas na CSV primeiro, `finish_cell` depois.'''
        rows = per_sample_rows(values_by_patient, run_id=plan.run_id(cell), pruned=pruned, **cell)
        csv = self.log_dir / plan.family / plan.results_csv_name()
        csv.parent.mkdir(parents=True, exist_ok=True)
        previous = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        pd.concat([previous, rows], ignore_index=True).to_csv(csv, index=False)
        plan.finish_cell(cell, None, rows)
        return rows

    def test_search_grid_is_25_trials_per_architecture_trial_outermost(self):
        cells = self._tuning().cells()
        self.assertEqual(len(cells), 25 * 5)
        self.assertEqual([c["trial"] for c in cells[:6]], [0, 0, 0, 0, 0, 1])
        self.assertEqual(list(cells[0]), ["seed", "trial", "arch"])

    def test_search_overrides_need_an_open_trial(self):
        plan = self._tuning()
        plan.begin()
        with self.assertRaises(RuntimeError):
            plan.overrides({"seed": 42, "trial": 0, "arch": "unet"})

    def test_search_cell_is_a_short_fold_zero_run_without_test(self):
        plan = self._tuning()
        plan.begin(fast_dev_run=True)
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        plan.start_cell(cell)
        overrides = plan.overrides(cell)
        params = overrides["data"]["params"]
        self.assertIsNone(params["test"])
        self.assertEqual(params["batch_size"], 8)
        self.assertEqual(params["shared_dataset_params"]["train_folds"], [0])
        self.assertIsNone(params["shared_dataset_params"]["cv_fold"])
        self.assertEqual(overrides["trainer"]["params"]["max_steps"], 4000)
        self.assertEqual(overrides["lr_scheduler"]["params"]["T_max"], 4000)
        self.assertIsNone(overrides["trainer"]["params"]["check_val_every_n_epoch"])
        point = overrides["optimizer"]["params"]
        self.assertTrue(1e-5 <= point["lr"] <= 1e-3)
        self.assertTrue(1e-6 <= point["weight_decay"] <= 1e-3)
        from src.tuning import OptunaPruning
        self.assertIsInstance(plan.callbacks(cell)[0], OptunaPruning)

    def test_startup_points_are_shared_between_architectures(self):
        plan = self._tuning()
        plan.begin(fast_dev_run=True)
        points = {}
        for arch in ("unet", "fpn"):
            cell = {"seed": 42, "trial": 0, "arch": arch}
            plan.start_cell(cell)
            points[arch] = plan.overrides(cell)["optimizer"]
        self.assertEqual(points["unet"], points["fpn"])

    def test_fast_dev_run_leaves_no_study_on_disk(self):
        plan = self._tuning()
        plan.begin(fast_dev_run=True)
        plan.start_cell({"seed": 42, "trial": 0, "arch": "unet"})
        self.assertFalse(plan.storage_path.exists())

    def test_tpe_takes_over_after_startup_and_tells_values(self):
        from optuna.trial import TrialState
        config = OmegaConf.load(TUNING_SWEEP)
        config.optuna.sampler.n_startup_trials = 2
        plan = self._tuning(config)
        plan.begin()
        for trial in range(4):
            cell = {"seed": 42, "trial": trial, "arch": "unet"}
            plan.start_cell(cell)
            lr = plan.overrides(cell)["optimizer"]["params"]["lr"]
            self._finish(plan, cell, {1: [lr * 1000], 2: [lr * 1000]})   # objetivo cresce com lr
        trials = plan.study("unet", 42).get_trials(deepcopy=False)
        self.assertEqual([t.state for t in trials], [TrialState.COMPLETE] * 4)
        self.assertAlmostEqual(trials[0].value, trials[0].params["lr"] * 1000)

    def test_finished_cell_is_released(self):
        '''
        O Trainer prende `pl_module.log` em todo callback; um `OptunaPruning` que sobrevive a celula
        mantem vivo o modulo, o trainer e o datamodule dela. O plano nao pode guardar nenhum.
        '''
        plan = self._tuning()
        plan.begin()
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        plan.start_cell(cell)
        self.assertEqual(len(plan.callbacks(cell)), 1)
        self._finish(plan, cell, {1: [0.8], 2: [0.7]})
        self.assertEqual(plan._pruning, {})
        self.assertEqual(plan._trials, {})

    def test_restart_resumes_the_same_running_trial(self):
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        first = self._tuning()
        first.begin()
        first.start_cell(cell)
        point = first.overrides(cell)["optimizer"]
        # Queda no meio do treino: um processo novo reabre o journal.
        second = self._tuning()
        second.begin()
        second.start_cell(cell)
        self.assertEqual(second.overrides(cell)["optimizer"], point)
        self.assertEqual(len(second.study("unet", 42).get_trials(deepcopy=False)), 1)

    def test_begin_tells_trials_already_in_the_csv(self):
        from optuna.trial import TrialState
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        plan = self._tuning()
        plan.begin()
        plan.start_cell(cell)
        rows = per_sample_rows({1: [0.8], 2: [0.6]}, run_id=plan.run_id(cell), pruned=False, **cell)
        csv = self.log_dir / plan.family / plan.results_csv_name()
        rows.to_csv(csv, index=False)   # CSV gravada, processo caiu antes do `tell`
        restarted = self._tuning()
        restarted.begin()
        trial = restarted.study("unet", 42).get_trials(deepcopy=False)[0]
        self.assertEqual(trial.state, TrialState.COMPLETE)
        self.assertAlmostEqual(trial.value, 0.7)

    def test_pruned_cell_is_told_as_pruned(self):
        from optuna.trial import TrialState
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        plan = self._tuning()
        plan.begin()
        plan.start_cell(cell)
        plan._trials[plan.run_id(cell)].report(0.1, step=500)
        self._finish(plan, cell, {1: [0.1]}, pruned=True)
        self.assertEqual(plan.study("unet", 42).get_trials(deepcopy=False)[0].state, TrialState.PRUNED)

    def test_fixed_trials_come_first(self):
        config = OmegaConf.load(TUNING_SWEEP)
        config.search_space.fixed_trials = [{"lr": 1e-3, "weight_decay": 1e-4}]
        plan = self._tuning(config)
        plan.begin(fast_dev_run=True)
        cell = {"seed": 42, "trial": 0, "arch": "unet"}
        plan.start_cell(cell)
        self.assertEqual(plan.overrides(cell)["optimizer"]["params"], {"lr": 1e-3, "weight_decay": 1e-4})

    # ------------------------------------------------------------------ confirmacao

    def _search_results(self, n_trials: int, archs=("unet", "fpn"), drop=()):
        '''CSV da busca: o trial t tem Dice 0.1 * t; o trial 3 foi podado (e seria o melhor).'''
        frames = []
        for arch in archs:
            for trial in range(n_trials):
                if (arch, trial) in drop:
                    continue
                frames.append(per_sample_rows({1: [0.1 * trial], 2: [0.1 * trial]}, arch=arch, trial=trial,
                                              seed=42, lr=1e-4 * (trial + 1), weight_decay=1e-5,
                                              pruned=(trial == 3), run_id=f"{arch}-t{trial:02d}-s42"))
        csv = self.log_dir / "e1-tuning-search" / "results_per_sample.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(csv, index=False)

    def _confirm(self, n_trials: int):
        source = OmegaConf.load(TUNING_SWEEP)
        source.search_space.n_trials = n_trials
        source_path = self.log_dir / "search.yaml"
        OmegaConf.save(source, source_path)
        config = OmegaConf.load(CONFIRM_SWEEP)
        config.source_sweep = str(source_path)
        config.axes.arch = ["unet", "fpn"]
        return self.confirm_cls(config, log_dir=self.log_dir)

    def test_confirm_takes_the_top3_completed_trials(self):
        self._search_results(5)
        plan = self._confirm(5)
        cells = plan.cells()
        # Trial 3 podado fica de fora: top-3 = 4, 2, 1. `stage1_rank` e o eixo mais externo.
        self.assertEqual([(c["stage1_rank"], c["arch"], c["trial"]) for c in cells],
                         [(1, "unet", 4), (1, "fpn", 4), (2, "unet", 2), (2, "fpn", 2),
                          (3, "unet", 1), (3, "fpn", 1)])
        overrides = plan.overrides(cells[0])
        self.assertAlmostEqual(overrides["optimizer"]["params"]["lr"], 5e-4)
        self.assertEqual(overrides["trainer"]["params"]["max_steps"], 8000)
        self.assertEqual(overrides["lr_scheduler"]["params"]["T_max"], 8000)
        self.assertEqual(overrides["data"]["params"]["batch_size"], 8)
        self.assertIsNone(overrides["data"]["params"]["test"])
        self.assertEqual(plan.run_id(cells[0]), "unet-t04-s42")

    def test_confirm_refuses_an_incomplete_search(self):
        self._search_results(5, drop={("fpn", 4)})
        with self.assertRaises(RuntimeError):
            self._confirm(5).cells()

    # ------------------------------------------------------------------ CV e congelamento

    def test_cv_refuses_to_train_before_the_freeze(self):
        config = OmegaConf.load(CV_SWEEP)
        config.hparams_file = "/nonexistent/e1-hparams.yaml"
        with self.assertRaises(FileNotFoundError):
            self.cv_cls(config).overrides({"seed": 42, "fold": 0, "arch": "unet"})

    def test_freeze_round_trip(self):
        from src.tuning import rank_trials, write_frozen_hparams
        rows = pd.concat([
            per_sample_rows({1: [0.6], 2: [0.6]}, arch="unet", trial=0, lr=1e-4, weight_decay=1e-5, seed=42),
            per_sample_rows({1: [0.9], 2: [0.7]}, arch="unet", trial=1, lr=3e-4, weight_decay=2e-5, seed=42),
        ])
        ranked = rank_trials(rows)
        self.assertEqual(ranked.iloc[0].trial, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_frozen_hparams(ranked[ranked["rank"] == 1], Path(tmp) / "e1-hparams.yaml", source="teste")
            with self.assertRaises(FileExistsError):
                write_frozen_hparams(ranked[ranked["rank"] == 1], path, source="teste")
            config = OmegaConf.load(CV_SWEEP)
            config.hparams_file = str(path)
            overrides = self.cv_cls(config).overrides({"seed": 42, "fold": 2, "arch": "unet"})
        self.assertEqual(overrides["optimizer"]["params"], {"lr": 3e-4, "weight_decay": 2e-5})
        self.assertEqual(overrides["data"]["params"]["batch_size"], 8)
        shared = overrides["data"]["params"]["shared_dataset_params"]
        self.assertEqual((shared["cv_fold"], shared["train_folds"]), (2, None))

    def test_rank_trials_drops_pruned_trials(self):
        from src.tuning import rank_trials
        rows = pd.concat([
            per_sample_rows({1: [0.9]}, arch="unet", trial=0, pruned=True),
            per_sample_rows({1: [0.5]}, arch="unet", trial=1, pruned=False),
        ])
        self.assertEqual(rank_trials(rows).trial.tolist(), [1])
        self.assertEqual(rank_trials(rows, completed_only=False).trial.tolist(), [0, 1])


class OptunaPruningTests(unittest.TestCase):

    def _trial(self, pruner):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        return study.ask()

    @staticmethod
    def _trainer(value, step=500):
        from types import SimpleNamespace
        return SimpleNamespace(sanity_checking=False, should_stop=False, global_step=step,
                               callback_metrics={"val/dice_patient": torch.tensor(value)})

    def test_reports_and_stops_when_the_pruner_says_so(self):
        import optuna
        from src.tuning import OptunaPruning
        callback = OptunaPruning(self._trial(optuna.pruners.ThresholdPruner(lower=0.5)))
        trainer = self._trainer(0.25)
        callback.on_validation_end(trainer, None)
        self.assertTrue(callback.pruned)
        self.assertTrue(trainer.should_stop)
        self.assertEqual(callback.trial.study.trials[0].intermediate_values, {500: 0.25})

    def test_keeps_training_otherwise(self):
        import optuna
        from src.tuning import OptunaPruning
        callback = OptunaPruning(self._trial(optuna.pruners.ThresholdPruner(lower=0.5)))
        trainer = self._trainer(0.8)
        callback.on_validation_end(trainer, None)
        self.assertFalse(callback.pruned)
        self.assertFalse(trainer.should_stop)


class BatchSizeTests(unittest.TestCase):
    '''E1 supervisionado: 8 imagens rotuladas por passo, no YAML base e em todos os sweeps.'''

    def test_e1_configs_and_sweeps_use_batch_8(self):
        for path in sorted((PROJECT_ROOT / "configs" / "experiment" / "e1").glob("*.yaml")):
            with self.subTest(config=path.name):
                self.assertEqual(OmegaConf.load(path).data.params.batch_size, 8)
        for path in (TUNING_SWEEP, CONFIRM_SWEEP, CV_SWEEP):
            with self.subTest(sweep=path.name):
                self.assertEqual(OmegaConf.load(path).training.batch_size, 8)


class ExperimentWithoutTestTests(unittest.TestCase):
    '''Treinar sem split de teste: e o que o tuning faz, e `test_dataloader` levantaria KeyError.'''

    def _config(self, with_test: bool) -> dict:
        dataset = {"target": "tests.helpers.ToyPatientFrames", "params": {"n_patients": 3, "size": 16}}
        data = {"batch_size": 4, "num_workers": 0, "train": dataset, "validation": dataset}
        if with_test:
            data["test"] = dataset
        return {
            "name": "toy",
            "type": "experiment",
            "model": {"target": "tests.helpers.TinyNet", "params": {"n_channels": 3, "n_classes": 2}},
            "lit_module": {"target": "src.models.ssl.supervised.SupervisedLitWrapper",
                           "params": {"report_class_ids": {1: "FG"}}},
            "optimizer": {"target": "torch.optim.SGD", "params": {"lr": 0.1}},
            "trainer": {
                "params": {"max_steps": 4, "max_epochs": -1, "accelerator": "cpu", "devices": 1,
                           "val_check_interval": 2, "check_val_every_n_epoch": None,
                           "enable_progress_bar": False, "num_sanity_val_steps": 0},
                "checkpoint": {"monitor": "val/dice_patient", "mode": "max", "save_top_k": 1},
                "callbacks": [{"target": "src.callbacks.PatientValidation",
                               "params": {"num_classes": 2, "class_ids": {1: "FG"}}}],
            },
            "data": {"target": "src.data.base.DataModuleFromConfig", "params": data},
        }

    def _run(self, with_test: bool):
        from src.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.yaml"
            path.write_text(yaml.safe_dump(self._config(with_test)))
            previous = os.environ.get("VFSS_TRACKING")
            os.environ["VFSS_TRACKING"] = "off"
            try:
                experiment = Experiment(config_path=path, log_dir=Path(tmp) / "logs", family="toy", run_uid="r0")
                outcome = experiment.run(seed=0)
                metrics = pd.read_csv(Path(outcome["output_dir"]) / "csv" / "metrics.csv")
            finally:
                if previous is None:
                    os.environ.pop("VFSS_TRACKING", None)
                else:
                    os.environ["VFSS_TRACKING"] = previous
        return outcome, metrics

    def test_no_test_split_skips_testing_but_keeps_the_best_checkpoint(self):
        outcome, metrics = self._run(with_test=False)
        self.assertEqual(outcome["test_metrics"], {})
        self.assertTrue(outcome["best_checkpoint_path"])
        self.assertIn("val/dice_patient", metrics.columns)
        self.assertTrue(metrics["val/dice_patient"].notna().any())

    def test_a_declared_test_split_is_still_tested(self):
        outcome, _ = self._run(with_test=True)
        self.assertIn("test/dice_score", outcome["test_metrics"])


class ExplodingNet(torch.nn.Module):
    '''Falha no primeiro forward: o stand-in de um erro de treino (o BatchNorm com 1 valor por canal).'''

    def __init__(self, n_channels: int = 3, n_classes: int = 2, **_):
        super().__init__()
        self.n_channels, self.n_classes = n_channels, n_classes
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        raise ValueError("Expected more than 1 value per channel when training")


class DropLastTrainTests(unittest.TestCase):
    '''
    Um ultimo lote de 1 frame quebra qualquer BatchNorm com 1 valor por canal (o ramo de pooling do
    ASPP do DeepLabV3+). `drop_last_train` o descarta, so no loader de treino.
    '''

    def _loaders(self, **kwargs):
        from src.data.base import DataModuleFromConfig
        dataset = {"target": "tests.helpers.ToyPatientFrames", "params": {"n_patients": 3, "frames_per_patient": 4}}
        # 12 frames em lotes de 11: 11 + 1.
        module = DataModuleFromConfig(batch_size=11, num_workers=0, train=dataset, validation=dataset, **kwargs)
        module.setup("fit")
        return module.train_dataloader(), module.val_dataloader()

    def test_default_keeps_the_trailing_batch(self):
        train, _ = self._loaders()
        self.assertEqual([len(batch["image"]) for batch in train], [11, 1])

    def test_drop_last_train_drops_the_trailing_batch_of_the_train_loader_only(self):
        train, validation = self._loaders(drop_last_train=True)
        self.assertEqual([len(batch["image"]) for batch in train], [11])
        # A validacao tem de ver todos os frames: descartar um mudaria o numero publicado.
        self.assertEqual([len(batch["image"]) for batch in validation], [11, 1])

    def test_e1_configs_enable_it(self):
        for path in sorted((PROJECT_ROOT / "configs" / "experiment" / "e1").glob("*.yaml")):
            with self.subTest(config=path.name):
                self.assertTrue(OmegaConf.load(path).data.params.drop_last_train)


class FailedRunRecordsItsTracebackTests(unittest.TestCase):
    '''O erro de treino tem de ficar no `console.log` e no `run.json`, nao so no terminal.'''

    def test_traceback_is_in_console_log_and_run_json(self):
        import json
        from src.experiment import Experiment
        config = ExperimentWithoutTestTests()._config(with_test=False)
        config["model"]["target"] = "tests.test_e1_protocol.ExplodingNet"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.yaml"
            path.write_text(yaml.safe_dump(config))
            previous = os.environ.get("VFSS_TRACKING")
            os.environ["VFSS_TRACKING"] = "off"
            try:
                experiment = Experiment(config_path=path, log_dir=Path(tmp) / "logs", family="toy", run_uid="r0")
                with self.assertRaises(ValueError):
                    experiment.run(seed=0)
                run_dir = experiment.experiment_run_dir
                console = (run_dir / "console.log").read_text()
                metadata = json.loads((run_dir / "run.json").read_text())
            finally:
                if previous is None:
                    os.environ.pop("VFSS_TRACKING", None)
                else:
                    os.environ["VFSS_TRACKING"] = previous
        self.assertIn("Traceback (most recent call last)", console)
        self.assertIn("ValueError: Expected more than 1 value per channel", console)
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("Traceback (most recent call last)", metadata["traceback"])


class PrunedRunStopsEarlyTests(unittest.TestCase):
    '''Um trial podado tem de PARAR o treino e ainda terminar como run concluido, com melhor checkpoint.'''

    def test_pruned_experiment_finishes_before_max_steps(self):
        import json
        import optuna
        from src.experiment import Experiment
        from src.tuning import OptunaPruning
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trial = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.ThresholdPruner(lower=2.0)).ask()
        pruning = OptunaPruning(trial)
        config = ExperimentWithoutTestTests()._config(with_test=False)
        config["trainer"]["params"]["max_steps"] = 6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.yaml"
            path.write_text(yaml.safe_dump(config))
            previous = os.environ.get("VFSS_TRACKING")
            os.environ["VFSS_TRACKING"] = "off"
            try:
                experiment = Experiment(config_path=path, log_dir=Path(tmp) / "logs", family="toy",
                                        run_uid="r0", callbacks=[pruning])
                outcome = experiment.run(seed=0)
                metadata = json.loads((experiment.experiment_run_dir / "run.json").read_text())
            finally:
                if previous is None:
                    os.environ.pop("VFSS_TRACKING", None)
                else:
                    os.environ["VFSS_TRACKING"] = previous
        self.assertTrue(pruning.pruned)
        self.assertEqual(metadata["status"], "done")
        self.assertEqual(metadata["final_step"], 2)   # primeira validacao (val_check_interval=2)
        self.assertTrue(outcome["best_checkpoint_path"])
        self.assertEqual(list(trial.study.trials[0].intermediate_values), [2])


if __name__ == "__main__":
    unittest.main()
