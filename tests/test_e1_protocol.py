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
TUNING_SWEEP = PROJECT_ROOT / "configs" / "sweep" / "e1-tuning.yaml"
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

    def setUp(self):
        from src.sweeps.e1 import E1CVSweep, E1TuningSweep
        self.tuning = E1TuningSweep(OmegaConf.load(TUNING_SWEEP))
        self.cv_cls = E1CVSweep

    def test_trial_points_are_deterministic_and_in_bounds(self):
        from src.sweeps.e1 import E1TuningSweep
        points = self.tuning.trial_points
        self.assertEqual(points, E1TuningSweep(OmegaConf.load(TUNING_SWEEP)).trial_points)
        self.assertEqual(len(points), 8)
        for point in points:
            self.assertTrue(1e-5 <= point["lr"] <= 1e-3)
            self.assertTrue(1e-6 <= point["weight_decay"] <= 1e-3)

    def test_architectures_share_the_same_points(self):
        a = self.tuning.overrides({"seed": 42, "trial": 3, "arch": "unet"})
        b = self.tuning.overrides({"seed": 42, "trial": 3, "arch": "fpn"})
        self.assertEqual(a["optimizer"], b["optimizer"])

    def test_tuning_has_no_test_split_and_trains_on_fold_zero(self):
        overrides = self.tuning.overrides({"seed": 42, "trial": 0, "arch": "unet"})
        params = overrides["data"]["params"]
        self.assertIsNone(params["test"])
        self.assertEqual(params["shared_dataset_params"]["train_folds"], [0])
        self.assertIsNone(params["shared_dataset_params"]["cv_fold"])
        self.assertEqual(overrides["trainer"]["params"]["val_check_interval"], 500)
        self.assertIsNone(overrides["trainer"]["params"]["check_val_every_n_epoch"])
        self.assertEqual(overrides["lr_scheduler"]["params"]["T_max"], 8000)

    def test_fixed_trials_come_first_and_candidates_filter(self):
        from src.sweeps.e1 import E1TuningSweep
        config = OmegaConf.load(TUNING_SWEEP)
        config.search_space.fixed_trials = [{"lr": 1e-3, "weight_decay": 1e-4}]
        config.candidates = {"unet": [0, 2]}
        plan = E1TuningSweep(config)
        self.assertEqual(plan.trial_points[0], {"lr": 1e-3, "weight_decay": 1e-4})
        self.assertEqual(plan.trial_points[1:], self.tuning.trial_points)
        self.assertEqual({(c["arch"], c["trial"]) for c in plan.cells()}, {("unet", 0), ("unet", 2)})

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
        shared = overrides["data"]["params"]["shared_dataset_params"]
        self.assertEqual((shared["cv_fold"], shared["train_folds"]), (2, None))


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


if __name__ == "__main__":
    unittest.main()
