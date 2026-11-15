'''
`ResourceMonitor` (tempo, VRAM, tamanho do modelo, carga por passo) e a tabela comparativa do E2.

Roda com:  python -m unittest tests.test_resource_monitor -v
'''

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.callbacks import ResourceMonitor
from src.evaluation import comparison_table, table_to_markdown
from tests.helpers import TinyNet


def toy_config(max_steps: int = 4) -> dict:
    dataset = {"target": "tests.helpers.ToyPatientFrames", "params": {"n_patients": 3, "size": 16}}
    return {
        "name": "toy",
        "type": "experiment",
        "model": {"target": "tests.helpers.TinyNet", "params": {"n_channels": 3, "n_classes": 2}},
        "lit_module": {"target": "src.models.ssl.mean_teacher.MeanTeacherLitWrapper",
                       "params": {"report_class_ids": {1: "FG"}, "consistency_warmup_fraction": 0.0}},
        "optimizer": {"target": "torch.optim.SGD", "params": {"lr": 0.1}},
        "trainer": {
            "params": {"max_steps": max_steps, "max_epochs": -1, "accelerator": "cpu", "devices": 1,
                       "val_check_interval": 2, "check_val_every_n_epoch": None,
                       "enable_progress_bar": False, "num_sanity_val_steps": 0},
            "checkpoint": {"monitor": "val/dice_patient", "mode": "max", "save_top_k": 1,
                           "filename": "best-{epoch}-{step}"},
            "callbacks": [
                {"target": "src.callbacks.PatientValidation", "params": {"num_classes": 2, "class_ids": {1: "FG"}}},
                {"target": "src.callbacks.ResourceMonitor", "params": {"input_size": 16, "latency_iters": 2}},
            ],
        },
        "data": {"target": "src.data.base.DataModuleFromConfig",
                 "params": {"batch_size": 4, "num_workers": 0, "train": dataset, "validation": dataset,
                            "test": dataset}},
    }


class ResourceMonitorExperimentTests(unittest.TestCase):

    def test_resources_land_in_run_json(self):
        from src.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.yaml"
            path.write_text(yaml.safe_dump(toy_config()))
            previous = os.environ.get("VFSS_TRACKING")
            os.environ["VFSS_TRACKING"] = "off"
            try:
                experiment = Experiment(config_path=path, log_dir=Path(tmp) / "logs", family="toy", run_uid="r0")
                outcome = experiment.run(seed=0)
                metadata = json.loads((experiment.experiment_run_dir / "run.json").read_text())
            finally:
                if previous is None:
                    os.environ.pop("VFSS_TRACKING", None)
                else:
                    os.environ["VFSS_TRACKING"] = previous

        resources = metadata["resources"]
        self.assertEqual(outcome["resources"], resources)
        self.assertEqual(resources["steps"], 4)
        self.assertEqual(resources["attempts"], 1)
        for key in ("fit_wall_s", "train_s", "val_s", "test_s", "s_per_step_mean"):
            self.assertGreater(resources[key], 0.0, key)
        self.assertGreaterEqual(resources["fit_wall_s"], resources["train_s"])
        expected = sum(p.numel() for p in TinyNet(3, 2).parameters()) / 1e6
        self.assertAlmostEqual(resources["params_total_m"], expected)
        self.assertAlmostEqual(resources["params_trainable_m"], expected)
        self.assertAlmostEqual(resources["teacher_params_m"], expected)   # MT: cópia EMA
        self.assertGreater(resources["gflops_per_frame"], 0.0)
        self.assertGreater(resources["latency_ms_per_frame_bs1"], 0.0)
        self.assertEqual(resources["images_per_step_student"], 4.0)
        self.assertGreater(resources["checkpoint_mb"], 0.0)
        self.assertIn(resources["best_step"], (2, 4))
        if not torch.cuda.is_available():
            self.assertEqual(resources["vram_train_peak_gb"], 0.0)


class ResourceMonitorStateTests(unittest.TestCase):

    def test_totals_accumulate_across_attempts(self):
        first = ResourceMonitor()
        first.totals.update({"fit_s": 10.0, "train_s": 8.0, "steps": 100, "attempts": 1})
        first.iteration_s = [0.1] * 100
        second = ResourceMonitor()
        second.load_state_dict(first.state_dict())
        second.totals["fit_s"] += 5.0
        second.totals["attempts"] += 1
        summary = second.summary()
        self.assertEqual(summary["fit_wall_s"], 15.0)
        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["steps"], 100)


class ComparisonTableTests(unittest.TestCase):

    def per_sample(self):
        rows = []
        for method, offset in (("supervised", 0.0), ("meanteacher", 0.1)):
            for eval_set in ("inca_test", "unm"):
                for patient in (1, 2):
                    for frame in range(3 if patient == 1 else 1):
                        value = 0.5 + offset + 0.1 * patient
                        rows.append({"method": method, "seed": 42, "eval_set": eval_set, "paciente_id": patient,
                                     "video_frame": f"v{patient}_f{frame}", "class": "FG",
                                     "dice": value, "iou": value, "assd": 1.0, "hd95": 2.0})
        return pd.DataFrame(rows)

    def test_columns_and_paired_delta(self):
        resources = pd.DataFrame([
            {"method": "supervised", "seed": 42, "fit_wall_s": 3600.0, "vram_train_peak_gb": 4.0, "params_total_m": 26.7},
            {"method": "meanteacher", "seed": 42, "fit_wall_s": 7200.0, "vram_train_peak_gb": 6.0, "params_total_m": 26.7},
        ])
        table = comparison_table(self.per_sample(), resources).set_index("method")
        self.assertAlmostEqual(table.loc["supervised", "inca_test dice"], 0.65)
        self.assertAlmostEqual(table.loc["meanteacher", "inca_test Δdice vs supervised"], 0.1)
        self.assertAlmostEqual(table.loc["supervised", "inca_test Δdice vs supervised"], 0.0)
        self.assertAlmostEqual(table.loc["meanteacher", "tempo total (h)"], 2.0)
        self.assertAlmostEqual(table.loc["meanteacher", "VRAM treino (GB)"], 6.0)
        self.assertIn("unm dice", table.columns)
        self.assertEqual(table.loc["supervised", "± entre"], "pacientes")
        markdown = table_to_markdown(table.reset_index())
        self.assertTrue(markdown.startswith("| method |"))


if __name__ == "__main__":
    unittest.main()
