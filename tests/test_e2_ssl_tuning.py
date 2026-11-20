'''
Tuning dos pesos SSL do E2 (`src/sweeps/e2_tuning.py`, docs/e2-tuning-ssl.md): busca, confirmação,
herança do MT temporal, congelamento e a leitura do congelado pelo `E2Sweep`. Nenhum teste treina.

    python -m unittest tests.test_e2_ssl_tuning -v
'''
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from omegaconf import OmegaConf

from src.sweep import shard_cells
from src.sweeps.e2 import E2CVSweep, E2Sweep
from src.sweeps.e2_tuning import E2TuningConfirmSweep, E2TuningSweep, freeze_ssl_hparams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH = PROJECT_ROOT / "configs" / "sweep" / "e2-ssl-tuning-search.yaml"
CONFIRM = PROJECT_ROOT / "configs" / "sweep" / "e2-ssl-tuning-confirm.yaml"
GROUP = {"seed": 42, "arch": "deeplabv3plus", "unlabeled_policy": "global"}


def per_sample_rows(values_by_patient: dict, **extra) -> pd.DataFrame:
    rows = []
    for patient, values in values_by_patient.items():
        for index, value in enumerate(values):
            rows.append({"video_frame": f"v{patient}_f{index}", "video_id": patient, "paciente_id": patient,
                         "class": "FG", "dice": value, "iou": value, "assd": 1.0 - value, "hd95": 2.0, **extra})
    return pd.DataFrame(rows)


def fake_experiment(tmp: Path):
    return SimpleNamespace(resources={}, experiment_run_dir=tmp / "run", extra_callbacks=[])


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def search(self, config=None) -> E2TuningSweep:
        config = config or OmegaConf.load(SEARCH)
        config.inherit_from = str(self.confirm_path())
        return E2TuningSweep(config, log_dir=self.log_dir)

    def confirm_path(self) -> Path:
        '''A confirmação apontando para a busca deste teste (os YAMLs reais, com caminhos trocados).'''
        path = self.log_dir / "confirm.yaml"
        if not path.exists():
            search = OmegaConf.load(SEARCH)
            search.inherit_from = str(path)
            search_path = self.log_dir / "search.yaml"
            OmegaConf.save(search, search_path)
            config = OmegaConf.load(CONFIRM)
            config.source_sweep = str(search_path)
            OmegaConf.save(config, path)
        return path

    def confirm(self, methods=None) -> E2TuningConfirmSweep:
        config = OmegaConf.load(self.confirm_path())
        if methods is not None:
            config.axes.method = methods
        return E2TuningConfirmSweep(config, log_dir=self.log_dir)

    def append_csv(self, family: str, rows: pd.DataFrame) -> None:
        csv = self.log_dir / family / "results_per_sample.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        previous = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        pd.concat([previous, rows], ignore_index=True).to_csv(csv, index=False)

    def write_search_results(self, method: str, hp, n_trials=25, pruned=(), score=None):
        '''Busca completa de `method`: trial t com Dice `score(t)` (padrão 0.01 * t) e pontos `hp(t)`.'''
        score = score or (lambda t: 0.01 * t)
        frames = []
        for trial in range(n_trials):
            point = {f"hp_{k}": v for k, v in hp(trial).items()}
            run_id = f"deeplabv3plus-{method}-global-t{trial:02d}-s42"
            frames.append(per_sample_rows({1: [score(trial)], 2: [score(trial)]}, run_id=run_id, method=method,
                                          trial=trial, pruned=trial in pruned, **GROUP, **point))
        self.append_csv("e2-ssl-tuning-search", pd.concat(frames, ignore_index=True))

    def write_confirm_results(self, method: str, plan: E2TuningConfirmSweep, scores: dict):
        frames = []
        for cell in plan.cells():
            if cell["method"] != method:
                continue
            point = {f"hp_{k}": v for k, v in plan.point(cell).items()}
            frames.append(per_sample_rows({1: [scores[cell["trial"]]]}, run_id=plan.run_id(cell),
                                          **cell, **point))
        self.append_csv("e2-ssl-tuning-confirm", pd.concat(frames, ignore_index=True))


class SearchGridTests(_Base):

    def test_reference_first_then_trials_outermost(self):
        cells = self.search().cells()
        self.assertEqual(len(cells), 1 + 25 * 3)
        self.assertEqual((cells[0]["method"], cells[0]["trial"]), ("supervised", None))
        self.assertEqual([(c["trial"], c["method"]) for c in cells[1:5]],
                         [(0, "meanteacher"), (0, "fixmatch"), (0, "dtc-trus"), (1, "meanteacher")])

    def test_inheriting_method_comes_last(self):
        config = OmegaConf.load(SEARCH)
        config.axes.method = ["mt-temporal-prev", "meanteacher"]
        cells = self.search(config).cells()
        self.assertEqual({c["method"] for c in cells[:25]}, {"meanteacher"})
        self.assertEqual({c["method"] for c in cells[25:]}, {"mt-temporal-prev"})

    def test_run_ids(self):
        plan = self.search()
        self.assertEqual(plan.run_id({**GROUP, "trial": None, "method": "supervised"}),
                         "deeplabv3plus-supervised-global-s42")
        self.assertEqual(plan.run_id({**GROUP, "trial": 7, "method": "dtc-trus"}),
                         "deeplabv3plus-dtc-trus-global-t07-s42")

    def test_shards_keep_a_method_and_dtc_with_its_teacher(self):
        plan = self.search()
        cells = plan.cells()
        by_shard = {k: {(c["method"]) for c in shard_cells(plan, cells, (k, 3))} for k in (1, 2, 3)}
        self.assertEqual(sorted(map(sorted, by_shard.values())),
                         [["dtc-trus", "supervised"], ["fixmatch"], ["meanteacher"]])

    def test_spaces_follow_the_two_decade_rule(self):
        '''valor atual / 10 a * 10, com a única exceção declarada (teto do proto em 3, o ponto do artigo).'''
        spaces = self.search().method_spaces()
        defaults = {("meanteacher", "consistency_weight"): 0.1, ("fixmatch", "consistency_weight"): 0.1,
                    ("mt-temporal-prev", "temporal_weight"): 0.1, ("dtc-trus", "kd_weight"): 1.0,
                    ("dtc-trus", "con_weight"): 2.0}
        for (method, param), value in defaults.items():
            spec = spaces[method]["params"][param]
            self.assertAlmostEqual(spec["low"], value / 10)
            self.assertAlmostEqual(spec["high"], value * 10)
            self.assertTrue(spec["log"])
            self.assertEqual(spaces[method]["fixed_trials"][0][param], value)
        self.assertEqual(spaces["dtc-trus"]["params"]["proto_weight"], {"low": 0.01, "high": 3.0, "log": True})
        self.assertEqual(spaces["mt-temporal-prev"]["inherit"], {"consistency_weight": "meanteacher"})

    def test_fixed_values_match_the_method_yamls(self):
        '''O trial 0 é o E2 "padrão": os pesos dos YAMLs base, não um número copiado à parte.'''
        spaces = self.search().method_spaces()
        files = OmegaConf.load(SEARCH).base_configs
        for method, space in spaces.items():
            lit = OmegaConf.load(str(PROJECT_ROOT / files[method])).lit_module.params
            for param, value in space["fixed_trials"][0].items():
                self.assertAlmostEqual(float(lit[param]), value, msg=f"{method}.{param}")


class SearchCellTests(_Base):

    def open(self, plan, method, trial):
        cell = {**GROUP, "trial": trial, "method": method}
        plan.start_cell(cell)
        return cell

    def test_trial_cell_is_a_short_fold_zero_run_without_test(self):
        plan = self.search()
        plan.begin(fast_dev_run=True)
        cell = self.open(plan, "meanteacher", 0)
        overrides = plan.overrides(cell)
        params = overrides["data"]["params"]
        self.assertIsNone(params["test"])
        self.assertEqual(params["cache_splits"], ["validation"])
        shared = params["shared_dataset_params"]
        self.assertEqual((shared["train_folds"], shared["cv_fold"], shared["holdout_test"]), ([0], None, False))
        self.assertEqual(overrides["trainer"]["params"]["max_steps"], 4000)
        self.assertEqual(overrides["lr_scheduler"]["params"]["T_max"], 4000)
        self.assertEqual(params["train_batch_sampler"]["params"]["labeled_batch_size"], 8)
        frozen = yaml.safe_load((PROJECT_ROOT / "configs/frozen/e1-hparams.yaml").read_text())["hparams"]
        self.assertEqual(overrides["optimizer"]["params"]["lr"], frozen["deeplabv3plus"]["lr"])
        self.assertEqual(overrides["lit_module"]["params"], {"consistency_weight": 0.1})   # trial 0 fixo
        from src.tuning import OptunaPruning
        self.assertIsInstance(plan.callbacks(cell)[0], OptunaPruning)

    def test_reference_cell_runs_the_full_schedule_without_a_trial(self):
        plan = self.search()
        plan.begin(fast_dev_run=True)
        cell = self.open(plan, "supervised", None)
        overrides = plan.overrides(cell)
        self.assertEqual(overrides["trainer"]["params"]["max_steps"], 8000)
        self.assertEqual(overrides["lr_scheduler"]["params"]["T_max"], 8000)
        self.assertNotIn("lit_module", overrides)
        self.assertEqual(plan.callbacks(cell), [])
        self.assertEqual(overrides["data"]["params"]["shared_dataset_params"]["train_folds"], [0])

    def test_dtc_trials_use_the_reference_supervised_as_teacher(self):
        plan = self.search()
        plan.begin(fast_dev_run=True)
        for trial in (0, 1):
            cell = self.open(plan, "dtc-trus", trial)
            with self.assertRaisesRegex(RuntimeError, "deeplabv3plus-supervised-global-s42"):
                plan.overrides(cell)
        checkpoint = self.log_dir / "teacher.ckpt"
        checkpoint.write_bytes(b"x")
        plan.sweep_dir.mkdir(parents=True, exist_ok=True)
        (plan.sweep_dir / "state.json").write_text(json.dumps({"cells": {"deeplabv3plus-supervised-global-s42": {
            "status": "done", "best_checkpoint_path": str(checkpoint)}}}))
        lit = plan.overrides(cell)["lit_module"]["params"]
        self.assertEqual(lit["teacher_checkpoint"], str(checkpoint))
        self.assertEqual({k: lit[k] for k in ("kd_weight", "con_weight", "proto_weight")},
                         {"kd_weight": 3.0, "con_weight": 0.75, "proto_weight": 3.0})   # trial 1 = artigo

    def test_startup_points_are_the_same_multipliers_for_every_method(self):
        '''O sorteio inicial do TPE dá os MESMOS múltiplos do valor atual a todo método (seed 42 + n, escala log).'''
        plan = self.search()
        plan.begin(fast_dev_run=True)
        ratios = {}
        for method, (param, default) in {"meanteacher": ("consistency_weight", 0.1),
                                         "fixmatch": ("consistency_weight", 0.1),
                                         "dtc-trus": ("kd_weight", 1.0)}.items():
            ratios[method] = []
            for trial in range(10):
                cell = self.open(plan, method, trial)
                value = plan.point(cell)[param]
                plan._tell(plan._trial(cell).study, trial, per_sample_rows({1: [0.5]}))
                if trial >= 2:   # 0 e 1 são pontos fixos (o 1 só existe no FixMatch e no DTC-TRUS)
                    ratios[method].append(value / default)
        for method in ("fixmatch", "dtc-trus"):
            for a, b in zip(ratios["meanteacher"], ratios[method]):
                self.assertAlmostEqual(a, b, places=9)

    def test_values_stay_inside_the_space_after_startup(self):
        config = OmegaConf.load(SEARCH)
        config.optuna.sampler.n_startup_trials = 2
        plan = self.search(config)
        plan.begin()
        for trial in range(5):
            cell = self.open(plan, "dtc-trus", trial)
            point = plan.point(cell)
            self.assertTrue(0.1 <= point["kd_weight"] <= 10 and 0.2 <= point["con_weight"] <= 20
                            and 0.01 <= point["proto_weight"] <= 3, point)
            rows = per_sample_rows({1: [math.log10(point["kd_weight"]) + 2]}, run_id=plan.run_id(cell))
            plan.finish_cell(cell, fake_experiment(self.log_dir), rows)
        self.assertEqual(plan._trials, {})
        study = plan.study({**GROUP, "method": "dtc-trus"})
        self.assertEqual(len(study.get_trials(deepcopy=False)), 5)

    def test_fixmatch_second_fixed_trial_is_the_published_lambda(self):
        plan = self.search()
        plan.begin(fast_dev_run=True)
        first = self.open(plan, "fixmatch", 0)
        self.assertEqual(plan.point(first), {"consistency_weight": 0.1})
        plan._tell(plan._trial(first).study, 0, per_sample_rows({1: [0.5]}))
        cell = self.open(plan, "fixmatch", 1)
        self.assertEqual(plan.point(cell), {"consistency_weight": 1.0})

    def test_inheriting_method_refuses_before_the_source_is_confirmed(self):
        plan = self.search()
        plan.begin()
        with self.assertRaises(FileNotFoundError):
            self.open(plan, "mt-temporal-prev", 0)
        # Nada ficou RUNNING no journal.
        self.assertFalse(plan.storage_path.exists() and
                         plan.study({**GROUP, "method": "mt-temporal-prev"}).get_trials(deepcopy=False))


class ConfirmAndInheritanceTests(_Base):

    def setUp(self):
        super().setUp()
        self.write_search_results("meanteacher", lambda t: {"consistency_weight": 0.01 * (t + 1)}, pruned={24})

    def test_confirm_takes_the_top3_completed_trials(self):
        plan = self.confirm(["meanteacher"])
        cells = plan.cells()
        # Trial 24 podado fica de fora: top-3 = 23, 22, 21.
        self.assertEqual([(c["stage1_rank"], c["trial"]) for c in cells], [(1, 23), (2, 22), (3, 21)])
        overrides = plan.overrides(cells[0])
        self.assertAlmostEqual(overrides["lit_module"]["params"]["consistency_weight"], 0.24)
        self.assertEqual(overrides["trainer"]["params"]["max_steps"], 8000)
        self.assertIsNone(overrides["data"]["params"]["test"])
        self.assertEqual(plan.run_id(cells[0]), "deeplabv3plus-meanteacher-global-t23-s42")

    def test_confirm_refuses_an_incomplete_search(self):
        with self.assertRaisesRegex(RuntimeError, "incompleta"):
            self.confirm(["fixmatch"]).cells()
        self.write_search_results("fixmatch", lambda t: {"consistency_weight": 0.1}, n_trials=24)
        with self.assertRaisesRegex(RuntimeError, "incompleta"):
            self.confirm(["fixmatch"]).cells()

    def test_confirm_dtc_reads_the_teacher_from_the_search(self):
        self.write_search_results("dtc-trus", lambda t: {"kd_weight": 1.0 + t, "con_weight": 2.0,
                                                         "proto_weight": 0.1})
        checkpoint = self.log_dir / "teacher.ckpt"
        checkpoint.write_bytes(b"x")
        shard = self.log_dir / "e2-ssl-tuning-search" / "_sweep" / "dl05"
        shard.mkdir(parents=True)
        (shard / "state.json").write_text(json.dumps({"cells": {"deeplabv3plus-supervised-global-s42": {
            "status": "done", "best_checkpoint_path": str(checkpoint)}}}))
        plan = self.confirm(["dtc-trus"])
        cell = plan.cells()[0]
        lit = plan.overrides(cell)["lit_module"]["params"]
        self.assertEqual(lit["teacher_checkpoint"], str(checkpoint))
        self.assertEqual(lit["kd_weight"], 25.0)

    def test_temporal_inherits_the_confirmed_mean_teacher_weight(self):
        confirm = self.confirm(["meanteacher"])
        with self.assertRaises(FileNotFoundError):
            confirm.winner({**GROUP, "method": "meanteacher"})
        # Na confirmação o 2o da busca (trial 22) vence.
        self.write_confirm_results("meanteacher", confirm, {23: 0.70, 22: 0.80, 21: 0.75})
        winner = confirm.winner({**GROUP, "method": "meanteacher"})
        self.assertEqual(winner["trial"], 22)
        self.assertAlmostEqual(winner["params"]["consistency_weight"], 0.23)

        plan = self.search()
        plan.begin(fast_dev_run=True)
        cell = {**GROUP, "trial": 0, "method": "mt-temporal-prev"}
        plan.start_cell(cell)
        self.assertEqual(plan.overrides(cell)["lit_module"]["params"],
                         {"consistency_weight": 0.23, "temporal_weight": 0.1})
        self.assertEqual(plan._trial(cell).user_attrs["inherited_consistency_weight"], 0.23)

    def test_winner_refuses_a_partial_confirmation(self):
        confirm = self.confirm(["meanteacher"])
        frames = per_sample_rows({1: [0.9]}, run_id="x", **GROUP, method="meanteacher", trial=23,
                                 hp_consistency_weight=0.24)
        self.append_csv("e2-ssl-tuning-confirm", frames)
        with self.assertRaisesRegex(RuntimeError, "incompleta"):
            confirm.winner({**GROUP, "method": "meanteacher"})


class FreezeAndConsumeTests(_Base):

    def confirm_everything(self):
        spaces = {"meanteacher": lambda t: {"consistency_weight": 0.01 * (t + 1)},
                  "fixmatch": lambda t: {"consistency_weight": 0.02 * (t + 1)},
                  "dtc-trus": lambda t: {"kd_weight": 1.0 + t, "con_weight": 2.0, "proto_weight": 0.1},
                  "mt-temporal-prev": lambda t: {"consistency_weight": 0.24, "temporal_weight": 0.03 * (t + 1)}}
        for method, hp in spaces.items():
            self.write_search_results(method, hp)
            confirm = self.confirm([method])
            self.write_confirm_results(method, confirm, {24: 0.9, 23: 0.8, 22: 0.7})

    def test_freeze_needs_every_method_confirmed(self):
        self.write_search_results("meanteacher", lambda t: {"consistency_weight": 0.1})
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            freeze_ssl_hparams(self.confirm_path(), self.log_dir / "frozen.yaml", log_dir=self.log_dir)

    def test_freeze_and_e2_tuned_round_trip(self):
        self.confirm_everything()
        path = freeze_ssl_hparams(self.confirm_path(), self.log_dir / "frozen.yaml", log_dir=self.log_dir)
        with self.assertRaises(FileExistsError):
            freeze_ssl_hparams(self.confirm_path(), path, log_dir=self.log_dir)
        frozen = yaml.safe_load(path.read_text())["hparams"]["deeplabv3plus"]
        self.assertEqual(frozen["dtc-trus"]["params"], {"kd_weight": 25.0, "con_weight": 2.0, "proto_weight": 0.1})
        self.assertEqual(frozen["mt-temporal-prev"]["inherited"], {"consistency_weight": "meanteacher"})
        self.assertEqual(frozen["mt-temporal-prev"]["tuned"], ["temporal_weight"])

        config = OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / "e2-tuned.yaml")
        config.ssl_hparams_file = str(path)
        plan = E2Sweep(config, log_dir=self.log_dir)
        lit = lambda method: plan.overrides({**GROUP, "method": method}).get("lit_module", {}).get("params", {})
        self.assertEqual(lit("supervised"), {})
        self.assertAlmostEqual(lit("meanteacher")["consistency_weight"], 0.25)
        self.assertAlmostEqual(lit("fixmatch")["consistency_weight"], 0.50)
        self.assertEqual(lit("mt-temporal-bi"), {"consistency_weight": 0.24, "temporal_weight": 0.75})
        self.assertEqual(lit("mt-temporal-proto-student"), {"consistency_weight": 0.24, "temporal_weight": 0.75})
        self.assertEqual(lit("mt-temporal-nomt"), {"temporal_weight": 0.75})   # o termo MT segue 0

        cv = OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / "e2-cv-tuned.yaml")
        cv.ssl_hparams_file = str(path)
        cv_plan = E2CVSweep(cv, log_dir=self.log_dir)
        overrides = cv_plan.overrides({**GROUP, "fold": 3, "method": "fixmatch"})
        self.assertAlmostEqual(overrides["lit_module"]["params"]["consistency_weight"], 0.50)
        self.assertEqual(overrides["data"]["params"]["shared_dataset_params"]["cv_fold"], 3)

    def test_tuned_sweeps_refuse_before_the_freeze_and_unknown_methods(self):
        config = OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / "e2-tuned.yaml")
        config.ssl_hparams_file = str(self.log_dir / "missing.yaml")
        with self.assertRaises(FileNotFoundError):
            E2Sweep(config, log_dir=self.log_dir).overrides({**GROUP, "method": "fixmatch"})
        self.confirm_everything()
        path = freeze_ssl_hparams(self.confirm_path(), self.log_dir / "frozen.yaml", log_dir=self.log_dir)
        config.ssl_hparams_file = str(path)
        config.ssl_hparams_from = {}
        with self.assertRaisesRegex(KeyError, "mt-temporal-bi"):
            E2Sweep(config, log_dir=self.log_dir).overrides({**GROUP, "method": "mt-temporal-bi"})

    def test_untuned_sweeps_are_unchanged(self):
        '''e2.yaml / e2-cv.yaml não declaram `ssl_hparams_file`: só o professor entra em `lit_module`.'''
        for name in ("e2", "e2-cv"):
            config = OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / f"{name}.yaml")
            self.assertNotIn("ssl_hparams_file", config)
            cls = E2CVSweep if name == "e2-cv" else E2Sweep
            cell = {**GROUP, "method": "fixmatch", **({"fold": 0} if name == "e2-cv" else {})}
            self.assertNotIn("lit_module", cls(config, log_dir=self.log_dir).overrides(cell))

    def test_tuned_sweeps_mirror_the_untuned_ones(self):
        '''e2-tuned = e2 fora de name/family/ssl_hparams_* (idem CV): a única diferença é o peso SSL.'''
        for untuned, tuned in (("e2", "e2-tuned"), ("e2-cv", "e2-cv-tuned")):
            a = OmegaConf.to_container(OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / f"{untuned}.yaml"))
            b = OmegaConf.to_container(OmegaConf.load(PROJECT_ROOT / "configs" / "sweep" / f"{tuned}.yaml"))
            for key in ("name", "family", "ssl_hparams_file", "ssl_hparams_untuned", "ssl_hparams_from"):
                a.pop(key, None)
                b.pop(key, None)
            self.assertEqual(a, b, tuned)


if __name__ == "__main__":
    unittest.main()
