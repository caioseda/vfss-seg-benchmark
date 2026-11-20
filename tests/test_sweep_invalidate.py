'''
`src/sweep_invalidate.py` + `invalidation_targets` + `remote.invalidate_script`: retreinar SO as celulas
com problema (ex.: o FixMatch do E2) sem apagar nada e sem tocar nas outras.

    python -m unittest tests.test_sweep_invalidate -v
'''
import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src import remote
from src.sweep import archive_label, invalidation_targets
from src.sweep_invalidate import invalidate

FAMILY = "e2"
BAD = "deeplabv3plus-fixmatch-global-s42"
GOOD = "deeplabv3plus-supervised-global-s42"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cell(run_id, status="done"):
    return {"status": status, "run_uid": "2026-11-17-17-16-34", "run_name": f"{FAMILY}-{run_id}",
            "run_dir": f"/somewhere/else/{run_id}"}


class Fixture:
    '''Uma familia com canonica + shard dl05, duas celulas (uma ruim, uma boa) e seus runs.'''

    def __init__(self, root: Path, bad_status="done"):
        self.logs = root / "logs"
        family = self.logs / FAMILY
        shard = family / "_sweep" / "dl05"
        for base in (family, shard):
            write_csv(base / "results_per_sample.csv",
                      [{"run_id": BAD, "dice": "0.05", "note": "a, \"quoted\""},
                       {"run_id": GOOD, "dice": "0.87", "note": ""},
                       {"run_id": BAD, "dice": "0.04", "note": ""}])
            write_csv(base / "results_resources.csv",
                      [{"run_id": GOOD, "fit_wall_s": "10"}, {"run_id": BAD, "fit_wall_s": "11"}])
        for path in (family / "_sweep" / "state.json", shard / "state.json"):
            path.write_text(json.dumps({"family": FAMILY, "cells": {BAD: cell(BAD, bad_status), GOOD: cell(GOOD)}}))
        (shard / "configs").mkdir()
        for run_id in (BAD, GOOD):
            (shard / "configs" / f"{run_id}.yaml").write_text("name: x\n")
            run = family / "runs" / f"{FAMILY}-{run_id}" / "2026-11-17-17-16-34"
            (run / "checkpoints").mkdir(parents=True)
            (run / "checkpoints" / "best.ckpt").write_bytes(b"weights")
        self.family = family
        self.shard = shard
        self.archive = root / "arquivo" / f"{FAMILY}-bug"


class TestInvalidate(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_removes_only_the_target_and_archives_everything(self):
        fx = Fixture(self.tmp)
        report = invalidate(fx.logs, FAMILY, [BAD], fx.archive, reason="bug")
        for base in (fx.family, fx.shard):
            self.assertEqual([r["run_id"] for r in read_csv(base / "results_per_sample.csv")], [GOOD])
            self.assertEqual([r["run_id"] for r in read_csv(base / "results_resources.csv")], [GOOD])
            self.assertEqual(set(json.loads((base / "_sweep" / "state.json" if base == fx.family
                                              else base / "state.json").read_text())["cells"]), {GOOD})
        self.assertFalse((fx.family / "runs" / f"{FAMILY}-{BAD}").exists())
        self.assertTrue((fx.family / "runs" / f"{FAMILY}-{GOOD}").exists())
        self.assertFalse((fx.shard / "configs" / f"{BAD}.yaml").exists())
        self.assertTrue((fx.shard / "configs" / f"{GOOD}.yaml").exists())
        # Nada se perdeu: linhas (inclusive o texto com virgula/aspas), estado, runs com checkpoint.
        archived = read_csv(fx.archive / FAMILY / "results_per_sample.csv")
        self.assertEqual([r["dice"] for r in archived], ["0.05", "0.04"])
        self.assertEqual(archived[0]["note"], "a, \"quoted\"")
        self.assertEqual(len(read_csv(fx.archive / FAMILY / "_sweep" / "dl05" / "results_per_sample.csv")), 2)
        self.assertTrue((fx.archive / FAMILY / "runs" / f"{FAMILY}-{BAD}" / "2026-11-17-17-16-34"
                         / "checkpoints" / "best.ckpt").exists())
        state = json.loads((fx.archive / "state_removido.json").read_text())
        self.assertEqual(set(state), {f"{FAMILY}/_sweep/state.json", f"{FAMILY}/_sweep/dl05/state.json"})
        manifest = json.loads((fx.archive / "MANIFESTO.json").read_text())
        self.assertEqual(manifest["invalidacoes"][0]["motivo"], "bug")
        self.assertEqual(len(report["moved"]), 2)

    def test_dry_run_changes_nothing(self):
        fx = Fixture(self.tmp)
        before = {p: p.read_bytes() for p in fx.logs.rglob("*") if p.is_file()}
        report = invalidate(fx.logs, FAMILY, [BAD], fx.archive, dry_run=True)
        after = {p: p.read_bytes() for p in fx.logs.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(fx.archive.exists())
        self.assertEqual(report["csv_rows"][f"{FAMILY}/results_per_sample.csv"], 2)

    def test_idempotent(self):
        fx = Fixture(self.tmp)
        invalidate(fx.logs, FAMILY, [BAD], fx.archive)
        report = invalidate(fx.logs, FAMILY, [BAD], fx.archive)
        self.assertEqual((report["csv_rows"], report["state_cells"], report["moved"]), ({}, {}, []))
        self.assertEqual(len(json.loads((fx.archive / "MANIFESTO.json").read_text())["invalidacoes"]), 1)

    def test_refuses_running_cells_unless_forced(self):
        fx = Fixture(self.tmp, bad_status="running")
        report = invalidate(fx.logs, FAMILY, [BAD], fx.archive)
        self.assertIn("refused", report)
        self.assertEqual(len(read_csv(fx.family / "results_per_sample.csv")), 3)
        self.assertTrue((fx.family / "runs" / f"{FAMILY}-{BAD}").exists())
        report = invalidate(fx.logs, FAMILY, [BAD], fx.archive, force=True)
        self.assertNotIn("refused", report)
        self.assertFalse((fx.family / "runs" / f"{FAMILY}-{BAD}").exists())

    def test_remote_script_runs_with_system_python(self):
        '''O script do `remote.py invalidate`, executado por bash + python3 do sistema (fora do env).'''
        python3 = shutil.which("python3", path="/usr/bin:/bin")
        if python3 is None:
            self.skipTest("sem python3 do sistema")
        fx = Fixture(self.tmp)
        script = remote.invalidate_script({"logs_root": str(fx.logs), "family": FAMILY, "run_ids": [BAD],
                                           "archive_dir": str(fx.archive), "reason": "bug"})
        done = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True,
                              env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("REPORT ", done.stdout)
        self.assertEqual([r["run_id"] for r in read_csv(fx.shard / "results_per_sample.csv")], [GOOD])
        self.assertTrue((fx.archive / "MANIFESTO.json").exists())


class TestTargets(unittest.TestCase):

    def test_requires_a_cut_of_the_grid(self):
        with self.assertRaises(ValueError):
            invalidation_targets("configs/sweep/e2.yaml", [])

    def test_fixmatch_cells_of_e2_and_e2_cv(self):
        e2 = invalidation_targets("configs/sweep/e2.yaml", ["axes.method=[fixmatch]"])
        self.assertEqual(e2["family"], "e2")
        self.assertEqual(e2["run_ids"], [BAD])
        self.assertEqual(e2["run_names"][BAD], f"e2-{BAD}")
        cv = invalidation_targets("configs/sweep/e2-cv.yaml", ["axes.method=[fixmatch]"])
        self.assertEqual(cv["family"], "e2-cv")
        self.assertEqual(cv["run_ids"], [f"deeplabv3plus-fixmatch-global-f{k}-s42" for k in range(5)])
        self.assertTrue(all(name.startswith("e2-cv-") for name in cv["run_names"].values()))

    def test_archive_label(self):
        self.assertEqual(archive_label("e2", "fixmatch-bn"), "e2-fixmatch-bn")
        self.assertEqual(archive_label("e2-cv", "e2-cv-fixmatch-bn"), "e2-cv-fixmatch-bn")
        with self.assertRaises(ValueError):
            archive_label("e2", "../x")


if __name__ == "__main__":
    unittest.main()
