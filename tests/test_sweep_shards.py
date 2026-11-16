'''
Divisao de um sweep em shards (`--shard K/N`) e a consolidacao deles (`merge_shards`).

O que precisa valer, porque e o que torna a divisao segura para a ciencia e para o disco:
  - os shards sao disjuntos e a uniao e a grade -- nenhuma celula treinada duas vezes, nenhuma esquecida;
  - todas as celulas de uma unidade de pareamento (os metodos de um fold/seed) caem no MESMO shard, logo
    na mesma GPU: a diferenca pareada nao absorve o efeito de hardware;
  - sem `--shard`, nenhum caminho muda;
  - um shard nunca escreve nos arquivos canonicos da familia, e o merge e idempotente.

Roda com:  python -m unittest tests.test_sweep_shards -v
'''
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sweep import Sweep, merge_shards, parse_shard, shard_cells
from src.sweeps.base import SweepPlan

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToyPlan(SweepPlan):
    def run_id(self, cell):
        return "-".join(f"{key}{value}" for key, value in cell.items())

    def overrides(self, cell):
        return {}


class StatefulToyPlan(ToyPlan):
    shardable = False


TOY_SWEEP = '''
type: sweep
name: toy
family: toy
plan: {plan}
axes:
  seed: [1, 2]
  fold: [0, 1, 2, 3, 4]
  method: [sup, mt, fm]
{extra}
'''


class ShardTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.logs = self.root / "logs"

    def tearDown(self):
        self._tmp.cleanup()

    def sweep(self, plan="tests.test_sweep_shards.ToyPlan", extra="", **kwargs) -> Sweep:
        config = self.root / "toy.yaml"
        config.write_text(TOY_SWEEP.format(plan=plan, extra=extra))
        return Sweep(config, log_dir=self.logs, **kwargs)


class ParseShard(unittest.TestCase):
    def test_one_indexed(self):
        self.assertEqual(parse_shard("1/2"), (1, 2))
        self.assertEqual(parse_shard(" 2 / 2 "), (2, 2))
        self.assertEqual(parse_shard("1/1"), (1, 1))
        self.assertIsNone(parse_shard(None))

    def test_rejects_zero_and_out_of_range(self):
        for spec in ["0/2", "3/2", "1/0", "a/b", "1", "-1/2"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_shard(spec)


class ShardPartition(ShardTestCase):
    def test_disjoint_and_complete(self):
        grid = {self.sweep().plan.run_id(c) for c in self.sweep().cells()}
        for total in (1, 2, 3, 4, 7, 10, 11):
            parts = [{s.plan.run_id(c) for c in s.cells()}
                     for s in (self.sweep(shard=f"{k}/{total}", shard_tag=f"h{k}") for k in range(1, total + 1))]
            with self.subTest(total=total):
                self.assertEqual(set().union(*parts), grid)
                self.assertEqual(sum(map(len, parts)), len(grid))

    def test_pairing_unit_never_split(self):
        for k in (1, 2, 3):
            sweep = self.sweep(shard=f"{k}/3", shard_tag=f"h{k}")
            units = {}
            for cell in sweep.cells():
                units.setdefault((cell["seed"], cell["fold"]), set()).add(cell["method"])
            with self.subTest(shard=k):
                self.assertTrue(units)
                for methods in units.values():
                    self.assertEqual(methods, {"sup", "mt", "fm"})

    def test_compared_axes_override_from_yaml(self):
        sweep = self.sweep(extra="compared_axes: [method, fold]", shard="1/2", shard_tag="a")
        # Unidade = seed: o shard 1 fica com a seed 1 inteira.
        self.assertEqual({c["seed"] for c in sweep.cells()}, {1})
        self.assertEqual(len(sweep.cells()), 15)

    def test_grid_order_preserved_inside_shard(self):
        sweep = self.sweep(shard="2/2", shard_tag="b")
        full = sweep.plan.cells()
        positions = [full.index(c) for c in sweep.cells()]
        self.assertEqual(positions, sorted(positions))

    def test_non_shardable_plan_refuses(self):
        with self.assertRaises(ValueError):
            self.sweep(plan="tests.test_sweep_shards.StatefulToyPlan", shard="1/2", shard_tag="a")
        self.sweep(plan="tests.test_sweep_shards.StatefulToyPlan")   # sem shard continua valendo


class ShardPaths(ShardTestCase):
    def test_unsharded_paths_unchanged(self):
        sweep = self.sweep()
        family = self.logs.resolve() / "toy"
        self.assertEqual(sweep.results_csv, family / "results_per_sample.csv")
        self.assertEqual(sweep.state_path, family / "_sweep" / "state.json")
        self.assertEqual(sweep.plan.sweep_dir, family / "_sweep")
        self.assertEqual(len(sweep.cells()), 30)

    def test_sharded_paths_isolated(self):
        sweep = self.sweep(shard="1/2", shard_tag="dl05")
        shard_dir = self.logs.resolve() / "toy" / "_sweep" / "dl05"
        self.assertEqual(sweep.results_csv, shard_dir / "results_per_sample.csv")
        self.assertEqual(sweep.state_path, shard_dir / "state.json")
        self.assertEqual(sweep.plan.sweep_dir, shard_dir)
        self.assertEqual(sweep.plan.tables_dir, shard_dir)

    def test_bad_tag_rejected(self):
        for tag in ["../x", "a/b", "configs", ""]:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.sweep(shard="1/2", shard_tag=tag or " ")


def _write_shard(family: Path, tag: str, run_ids, n_rows: int = 2, status: str = "done"):
    shard = family / "_sweep" / tag
    shard.mkdir(parents=True, exist_ok=True)
    rows = [{"run_id": r, "dice": 0.5, "output_dir": str(family / "runs" / r / "uid")}
            for r in run_ids for _ in range(n_rows)]
    pd.DataFrame(rows).to_csv(shard / "results_per_sample.csv", index=False)
    pd.DataFrame([{"run_id": r, "fit_wall_s": 1.0} for r in run_ids]).to_csv(
        shard / "results_resources.csv", index=False)
    (shard / "state.json").write_text(json.dumps({"cells": {
        r: {"status": status, "best_checkpoint_path": f"/ck/{r}.ckpt", "updated_at": "2026-11-16T10:00:00"}
        for r in run_ids}}))
    for r in run_ids:
        (family / "runs" / r / "uid").mkdir(parents=True, exist_ok=True)


class CompletedAndMerge(ShardTestCase):
    def test_completed_sees_shard_csvs(self):
        sweep = self.sweep()
        cells = sweep.cells()
        _write_shard(sweep.family_dir, "dl01", [sweep.plan.run_id(cells[0])])
        _write_shard(sweep.family_dir, "dl05", [sweep.plan.run_id(cells[1])])
        self.assertEqual(sweep.completed_run_ids(), {sweep.plan.run_id(c) for c in cells[:2]})
        summary = sweep.plan_summary()
        self.assertIn("host", summary.columns)
        self.assertEqual(set(summary.host[summary.status.str.startswith("done")]), {"dl01", "dl05"})
        # Um shard tambem ve o que outro shard ja concluiu (relancamento apos um merge parcial).
        other = self.sweep(shard="1/1", shard_tag="delaunay")
        self.assertEqual(len(other.completed_run_ids()), 2)

    def test_merge_consolidates_and_is_idempotent(self):
        family = self.logs / "toy"
        family.mkdir(parents=True)
        pd.DataFrame([{"run_id": "a", "dice": 0.9, "output_dir": ""}]).to_csv(
            family / "results_per_sample.csv", index=False)
        _write_shard(family, "dl01", ["a", "b"])        # "a" ja esta na canonica: nao duplica
        _write_shard(family, "dl05", ["c", "b"])        # "b" em dois shards: avisado, entra uma vez

        report = merge_shards(family)
        merged = pd.read_csv(family / "results_per_sample.csv")
        self.assertEqual(sorted(merged.run_id.unique()), ["a", "b", "c"])
        self.assertEqual((merged.run_id == "a").sum(), 1)      # a canonica tem precedencia
        self.assertEqual((merged.run_id == "b").sum(), 2)      # so as linhas do primeiro shard
        self.assertIn("results_per_sample.csv", report["duplicates"])
        self.assertTrue((family / "results_resources.csv").exists())
        state = json.loads((family / "_sweep" / "state.json").read_text())
        self.assertEqual(state["cells"]["c"]["host"], "dl05")
        self.assertEqual(report["missing_output_dirs"], [])

        before = (family / "results_per_sample.csv").read_text()
        again = merge_shards(family)
        self.assertEqual((family / "results_per_sample.csv").read_text(), before)
        self.assertEqual(again["state_cells_added"], 0)
        self.assertTrue(all(t["rows_added"] == 0 for t in again["tables"].values()))

    def test_merge_dry_run_writes_nothing(self):
        family = self.logs / "toy"
        _write_shard(family, "dl01", ["a"])
        report = merge_shards(family, dry_run=True)
        self.assertEqual(report["tables"]["results_per_sample.csv"]["run_ids_added"], 1)
        self.assertFalse((family / "results_per_sample.csv").exists())
        self.assertFalse((family / "_sweep" / "state.json").exists())


class RealSweepsPairing(unittest.TestCase):
    '''As grades reais: E2 pareia por metodo, E1 por arquitetura, a busca do Optuna recusa shard.'''

    def test_e2_cv_keeps_each_fold_on_one_shard(self):
        config = PROJECT_ROOT / "configs" / "sweep" / "e2-cv.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}
            for k in (1, 2, 3):
                sweep = Sweep(config, log_dir=tmp, shard=f"{k}/3", shard_tag=f"h{k}")
                for cell in sweep.cells():
                    seen.setdefault(cell["fold"], set()).add(k)
            self.assertTrue(all(len(shards) == 1 for shards in seen.values()), seen)

    def test_e1_cv_pairs_architectures(self):
        sweep = Sweep(PROJECT_ROOT / "configs" / "sweep" / "e1-cv.yaml")
        plan = sweep.plan
        cells = plan.cells()
        keys = {plan.pairing_key(c) for c in cells}
        archs = set(plan.axes()["arch"])
        self.assertEqual(len(keys), len(cells) // len(archs))
        part = shard_cells(plan, cells, (1, 2))
        by_fold = {}
        for cell in part:
            by_fold.setdefault((cell["seed"], cell["fold"]), set()).add(cell["arch"])
        self.assertTrue(all(found == archs for found in by_fold.values()))

    def test_tuning_search_is_not_shardable(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            Sweep(PROJECT_ROOT / "configs" / "sweep" / "e1-tuning-search.yaml", log_dir=tmp,
                  shard="1/2", shard_tag="a")


if __name__ == "__main__":
    unittest.main()
