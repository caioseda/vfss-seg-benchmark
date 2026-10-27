'''
Tracking ao vivo: o treino e o indexador concordam sobre a identidade do run, e o tracker nunca
derruba o treino.

Por que cada teste existe -- todos pegam falhas que "rodam normalmente" e estragam o registro:

  identidade   Se o `run_uid` que o treino grava difere do que o indexador calcula, o backfill nao
               reconhece o run logado ao vivo e o reindexa: o mesmo treino aparece duas vezes.
  retomada     Cada tentativa de um run retomado reabre o mesmo diretorio. Sem reaproveitar o run do
               MLflow, um unico treino vira varios.
  resiliencia  Uma excecao do backend (banco travado, disco cheio) levantada dentro de `log_metrics`
               interromperia um treino de horas por algo que nao e o experimento.
  publicado    O numero publicado so nasce no `evaluate()`, depois de os loggers fecharem; sem
               `attach_published_metrics` o run logado ao vivo ficaria sem o resultado da celula.

Todos usam um banco SQLite TEMPORARIO: nunca tocam o `logs/_tracking/mlflow.db` real.

Roda com:  python -m unittest tests.test_tracking -v
'''

import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytorch_lightning as pl

from src.models.ssl import SupervisedLitWrapper
from datetime import datetime

from src.tracking import (
    MetricPoint,
    attach_published_metrics,
    build_loggers,
    discover_runs,
    interpolate_timestamps,
    resolve_tracking_config,
    run_time_span,
    run_uid_for,
)
from tests.test_resume import ToyDataModule, build_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_indexer():
    spec = importlib.util.spec_from_file_location("index_runs", PROJECT_ROOT / "scripts" / "index_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackingCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.logs_root = Path(self._tmp.name) / "logs"
        self.run_dir = self.logs_root / "fam" / "runs" / "fam-cell-s42" / "uid-0001"
        self.run_dir.mkdir(parents=True)
        self.tracking_uri = f"sqlite:///{Path(self._tmp.name) / 'mlflow.db'}"
        self.config = {"mlflow": {"enabled": True, "tracking_uri": self.tracking_uri},
                       "aim": {"enabled": False}}

    def client(self):
        from mlflow.tracking import MlflowClient
        return MlflowClient(tracking_uri=self.tracking_uri)

    def fit(self, max_steps: int = 3, config=None):
        '''Um fit de verdade, com os loggers que o `Experiment` construiria.'''
        loggers = build_loggers(self.run_dir, self.logs_root, run_name="fam-cell-s42",
                                logging_cfg=config or self.config)
        for logger in loggers:
            logger.log_hyperparams({"trainer/params/max_steps": max_steps, "seed": 42})
        trainer = pl.Trainer(
            logger=loggers, max_steps=max_steps, max_epochs=-1, accelerator="cpu", devices=1,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=1,
        )
        trainer.fit(build_module(SupervisedLitWrapper), ToyDataModule(num_workers=0))
        return loggers

    def mlflow_runs(self, experiment_name="fam"):
        client = self.client()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        return client.search_runs([experiment.experiment_id], max_results=100)


class LiveRunIdentity(TrackingCase):

    def test_live_run_carries_the_identity_the_indexer_computes(self):
        self.fit()
        runs = self.mlflow_runs()
        self.assertEqual(len(runs), 1)
        tags = runs[0].data.tags
        self.assertEqual(tags["run_uid"], run_uid_for(self.run_dir, self.logs_root),
                         "o run_uid logado ao vivo difere do que o indexador calcula")
        self.assertEqual(tags["family"], "fam")
        self.assertEqual(tags["run_dir_relative"], "fam/runs/fam-cell-s42/uid-0001")
        self.assertTrue(runs[0].data.metrics, "nenhuma metrica chegou ao MLflow")

    def test_indexer_recognises_a_live_run_instead_of_duplicating_it(self):
        self.fit()
        indexer = load_indexer()
        records = discover_runs(self.logs_root)
        self.assertEqual(len(records), 1)

        counts = indexer.index_into_mlflow(records, self.tracking_uri, published={}, verbose=False)
        self.assertEqual(counts.get("indexado", 0), 0, "o indexador recriou um run que ja existia")
        self.assertEqual(counts["ja indexado"], 1)
        self.assertEqual(len(self.mlflow_runs()), 1)

    def test_no_artifacts_are_copied(self):
        self.fit()
        self.assertFalse(list(Path(self._tmp.name).rglob("*.ckpt")), "um checkpoint foi copiado")
        self.assertFalse((Path(os.getcwd()) / "mlruns").exists() and
                         any((Path(os.getcwd()) / "mlruns").rglob("*.ckpt")))


class ResumeContinuesTheSameRun(TrackingCase):

    def test_second_attempt_reuses_the_mlflow_run(self):
        self.fit(max_steps=2)
        first = self.mlflow_runs()
        self.fit(max_steps=4)        # mesma pasta => mesmo run_uid, como numa retomada
        second = self.mlflow_runs()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1,
                         "a retomada abriu um segundo run no MLflow para o mesmo treino")
        self.assertEqual(first[0].info.run_id, second[0].info.run_id)


class TrackerNeverKillsTraining(TrackingCase):

    def test_backend_failure_during_training_is_survivable(self):
        from mlflow.tracking import MlflowClient
        with mock.patch.object(MlflowClient, "log_batch", side_effect=RuntimeError("database is locked")):
            self.fit(max_steps=3)   # nao pode levantar
        self.assertTrue((self.run_dir / "csv" / "metrics.csv").is_file(),
                        "a fonte da verdade (CSV) tem de existir mesmo com o tracker falho")

    def test_failure_count_is_per_run_not_per_process(self):
        '''
        Um sweep roda varias celulas no mesmo processo. Com um contador de classe, os avisos da
        segunda celula em diante sumiriam depois das tres primeiras falhas somadas.
        '''
        from mlflow.tracking import MlflowClient
        with mock.patch.object(MlflowClient, "log_batch", side_effect=RuntimeError("boom")):
            first = self.fit(max_steps=3)
            self.run_dir = self.logs_root / "fam" / "runs" / "fam-cell-s42" / "uid-0002"
            self.run_dir.mkdir(parents=True)
            second = self.fit(max_steps=3)
        counts = [logger.failures for loggers in (first, second) for logger in loggers
                  if hasattr(logger, "failures")]
        self.assertEqual(len(counts), 2)
        self.assertTrue(all(count > 0 for count in counts), counts)

    def test_unbuildable_tracker_is_skipped_not_fatal(self):
        config = {"mlflow": {"enabled": True, "tracking_uri": "nao-e-um-backend://x"},
                  "aim": {"enabled": False}}
        loggers = build_loggers(self.run_dir, self.logs_root, run_name="r", logging_cfg=config)
        self.assertEqual([type(l).__name__ for l in loggers], ["TensorBoardLogger", "CSVLogger"])

    def test_disabled_tracker_adds_nothing(self):
        config = {"mlflow": {"enabled": False}, "aim": {"enabled": False}}
        loggers = build_loggers(self.run_dir, self.logs_root, run_name="r", logging_cfg=config)
        self.assertEqual(len(loggers), 2)


class PublishedNumber(TrackingCase):

    def write_results_csv(self, dice: float):
        path = self.logs_root / "fam" / "results_per_sample.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "output_dir", "dice"])
            writer.writeheader()
            writer.writerow({"run_id": "cell", "output_dir": str(self.run_dir), "dice": dice})
            writer.writerow({"run_id": "cell", "output_dir": str(self.run_dir), "dice": dice + 0.2})

    def test_published_number_is_attached_after_the_fit(self):
        self.fit()
        self.write_results_csv(0.5)
        attached = attach_published_metrics(self.config, self.run_dir, self.logs_root)
        self.assertGreater(attached, 0)
        metrics = self.mlflow_runs()[0].data.metrics
        self.assertAlmostEqual(metrics["published/dice"], 0.6, places=6)   # media das duas linhas

    def test_attaching_twice_does_not_duplicate(self):
        self.fit()
        self.write_results_csv(0.5)
        self.assertGreater(attach_published_metrics(self.config, self.run_dir, self.logs_root), 0)
        self.assertEqual(attach_published_metrics(self.config, self.run_dir, self.logs_root), 0)

    def test_indexer_tops_up_a_live_run_that_lacks_the_number(self):
        self.fit()
        self.write_results_csv(0.5)
        indexer = load_indexer()
        records = discover_runs(self.logs_root)
        published = indexer.published_metrics_by_run_dir(self.logs_root)
        counts = indexer.index_into_mlflow(records, self.tracking_uri, published, verbose=False)
        self.assertEqual(counts["numero publicado anexado"], 1)
        self.assertEqual(len(self.mlflow_runs()), 1)

    def test_replace_repairs_a_run_instead_of_skipping_it(self):
        self.fit()
        indexer = load_indexer()
        records = discover_runs(self.logs_root)
        counts = indexer.index_into_mlflow(records, self.tracking_uri, {}, verbose=False, replace=True)
        self.assertEqual(counts["substituido"], 1)
        self.assertEqual(counts["indexado"], 1)
        self.assertEqual(len(self.mlflow_runs()), 1, "o run antigo deveria estar apagado (soft-delete)")

    def test_disabled_mlflow_attaches_nothing(self):
        self.write_results_csv(0.5)
        self.assertEqual(attach_published_metrics({"mlflow": {"enabled": False}},
                                                  self.run_dir, self.logs_root), 0)


def local_ms(*parts) -> int:
    return int(datetime(*parts).timestamp() * 1000)


class WhenTheRunHappened(unittest.TestCase):
    '''
    O backfill cria o run no MLflow quando INDEXA. Sem os tempos reais a interface mostra
    "Created: 5 hours ago" e "Duration: 1.7s" para um treino de horas de semanas atras.
    '''

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.logs = Path(self._tmp.name) / "logs"

    def make_run(self, relative: str, *, end=None, run_json=None):
        run_dir = self.logs / relative
        (run_dir / "csv").mkdir(parents=True)
        metrics = run_dir / "csv" / "metrics.csv"
        metrics.write_text("step,val/dice\n0,0.1\n10,0.9\n")
        if end is not None:
            os.utime(metrics, (end, end))
        if run_json:
            (run_dir / "run.json").write_text(__import__("json").dumps(run_json))
        return discover_runs(self.logs)[0]

    def test_start_comes_from_the_directory_timestamp(self):
        end = datetime(2026, 9, 14, 18, 35, 0).timestamp()
        record = self.make_run("x1/runs/x1-cell/2026-09-14-17-46-29", end=end)
        start, finish = run_time_span(record)
        self.assertEqual(start, local_ms(2026, 9, 14, 17, 46, 29))
        self.assertEqual(finish, int(end * 1000))

    def test_benchmark_layout_uses_the_parent_timestamp(self):
        # logs/_archive/benchmarks/<nome>/<NNN-ts>/<NN-cfg>: o timestamp esta no PAI do run.
        record = self.make_run("_archive/benchmarks/b/006-2026-03-02-18-53-26/01-vfss-unet",
                               end=datetime(2026, 3, 2, 19, 0, 0).timestamp())
        self.assertEqual(run_time_span(record)[0], local_ms(2026, 3, 2, 18, 53, 26))

    def test_run_json_wins_over_the_directory_name_and_mtime(self):
        record = self.make_run(
            "x1/runs/x1-cell/2026-09-14-17-46-29", end=datetime(2026, 9, 14, 23, 0, 0).timestamp(),
            run_json={"started_at": "2026-09-14T17:50:00", "finished_at": "2026-09-14T18:20:00"})
        start, finish = run_time_span(record)
        self.assertEqual(start, local_ms(2026, 9, 14, 17, 50, 0))
        self.assertEqual(finish, local_ms(2026, 9, 14, 18, 20, 0))

    def test_end_is_never_before_start(self):
        record = self.make_run("x1/runs/x1-cell/2026-09-14-17-46-29",
                               end=datetime(2020, 1, 1).timestamp())   # relogio torto / arquivo copiado
        start, finish = run_time_span(record)
        self.assertEqual(finish, start)

    def test_no_timestamp_anywhere_gives_none_not_a_guess(self):
        record = self.make_run("x1/runs/x1-cell/no-timestamp-here")
        self.assertIsNone(run_time_span(record)[0])

    def test_interpolation_orders_points_inside_the_span(self):
        points = [MetricPoint("loss", 1.0, step) for step in (0, 5, 10)]
        stamps = interpolate_timestamps(points, 1_000_000, 2_000_000)
        self.assertEqual(stamps, [1_000_000, 1_500_000, 2_000_000])

    def test_a_real_wall_time_is_kept(self):
        points = [MetricPoint("loss", 1.0, 3, wall_time=1234.0)]
        self.assertEqual(interpolate_timestamps(points, 1_000_000, 2_000_000), [1_234_000])


class IndexerWritesRealTimes(TrackingCase):

    def test_backfilled_run_carries_when_it_happened(self):
        run_dir = self.logs_root / "fam" / "runs" / "cell" / "2026-09-14-17-46-29"
        (run_dir / "csv").mkdir(parents=True)
        metrics = run_dir / "csv" / "metrics.csv"
        metrics.write_text("step,val/dice\n0,0.1\n10,0.9\n")
        end = datetime(2026, 9, 14, 18, 35, 0).timestamp()
        os.utime(metrics, (end, end))

        indexer = load_indexer()
        indexer.index_into_mlflow(discover_runs(self.logs_root), self.tracking_uri, {}, verbose=False)

        run = self.mlflow_runs()[0]
        start_ms = local_ms(2026, 9, 14, 17, 46, 29)
        self.assertEqual(run.info.start_time, start_ms, "o run foi criado quando o indexador rodou")
        self.assertEqual(run.info.end_time, int(end * 1000))
        history = self.client().get_metric_history(run.info.run_id, "val/dice")
        self.assertTrue(all(start_ms <= h.timestamp <= int(end * 1000) for h in history),
                        [h.timestamp for h in history])
        self.assertNotIn(0, [h.timestamp for h in history], "timestamp 0 coloca o grafico em 1970")


class TrackingConfig(unittest.TestCase):

    def test_precedence_experiment_overrides_defaults(self):
        base = resolve_tracking_config(Path("/x/logs"))
        self.assertTrue(base["mlflow"]["enabled"])
        self.assertFalse(base["aim"]["enabled"], "o Aim nao pode vir ligado por padrao")
        self.assertEqual(base["mlflow"]["tracking_uri"], "sqlite:////x/logs/_tracking/mlflow.db")
        over = resolve_tracking_config(Path("/x/logs"), {"mlflow": {"enabled": False}})
        self.assertFalse(over["mlflow"]["enabled"])

    def test_env_switch_turns_everything_off(self):
        with mock.patch.dict(os.environ, {"VFSS_TRACKING": "off"}):
            config = resolve_tracking_config(Path("/x/logs"), {"mlflow": {"enabled": True}})
        self.assertFalse(config["mlflow"]["enabled"])
        self.assertFalse(config["aim"]["enabled"])

    def test_legacy_loggers_list_survives_the_merge(self):
        config = resolve_tracking_config(Path("/x/logs"),
                                         {"loggers": [{"target": "a.B", "params": {}}]})
        self.assertEqual(config["loggers"], [{"target": "a.B", "params": {}}])


if __name__ == "__main__":
    unittest.main()
