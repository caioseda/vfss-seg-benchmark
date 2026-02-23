import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from omegaconf import ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import run_experiment
from src.cli import get_cli_args
from src.utils import _resolve_config_path


def _load_experiment_list(benchmark_cfg_path: Path) -> List[str]:
    benchmark_cfg_path = benchmark_cfg_path.expanduser().resolve()
    cfg = OmegaConf.load(str(benchmark_cfg_path))
    experiments = cfg.get("experiments")
    if not isinstance(experiments, (list, ListConfig)) or len(experiments) == 0:
        raise ValueError(
            f"`experiments` must be a non-empty list in {benchmark_cfg_path}"
        )

    valid_experiment_cfg = []
    for item in experiments:
        exp_path = Path(str(item)).expanduser()
        if not exp_path.is_absolute():
            exp_path = (PROJECT_ROOT / exp_path).resolve()
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        valid_experiment_cfg.append(str(exp_path))
    return valid_experiment_cfg


def _next_benchmark_id(benchmark_root: Path) -> int:
    benchmark_root.mkdir(parents=True, exist_ok=True)
    max_id = 0
    pattern = re.compile(r"^(\d+)-")
    for child in benchmark_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _flatten_result_for_csv(result: Dict, index: int) -> Dict[str, object]:
    row: Dict[str, object] = {
        "experiment_index": index,
        "experiment_name": result.get("experiment_name", ""),
        "config_path": result.get("config_path", ""),
        "output_dir": result.get("output_dir", ""),
        "run_version": result.get("run_version", ""),
    }
    for k, v in result.get("fit_metrics", {}).items():
        row[f"fit::{k}"] = v
    for k, v in result.get("test_metrics", {}).items():
        row[f"test::{k}"] = v
    return row


def _save_benchmark_csv(results: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_result_for_csv(result, idx + 1) for idx, result in enumerate(results)]
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = get_cli_args(
        argparse_description="Run a benchmark of multiple experiments defined in a YAML config.",
        default_config="configs/benchmark/vfss-transunet.yaml",
    )
    benchmark_cfg_path = _resolve_config_path(args.config)
    benchmark_cfg = OmegaConf.load(str(benchmark_cfg_path))
    benchmark_name = benchmark_cfg.get("benchmark_name", benchmark_cfg_path.stem)

    experiment_paths = _load_experiment_list(benchmark_cfg_path)

    benchmark_root = (PROJECT_ROOT / "logs" / "benchmarks" / benchmark_name).resolve()
    benchmark_id = _next_benchmark_id(benchmark_root)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    benchmark_run_dir = benchmark_root / f"{benchmark_id:03d}-{timestamp}"
    benchmark_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark: {benchmark_name}")
    print(f"Benchmark config: {benchmark_cfg_path}")
    print(f"Benchmark run dir: {benchmark_run_dir}")
    print(f"Total experiments: {len(experiment_paths)}")

    results = []
    for idx, exp_path in enumerate(experiment_paths, start=1):
        exp_stem = Path(exp_path).stem
        experiment_subdir = f"{idx:02d}-{exp_stem}"
        suffix = f"{benchmark_name}-{idx:02d}"
        print(f"\n[{idx}/{len(experiment_paths)}] Running: {exp_path}")
        result = run_experiment(
            exp_path,
            run_suffix=suffix,
            logs_root=benchmark_run_dir,
            experiment_subdir=experiment_subdir,
        )
        results.append(result)

    results_csv = benchmark_run_dir / "results.csv"
    _save_benchmark_csv(results, results_csv)

    print("\nBenchmark summary")
    for result in results:
        print(
            f"- {result['experiment_name']} | {result['run_version']} | "
            f"{Path(result['config_path']).name} | {result['output_dir']}"
        )
    print(f"CSV summary: {results_csv}")
