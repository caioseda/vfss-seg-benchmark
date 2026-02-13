import argparse
import sys
from pathlib import Path
from typing import List

from omegaconf import ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import run_experiment


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments from a benchmark YAML."
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark/default.yaml",
        help="Path to benchmark YAML file.",
    )
    return parser.parse_args()


def _resolve_benchmark_config(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config file not found: {path}")
    return path


def _load_experiment_list(benchmark_cfg_path: Path) -> List[str]:
    cfg = OmegaConf.load(str(benchmark_cfg_path))
    experiments = cfg.get("experiments")
    if not isinstance(experiments, (list, ListConfig)) or len(experiments) == 0:
        raise ValueError(
            f"`experiments` must be a non-empty list in {benchmark_cfg_path}"
        )

    resolved = []
    for item in experiments:
        exp_path = Path(str(item)).expanduser()
        if not exp_path.is_absolute():
            exp_path = (benchmark_cfg_path.parent / exp_path).resolve()
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        resolved.append(str(exp_path))
    return resolved


if __name__ == "__main__":
    args = _parse_args()
    benchmark_cfg_path = _resolve_benchmark_config(args.benchmark_config)
    benchmark_cfg = OmegaConf.load(str(benchmark_cfg_path))
    benchmark_name = benchmark_cfg.get("benchmark_name", benchmark_cfg_path.stem)

    experiment_paths = _load_experiment_list(benchmark_cfg_path)
    print(f"Benchmark: {benchmark_name}")
    print(f"Benchmark config: {benchmark_cfg_path}")
    print(f"Total experiments: {len(experiment_paths)}")

    results = []
    for idx, exp_path in enumerate(experiment_paths, start=1):
        suffix = f"{benchmark_name}-{idx:02d}"
        print(f"\n[{idx}/{len(experiment_paths)}] Running: {exp_path}")
        result = run_experiment(exp_path, run_suffix=suffix)
        results.append(result)

    print("\nBenchmark summary")
    for result in results:
        print(
            f"- {result['experiment_name']} | {result['run_version']} | "
            f"{Path(result['config_path']).name}"
        )
