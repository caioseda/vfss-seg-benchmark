import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from omegaconf import ListConfig, OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import get_cli_args
from src.experiment import Experiment
from src.utils import _resolve_config_path, capture_console_output, _flatten_result_for_csv, _print_metric_block

class Benchmark:
    '''Class to run a benchmark of multiple experiments defined in a YAML config.'''
    def __init__(self, benchmark_cfg_path: Path):
        self.benchmark_cfg_path = Path(benchmark_cfg_path)
        self.config = self._load_config(benchmark_cfg_path)
        self.experiment_list = self._load_experiment_list(self.config)
        self.benchmark_name = self.config.get("name", self.config.get("benchmark_name", self.benchmark_cfg_path.stem))

    def setup_benchmark_run(self) -> None:
        # Set up benchmark directories
        self._timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        self._benchmark_root = (PROJECT_ROOT / "logs" / "benchmarks" / self.benchmark_name).resolve()
        self._benchmark_root.mkdir(parents=True, exist_ok=True)
        
        self._benchmark_id = self._get_benchmark_id(self._benchmark_root)

        self.benchmark_run_dir = self._benchmark_root / f"{self._benchmark_id:03d}-{self._timestamp}"
        self.benchmark_run_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_console_log = self.benchmark_run_dir / "console.log"

    def run(self, fast_dev_run: bool = False) -> None:
        '''Run the benchmark'''
        self.setup_benchmark_run()
        with capture_console_output(self.benchmark_console_log):
            print(f"Benchmark: {self.benchmark_name}")
            print(f"Benchmark config: {self.benchmark_cfg_path}")
            print(f"Benchmark run dir: {self.benchmark_run_dir}")
            print(f"Benchmark console log: {self.benchmark_console_log}")
            print(f"Total experiments: {len(self.experiment_list)}")

            benchmark_results = []
            with tqdm(
                total=len(self.experiment_list),
                desc="Benchmark Progress",
                unit="exp",
                dynamic_ncols=True,
                leave=True,
            ) as benchmark_progress:
                
                for idx, experiment_config_path in enumerate(self.experiment_list, start=1):
                    print(f"\n[{idx}/{len(self.experiment_list)}] Running: {experiment_config_path}")
                    
                    experiment_id = f"{idx:02d}"
                    experiment = Experiment(
                        config_path=experiment_config_path, 
                        benchmark_dir=self.benchmark_run_dir,
                        experiment_id=experiment_id
                    )
                    result = experiment.run(fast_dev_run=fast_dev_run)
                    # result = run_experiment(
                    #     exp_path,
                    #     run_suffix=suffix,
                    #     benchmark_dir=self.benchmark_run_dir,
                    #     experiment_id=experiment_id,
                    #     fast_dev_run=fast_dev_run,
                    # )
                    benchmark_results.append(result)
                    benchmark_progress.update(1)

            self.print_results(benchmark_results)
            self.save_results(benchmark_results)
            return benchmark_results
        
    def save_results(self, benchmark_results: List[Dict], csv_path: str = None) -> None:
        output_csv = csv_path or (self.benchmark_run_dir / "results.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        rows = [_flatten_result_for_csv(result, idx + 1) for idx, result in enumerate(benchmark_results)]
        fieldnames = sorted({key for row in rows for key in row.keys()})

        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def print_results(self, benchmark_results: List[Dict]) -> None:
        print("\nBenchmark Results:")
        for idx, result in enumerate(benchmark_results, start=1):
            exp_name = result.get("experiment_name", f"Experiment {idx}")
            print(f"\n[{idx}] {exp_name}")
            fit_metrics = result.get("fit_metrics", {})
            test_metrics = result.get("test_metrics", {})
            _print_metric_block("Fit Metrics:", fit_metrics)
            _print_metric_block("Test Metrics:", test_metrics)

    def _load_config(self, benchmark_cfg_path: Path) -> OmegaConf:
        '''Load benchmark config from YAML file'''
        benchmark_cfg_path = _resolve_config_path(benchmark_cfg_path)
        config = OmegaConf.load(str(benchmark_cfg_path))
        
        # Assert required fields
        _required_fields = ['experiments', 'name', 'type']
        for field in _required_fields:
            if field not in config:
                raise ValueError(f"Missing required field `{field}` in {benchmark_cfg_path}")
            
        assert config.type == "benchmark", f"Config type must be `benchmark`, got `{config.type}` in {benchmark_cfg_path}"
        return config


    def _load_experiment_list(self, benchmark_config: OmegaConf) -> List[str]:
        '''Load experiment list from benchmark config file'''
        experiments = benchmark_config.get("experiments")
        if not isinstance(experiments, (list, ListConfig)) or len(experiments) == 0:
            raise ValueError(
                f"`experiments` must be a non-empty list in {str(self.benchmark_cfg_path)}"
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

    def _get_benchmark_id(self, benchmark_root: Path) -> int:
        '''Get benchmark id for current run. if already exists benchmarks runs in the same benchmark_name, it will increment the id'''
        max_id = 0
        pattern = re.compile(r"^(\d+)-")
        for child in benchmark_root.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                potential_max = int(match.group(1))
                max_id = max(max_id, potential_max)
        return max_id + 1
    

if __name__ == "__main__":
    args = get_cli_args(
        argparse_description="Run a benchmark of multiple experiments defined in a YAML config.",
        default_config="configs/benchmark/vfss-transunet.yaml",
    )
    benchmark = Benchmark(args.config)
    benchmark.run(fast_dev_run=args.fast_dev_run)
    
