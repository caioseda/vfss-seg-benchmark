import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from src.cli import get_cli_args
from src.experiment import Experiment
from src.benchmark import Benchmark
from src.utils import _resolve_config_path

def get_config_type(config_path: str) -> str:
    resolved_path = _resolve_config_path(config_path)
    if not resolved_path.is_file():
        raise ValueError(f"Config path {config_path} does not exist or is not a file.")
    
    config = OmegaConf.load(str(resolved_path))
    if "type" not in config:
        raise ValueError(f"Config file {config_path} must contain a 'type' field indicating 'experiment' or 'benchmark'.")
    
    config_type = config.type.lower()
    if config_type not in ["experiment", "benchmark"]:
        raise ValueError(f"Config type must be either 'experiment' or 'benchmark', got '{config.type}' in {config}")
    
    return config_type

if __name__ == "__main__":
    args = get_cli_args(default_config="configs/experiment/unet/vfss-inca-unet.yaml")
    config_type = get_config_type(args.config)

    if config_type == "experiment":
        experiment = Experiment(config_path=args.config)
        experiment.run(fast_dev_run=args.fast_dev_run)
    elif config_type == "benchmark":
        benchmark = Benchmark(args.config)
        benchmark.run(fast_dev_run=args.fast_dev_run)
    else:
        raise ValueError(f"Unsupported config type: {config_type}")

    
