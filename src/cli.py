import argparse
from pathlib import Path

from omegaconf import OmegaConf


def get_arg_parser(default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to experiment YAML file.",
    )
    return parser

def get_cli_args(default_config: str = "configs/vfss-inca-unet.yaml"):
    parser = get_arg_parser(default_config=default_config)
    args = parser.parse_args()
    return args
