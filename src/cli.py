import argparse
from pathlib import Path

from omegaconf import OmegaConf

def _get_arg_parser(argparse_description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=argparse_description)
    return parser

def _add_arguments(parser: argparse.ArgumentParser, default_config: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the YAML configuration file for the experiment.",
    )
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        const=True,
        default=False,
        type=int,
        help=(
            "Enable PyTorch Lightning fast_dev_run. Use flag alone for True "
            "or pass an integer number of batches (e.g. --fast-dev-run 5)."
        ),
    )
    return parser

def get_cli_args(argparse_description='', default_config: str=''):
    parser = _get_arg_parser(argparse_description=argparse_description)
    parser = _add_arguments(parser, default_config=default_config)
    args = parser.parse_args()
    return args
