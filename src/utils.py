import importlib
import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO, Union

import torch
import os


def instantiate_from_config(config: Dict) -> object:
    ''' 
    Instantiate an object from a config dict with `target` and optional `params`.
    Helper to be able to specify objects in config files.
    '''
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", dict())
    obj = get_obj_from_str(config["target"])
    return obj(**params)

def get_obj_from_str(string, reload=False) -> object:
    '''Instantiate an object from a string like "module.submodule.ClassName".'''
    module_name, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_name)
        importlib.reload(module_imp)
    module_imp = importlib.import_module(module_name, package=None)
    return getattr(module_imp, cls)

def _resolve_config_path(config_path: str) -> Path:
    '''Expands, resolves and validates that the config path exists'''
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path

def _to_scalar(value: Any, operation: str = "mean") -> Any:
    '''
    Converts tensors to scalars for cleaner logging output. 
    If the tensor has more than one element, it applies the specified aggregation to reduce it to a single scalar value (default is "mean").
    Helper for logging.
    '''
    operations = {
        "mean": lambda x: x.mean().item(),
        "sum": lambda x: x.sum().item(),
        "max": lambda x: x.max().item(),
        "min": lambda x: x.min().item(),
    }

    if operation not in operations:
            raise ValueError(f"Unsupported operation '{operation}' for tensor reduction. Supported operations: {list(operations.keys())}")

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        
        operation_func = operations[operation]
        return operation_func(value.detach().cpu())
    
    return value

def _print_metric_block(title: str, metrics: Dict[str, Any]) -> None:
    '''Helper to print metrics with title.'''
    print(f"\n{title}")
    if not metrics:
        print("  (no metrics found)")
        return

    for name in sorted(metrics):
        value = _to_scalar(metrics[name])
        if isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")

def _flatten_result_for_csv(result: Dict, index: int) -> Dict[str, object]:
    '''Flattens the result dict for easier CSV writing.'''
    row = {
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

def _resolve_path(path: Union[str,  Path]) -> Path:
    '''Resolve a potentially relative path using the current working directory.'''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    return Path(path).expanduser().resolve()

class _TeeStream:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


_ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1B
    (?:
        \[[0-?]*[ -/]*[@-~]      # CSI sequences, including colors and cursor movement.
        |\][^\x07]*(?:\x07|\x1B\\) # OSC sequences.
        |[@-Z\\-_]                # Other single-character escape sequences.
    )
    """,
    re.VERBOSE,
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _ConsoleLogStream:
    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, data: str) -> int:
        clean_data = _ANSI_ESCAPE_RE.sub("", data)
        clean_data = clean_data.replace("\r", "\n")
        clean_data = _CONTROL_CHARS_RE.sub("", clean_data)
        self._stream.write(clean_data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def capture_console_output(log_path: Union[str, Path]) -> Iterator[None]:
    '''Context manager to capture console output to a log file while still printing to the console.'''
    resolved_log_path = Path(log_path).expanduser().resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log_path.open("a", encoding="utf-8") as log_file:
        clean_log_file = _ConsoleLogStream(log_file)
        tee_stdout = _TeeStream(sys.stdout, clean_log_file)
        tee_stderr = _TeeStream(sys.stderr, clean_log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            yield
