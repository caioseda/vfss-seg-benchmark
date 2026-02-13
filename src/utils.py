import importlib
from pathlib import Path
import torch
from typing import Any, Dict


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", dict())
    obj = get_obj_from_str(config["target"])
    return obj(**params)

def get_obj_from_str(string, reload=False):
    module_name, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_name)
        importlib.reload(module_imp)
    module_imp = importlib.import_module(module_name, package=None)
    return getattr(module_imp, cls)

def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path

def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().mean().item()
    return value

def _print_metric_block(title: str, metrics: Dict[str, Any]) -> None:
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