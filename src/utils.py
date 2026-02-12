import importlib


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
