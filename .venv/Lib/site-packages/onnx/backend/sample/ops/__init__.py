# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import pkgutil
import sys
from types import ModuleType
from typing import Dict


def collect_sample_implementations() -> Dict[str, str]:
    dict_: Dict[str, str] = {}
    _recursive_scan(sys.modules[__name__], dict_)
    return dict_


def _recursive_scan(package: ModuleType, dict_: Dict[str, str]) -> None:
    pkg_dir = package.__path__  # type: ignore
    module_location = package.__name__
    for _module_loader, name, ispkg in pkgutil.iter_modules(pkg_dir):  # type: ignore
        module_name = f"{module_location}.{name}"  # Module/package
        module = importlib.import_module(module_name)
        dict_[name] = inspect.getsource(module)
        if ispkg:
            _recursive_scan(module, dict_)
