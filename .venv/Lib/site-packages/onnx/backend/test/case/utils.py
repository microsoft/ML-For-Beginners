# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
from types import ModuleType
from typing import List, Optional

import numpy as np

from onnx import ONNX_ML

all_numeric_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]


def import_recursive(package: ModuleType) -> None:
    """
    Takes a package and imports all modules underneath it
    """
    pkg_dir: Optional[List[str]] = None
    pkg_dir = package.__path__  # type: ignore
    module_location = package.__name__
    for _module_loader, name, ispkg in pkgutil.iter_modules(pkg_dir):
        module_name = f"{module_location}.{name}"  # Module/package
        if not ONNX_ML and module_name.startswith(
            "onnx.backend.test.case.node.ai_onnx_ml"
        ):
            continue

        module = importlib.import_module(module_name)
        if ispkg:
            import_recursive(module)
