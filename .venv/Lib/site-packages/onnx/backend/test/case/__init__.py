# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Dict, List, Tuple

from onnx.backend.test.case.base import Snippets
from onnx.backend.test.case.utils import import_recursive


def collect_snippets() -> Dict[str, List[Tuple[str, str]]]:
    import_recursive(sys.modules[__name__])
    return Snippets
