# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

import onnx


@dataclass
class TestCase:
    name: str
    model_name: str
    url: Optional[str]
    model_dir: Optional[str]
    model: Optional[onnx.ModelProto]
    data_sets: Optional[Sequence[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]]
    kind: str
    rtol: float
    atol: float
    # Tell PyTest this isn't a real test.
    __test__: bool = False
