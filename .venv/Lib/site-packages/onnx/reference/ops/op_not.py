# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunUnary


class Not(OpRunUnary):
    def _run(self, x):  # type: ignore
        return (np.logical_not(x),)
