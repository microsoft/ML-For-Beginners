# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinaryComparison


class Greater(OpRunBinaryComparison):
    def _run(self, a, b):  # type: ignore
        return (np.greater(a, b),)
