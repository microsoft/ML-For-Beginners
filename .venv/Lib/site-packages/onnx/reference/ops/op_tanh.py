# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Tanh(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.tanh(x),)
