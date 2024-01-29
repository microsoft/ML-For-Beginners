# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):
    def _run(self, x, alpha=None, beta=None):  # type: ignore
        alpha = alpha or self.alpha  # type: ignore
        beta = beta or self.beta  # type: ignore
        y = np.maximum(0, np.minimum(1, x * alpha + beta))
        return (y,)
