# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):  # type: ignore
        axis = axis or self.axis  # type: ignore
        p = p or self.p  # type: ignore
        norm = np.power(np.power(x, p).sum(axis=axis), 1.0 / p)  # type: ignore
        norm = np.expand_dims(norm, axis)  # type: ignore
        return ((x / norm).astype(x.dtype),)
