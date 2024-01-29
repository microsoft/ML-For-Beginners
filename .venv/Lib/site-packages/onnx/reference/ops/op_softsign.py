# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softsign(OpRunUnaryNum):
    def _run(self, X):  # type: ignore
        tmp = np.abs(X)
        tmp += 1
        np.divide(X, tmp, out=tmp)
        return (tmp,)
