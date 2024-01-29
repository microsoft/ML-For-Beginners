# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):
    def _run(self, X):  # type: ignore
        tmp = np.exp(X).astype(X.dtype)
        tmp += 1
        np.log(tmp, out=tmp)
        return (tmp,)
