# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):  # type: ignore
        Y = Softmax._run(self, X)[0]
        np.log(Y, out=Y)
        return (Y,)
