# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunUnary


class BitwiseNot(OpRunUnary):
    def _run(self, X):
        return (np.bitwise_not(X),)
