# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinary


class BitwiseXor(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.bitwise_xor(x, y),)
