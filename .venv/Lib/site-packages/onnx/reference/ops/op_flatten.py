# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunUnary


class Flatten(OpRunUnary):
    def _run(self, x, axis=None):  # type: ignore
        i = axis or self.axis  # type: ignore
        shape = x.shape
        new_shape = (1, -1) if i == 0 else (np.prod(shape[:i]).astype(int), -1)
        return (x.reshape(new_shape),)  # type: ignore
