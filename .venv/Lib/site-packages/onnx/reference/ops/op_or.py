# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinary


class Or(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.logical_or(x, y),)
