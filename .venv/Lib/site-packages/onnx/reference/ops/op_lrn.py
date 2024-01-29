# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import math

import numpy as np

from onnx.reference.op_run import OpRun


class LRN(OpRun):
    def _run(self, x, alpha=None, beta=None, bias=None, size=None):  # type: ignore
        if len(x.shape) != 4:
            raise RuntimeError(
                f"LRN only applies on 4D tensors but shape is {x.shape!r}."
            )
        square_sum = np.zeros(x.shape).astype(x.dtype)
        minc = x.shape[1]
        c1 = int(math.floor((size - 1) / 2))
        c2 = int(math.ceil((size - 1) / 2)) + 1
        for c in range(x.shape[0]):
            begin = max(0, c - c1)
            end = min(minc, c + c2)
            square_sum[:, c, :, :] = np.sum(x[:, begin:end, :, :] ** 2, axis=1)
        y = x / ((bias + (alpha / size) * square_sum) ** beta)
        return (y.astype(x.dtype),)
