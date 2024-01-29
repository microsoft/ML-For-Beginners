# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _global_average_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    y = np.average(x, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y  # type: ignore


class GlobalAveragePool(OpRun):
    def _run(self, x):  # type: ignore
        return (_global_average_pool(x).astype(x.dtype),)
