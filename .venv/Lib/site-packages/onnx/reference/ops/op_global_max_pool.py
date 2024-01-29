# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _global_max_pool(x: np.ndarray) -> np.ndarray:
    spatial_shape = np.ndim(x) - 2
    y = x.max(axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = np.expand_dims(y, -1)
    return y  # type: ignore


class GlobalMaxPool(OpRun):
    def _run(self, x):  # type: ignore
        res = _global_max_pool(x)
        return (res,)
