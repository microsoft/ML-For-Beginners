# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _argmin(data, axis=0, keepdims=True):  # type: ignore
    result = np.argmin(data, axis=axis)
    if keepdims and len(result.shape) < len(data.shape):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def _argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):  # type: ignore
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


class _ArgMin(OpRun):
    def _run(self, data, axis=None, keepdims=None):  # type: ignore
        return (_argmin(data, axis=axis, keepdims=keepdims),)


class ArgMin_1(_ArgMin):
    pass


class ArgMin_12(_ArgMin):
    def _run(self, data, axis=None, keepdims=None, select_last_index=None):  # type: ignore
        if select_last_index == 0:  # type: ignore
            return _ArgMin._run(self, data, axis=axis, keepdims=keepdims)
        return (
            _argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims),
        )
