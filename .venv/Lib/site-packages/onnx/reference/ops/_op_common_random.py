# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


class _CommonRandom(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if hasattr(self, "shape") and len(self.shape) == 0:  # type: ignore
            raise ValueError(  # pragma: no cover
                f"shape cannot be empty for operator {self.__class__.__name__}."
            )

    @staticmethod
    def numpy_type(dtype):  # type: ignore
        return tensor_dtype_to_np_dtype(dtype)

    @staticmethod
    def _dtype(*data, dtype=None, dtype_first=False):  # type: ignore
        numpy_type = _CommonRandom.numpy_type(dtype)
        if dtype_first and numpy_type is not None:
            if dtype != 0:  # type: ignore
                return numpy_type
            if data:
                return data[0].dtype
            raise RuntimeError(
                f"dtype cannot be None for a random operator {_CommonRandom.__name__!r}, "
                f"numpy_type={numpy_type}, len(data)={len(data)}."
            )
        res = None
        if not data:
            res = numpy_type
        elif numpy_type is not None:
            res = numpy_type
        elif hasattr(data[0], "dtype"):
            res = data[0].dtype
        if res is None:
            raise RuntimeError(
                f"dtype cannot be None, numpy_type={numpy_type}, type(data[0])={type(data[0])}."
            )
        return res

    @staticmethod
    def _get_state(seed):  # type: ignore
        if seed is None or np.isnan(seed):  # type: ignore
            state = np.random.RandomState()
        else:
            state = np.random.RandomState(seed=int(seed))  # type: ignore
        return state
