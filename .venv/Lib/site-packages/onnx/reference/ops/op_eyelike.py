# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_pb import TensorProto
from onnx.reference.op_run import OpRun


class EyeLike(OpRun):
    def _run(self, data, *args, dtype=None, k=None):
        if dtype is None:
            if data is None:
                _dtype = np.float32
            else:
                _dtype = data.dtype
        elif dtype == TensorProto.STRING:
            _dtype = np.str_  # type: ignore[assignment]
        else:
            _dtype = tensor_dtype_to_np_dtype(dtype)  # type: ignore[assignment]
        shape = data.shape
        if len(shape) == 1:
            sh = (shape[0], shape[0])
        elif len(shape) == 2:
            sh = shape
        else:
            raise RuntimeError(f"EyeLike only accept 1D or 2D tensors not {shape!r}.")
        return (np.eye(*sh, k=k, dtype=_dtype),)  # type: ignore
