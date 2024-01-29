# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.helper import np_dtype_to_tensor_dtype
from onnx.reference.ops._op_common_random import _CommonRandom


class RandomUniformLike(_CommonRandom):
    def _run(self, x, dtype=None, high=None, low=None, seed=None):  # type: ignore
        if dtype is None:
            dtype = np_dtype_to_tensor_dtype(x.dtype)
        dtype = self._dtype(x, dtype=dtype)
        state = self._get_state(seed)
        res = state.rand(*x.shape).astype(dtype)
        res *= high - low  # type: ignore
        res += low  # type: ignore
        return (res.astype(dtype),)
