# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.helper import np_dtype_to_tensor_dtype
from onnx.reference.ops._op_common_random import _CommonRandom


class Bernoulli(_CommonRandom):
    def _run(self, x, dtype=None, seed=None):  # type: ignore
        if dtype is None:
            dtype = np_dtype_to_tensor_dtype(x.dtype)
        dtype = self._dtype(x, dtype=dtype, dtype_first=True)
        state = self._get_state(seed)
        res = state.binomial(1, p=x).astype(dtype)
        return (res.astype(dtype),)
