# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops._op_common_random import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(self, dtype=None, high=None, low=None, seed=None, shape=None):  # type: ignore
        dtype = self._dtype(dtype=dtype)
        state = self._get_state(seed)
        res = state.rand(*shape).astype(dtype)  # type: ignore
        res *= high - low  # type: ignore
        res += low  # type: ignore
        return (res.astype(dtype),)
