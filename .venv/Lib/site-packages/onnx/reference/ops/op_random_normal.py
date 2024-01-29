# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops._op_common_random import _CommonRandom


class RandomNormal(_CommonRandom):
    def _run(self, dtype=None, mean=None, scale=None, seed=None, shape=None):  # type: ignore
        state = self._get_state(seed)
        numpy_type = self.numpy_type(dtype)
        res = state.randn(*shape).astype(numpy_type)  # type: ignore
        res *= scale  # type: ignore
        res += mean  # type: ignore
        return (res.astype(numpy_type),)
