# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceSum_1(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None):  # type: ignore
        axes = tuple(axes) if axes is not None else None
        res = np.sum(x, axis=axes, keepdims=keepdims, dtype=x.dtype)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceSum_13(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None, noop_with_empty_axes=None):  # type: ignore
        if (axes is None or axes.shape == (0,)) and noop_with_empty_axes:
            return (x,)
        axes = self.handle_axes(axes)
        try:
            res = np.sum(x, axis=axes, keepdims=keepdims, dtype=x.dtype)
            if keepdims == 0 and not isinstance(res, np.ndarray):
                # The runtime must return a numpy array of a single float.
                res = np.array(res)
            return (res,)  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {x.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
