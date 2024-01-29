# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinaryNum


def numpy_matmul(a, b):  # type: ignore
    """
    Implements a matmul product. See :func:`np.matmul`.
    Handles sparse matrices.
    """
    try:
        if len(a.shape) <= 2 and len(b.shape) <= 2:
            return np.dot(a, b)
        return np.matmul(a, b)
    except ValueError as e:
        raise ValueError(f"Unable to multiply shapes {a.shape!r}, {b.shape!r}.") from e


class MatMul(OpRunBinaryNum):
    def _run(self, a, b):  # type: ignore
        return (numpy_matmul(a, b),)
