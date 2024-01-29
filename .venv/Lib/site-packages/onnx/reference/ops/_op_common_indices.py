# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np


def _get_indices(i, shape):  # type: ignore
    res = np.empty((len(shape),), dtype=np.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res


def _is_out(ind, shape):  # type: ignore
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False


def _get_index(indices, shape):  # type: ignore
    ind = 0
    mul = 1
    for pos, sh in zip(reversed(indices), reversed(shape)):
        ind += pos * mul
        mul *= sh
    return ind
