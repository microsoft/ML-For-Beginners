# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


def _scatter_nd_impl(data, indices, updates, reduction=None):  # type: ignore
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if reduction == "add":
            output[indices[i]] += updates[i]
        elif reduction == "mul":
            output[indices[i]] *= updates[i]
        elif reduction == "max":
            output[indices[i]] = np.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[indices[i]] = np.minimum(output[indices[i]], updates[i])
        else:
            output[indices[i]] = updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None):  # type: ignore
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)
