# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221,W0622

import numpy as np

from onnx.reference.op_run import OpRun


def _specify_int64(indices, inverse_indices, counts):  # type: ignore
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


class Unique(OpRun):
    def _run(self, x, axis=None, sorted=None):  # type: ignore
        if axis is None or np.isnan(axis):
            y, indices, inverse_indices, counts = np.unique(x, True, True, True)
        else:
            y, indices, inverse_indices, counts = np.unique(
                x, True, True, True, axis=axis
            )
        if len(self.onnx_node.output) == 1:
            return (y,)

        if not sorted:
            argsorted_indices = np.argsort(indices)
            inverse_indices_map = dict(
                zip(argsorted_indices, np.arange(len(argsorted_indices)))
            )
            indices = indices[argsorted_indices]
            y = np.take(x, indices, axis=0)
            inverse_indices = np.asarray(
                [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64
            )
            counts = counts[argsorted_indices]

        indices, inverse_indices, counts = _specify_int64(
            indices, inverse_indices, counts
        )
        if len(self.onnx_node.output) == 2:
            return (y, indices)
        if len(self.onnx_node.output) == 3:
            return (y, indices, inverse_indices)
        return (y, indices, inverse_indices, counts)
