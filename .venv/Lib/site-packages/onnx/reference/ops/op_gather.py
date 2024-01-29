# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Gather(OpRun):
    def _run(self, x, indices, axis=None):  # type: ignore
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        if indices.size == 0:
            return (np.empty((0,), dtype=x.dtype),)
        try:
            return (np.take(x, indices, axis=axis),)
        except TypeError:
            # distribution x86 requires int32.
            return (np.take(x, indices.astype(int), axis=axis),)
