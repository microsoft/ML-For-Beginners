# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Trilu(OpRun):
    def _run(self, x, k=None, upper=None):  # type: ignore
        k = 0 if k is None else int(k)
        if upper:  # type: ignore
            return (np.triu(x, k),)
        return (np.tril(x, k).astype(x.dtype),)
