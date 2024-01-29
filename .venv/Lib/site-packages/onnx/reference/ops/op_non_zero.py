# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class NonZero(OpRun):
    def _run(self, x):  # type: ignore
        # Specify np.int64 for Windows x86 machines
        res = np.vstack(np.nonzero(x)).astype(np.int64)
        return (res,)
