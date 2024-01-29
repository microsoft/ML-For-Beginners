# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class MatMulInteger(OpRun):
    def _run(self, A, B, a_zero_point=None, b_zero_point=None):  # type: ignore
        A32 = A.astype(np.int32)
        if a_zero_point is not None:
            A32 -= a_zero_point
        B32 = B.astype(np.int32)
        if b_zero_point is not None:
            B32 -= b_zero_point
        return (A32 @ B32,)
