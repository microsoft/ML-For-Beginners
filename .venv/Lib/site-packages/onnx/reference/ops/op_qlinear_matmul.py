# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class QLinearMatMul(OpRun):
    def _run(  # type: ignore
        self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
    ):
        A = a.astype(np.int32)
        if a_zero_point is not None:
            A -= a_zero_point.astype(np.int32)
        B = b.astype(np.int32)
        if b_zero_point is not None:
            B -= b_zero_point.astype(np.int32)
        C = np.matmul(A, B)
        D = C * (a_scale * b_scale / y_scale)
        if y_zero_point is not None:
            D += y_zero_point
            return (np.round(D).astype(y_zero_point.dtype),)
        return (np.round(D).astype(a.dtype),)
