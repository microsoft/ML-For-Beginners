# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class DynamicQuantizeLinear(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.dtype = np.uint8

    def _run(self, x):  # type: ignore
        # args: x, y_scale, zero_point
        qmin, qmax = 0, 255
        maxx = np.maximum(0, np.max(x))
        minx = np.minimum(0, np.min(x))
        y_scale = (maxx - minx) / (qmax - qmin)
        intermediate_zero_point = np.round(qmin - minx) / y_scale
        y_zero_point = np.round(np.clip(intermediate_zero_point, qmin, qmax)).astype(
            self.dtype
        )
        y = np.clip(np.round(x / y_scale) + y_zero_point, qmin, qmax)
        return (
            y.astype(self.dtype),
            y_scale.astype(x.dtype),
            y_zero_point.astype(self.dtype),
        )
