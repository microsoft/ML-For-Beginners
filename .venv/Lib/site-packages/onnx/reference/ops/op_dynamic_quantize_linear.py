# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class DynamicQuantizeLinear(OpRun):
    def _run(self, x):  # type: ignore
        # args: x, y_scale, zero_point
        dtype, qmin, qmax = np.uint8, 0, 255
        maxx = np.float32(np.maximum(0, np.max(x)))
        minx = np.float32(np.minimum(0, np.min(x)))
        y_scale = np.float32(1.0 if maxx == minx else (maxx - minx)) / np.float32(
            qmax - qmin
        )

        # scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);

        initial_zero_point = np.float32(qmin) - minx / y_scale
        zp = max(qmin, min(qmax, initial_zero_point))
        zpi = np.rint(zp)

        y = np.clip(np.rint(x / y_scale) + zpi, qmin, qmax)
        return (
            y.astype(dtype),
            y_scale.astype(x.dtype),
            zpi.astype(dtype),
        )
