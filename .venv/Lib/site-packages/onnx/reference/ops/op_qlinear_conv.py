# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_conv import _conv_implementation


class QLinearConv(OpRun):
    def _run(  # type: ignore
        self,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        y_scale,
        y_zero_point,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        auto_pad = auto_pad or self.auto_pad  # type: ignore
        dilations = dilations or self.dilations  # type: ignore
        group = group or self.group  # type: ignore
        kernel_shape = kernel_shape or self.kernel_shape  # type: ignore
        pads = pads or self.pads  # type: ignore
        strides = strides or self.strides  # type: ignore

        X = x.astype(np.int32)
        if x_zero_point is not None:
            X -= x_zero_point
        W = w.astype(np.int32)
        if w_zero_point is not None:
            if len(w_zero_point.shape) == 1 and w_zero_point.shape[0] == W.shape[0]:
                missing = (w_zero_point.shape[0],) + (1,) * (len(W.shape) - 1)
                W -= w_zero_point.reshape(missing)
            else:
                W -= w_zero_point

        res = _conv_implementation(
            X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
        ).astype(np.int32)

        R = res * (x_scale * w_scale / y_scale)
        if y_zero_point is not None:
            R += y_zero_point
            if y_zero_point.dtype == np.int8:
                R = np.clip(R, -128, 127)
            else:
                R = np.clip(R, 0, 255)
            return (np.round(R).astype(y_zero_point.dtype),)
        if x.dtype == np.int8:
            R = np.clip(R, -128, 127)
        else:
            R = np.clip(R, 0, 255)
        return (np.round(R).astype(x.dtype),)
