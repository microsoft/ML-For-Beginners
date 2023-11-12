# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

from onnx.reference.ops.experimental._op_run_experimental import OpRunExperimental
from onnx.reference.ops.op_conv import im2col


class Im2Col(OpRunExperimental):
    def _run(self, img, kernel_shape, dilations=None, pads=None, strides=None):  # type: ignore
        if dilations is None:
            dilations = [1 for s in img.shape[2:]]
        if pads is None:
            pads = [0 for s in img.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in img.shape[2:]]

        return (im2col(img, tuple(kernel_shape[2:]), dilations, pads, strides),)  # type: ignore
