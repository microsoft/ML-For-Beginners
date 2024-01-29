# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops.experimental._op_run_experimental import OpRunExperimental
from onnx.reference.ops_optimized.op_conv_optimized import im2col_fast


class Im2Col(OpRunExperimental):
    def _run(self, img, kernel_shape, dilations=None, pads=None, strides=None):  # type: ignore
        if dilations is None:
            dilations = [1 for s in img.shape[2:]]
        if pads is None:
            pads = [0 for s in img.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in img.shape[2:]]

        if min(dilations) == max(dilations) == 1:
            return (im2col_fast(img, tuple(kernel_shape[2:]), pads, strides)[0],)  # type: ignore

        if dilations[0] != 1 or min(dilations) != max(dilations):
            # Let's compute the dilated kernel.
            nd = len(dilations)
            new_kernel_shape = []
            new_shape = list(kernel_shape)
            for i, d in enumerate(dilations):
                di = len(kernel_shape) - nd + i
                new_shape.append(kernel_shape[di] + (kernel_shape[di] - 1) * (d - 1))
                new_kernel_shape.append(
                    kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1)
                )
            kernel_shape = new_kernel_shape

        return (im2col_fast(img, tuple(kernel_shape[2:]), pads, strides),)  # type: ignore
