# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class MaxUnpool(OpRun):
    def _run(self, X, indices, output_shape=None, kernel_shape=None, pads=None, strides=None):  # type: ignore
        pooling_dims = len(X.shape) - 2
        if pooling_dims > 3:
            raise NotImplementedError(
                f"Unsupported pooling size {pooling_dims} for operator MaxUnpool."
            )
        kernel_shape = kernel_shape or self.kernel_shape  # type: ignore
        pads = pads or self.pads  # type: ignore
        strides = strides or self.strides  # type: ignore

        if strides is None:
            strides = [1 for d in kernel_shape]
        if pads is None:
            pads = [0 for d in range(len(kernel_shape) * 2)]

        inferred_shape = np.empty((len(X.shape),), dtype=np.int64)
        inferred_shape[0] = X.shape[0]
        inferred_shape[1] = X.shape[1]

        for dim in range(0, len(kernel_shape)):
            inferred_shape[dim + 2] = (
                (X.shape[dim + 2] - 1) * strides[dim]
                - (pads[dim] + pads[len(kernel_shape) + dim])
                + kernel_shape[dim]
            )

        if output_shape is None:
            shape = inferred_shape
        else:
            shape = output_shape

        total_elements = np.prod(X.shape)
        Y = np.zeros((np.prod(inferred_shape),), dtype=X.dtype)

        I_data = indices.flatten()
        X_data = X.flatten()

        for cur_elem in range(total_elements):
            Y[I_data[cur_elem]] = X_data[cur_elem]

        Y = Y.reshape(tuple(inferred_shape))
        res = np.zeros(shape, dtype=Y.dtype)
        slices = tuple(slice(0, i) for i in inferred_shape)
        res[slices] = Y
        return (res,)
