# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_col2im import col2im_naive_implementation


class ConvTranspose(OpRun):
    def _run(  # type: ignore
        self,
        X,
        W,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        pads=None,
        strides=None,
    ):
        if group != 1:
            raise RuntimeError(f"group={group} != 1 is not implemented yet.")
        if dilations is None:
            dilations = [1 for s in X.shape[2:]]
        if kernel_shape is None:
            kernel_shape = W.shape[2:]
        if output_padding is None:
            output_padding = [0 for s in X.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in X.shape[2:]]
        if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
            pads = [0 for i in range(2 * len(strides))]
        if pads is None:
            if output_shape is None:
                output_shape = [
                    X.shape[i + 2] * strides[i] for i in range(len(strides))
                ]
            total_padding = [
                strides[i] * (X.shape[i + 2] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - output_shape[i]
                for i in range(len(output_shape))
            ]
            pads_1 = []
            pads_2 = []
            for i in range(len(output_shape)):
                if auto_pad == "SAME_UPPER":
                    pads_1.append(total_padding[i] // 2)
                    pads_2.append(total_padding[i] - (total_padding[i] // 2))
                else:
                    pads_1.append(total_padding[i] - (total_padding[i] // 2))
                    pads_2.append(total_padding[i] // 2)
            pads = pads_1 + pads_2
            n_dims = len(pads) // 2
        else:
            n_dims = len(X.shape) - 2
            new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
            if output_shape is None:
                output_shape = [
                    strides[i] * (X.shape[i + 2] - 1)
                    + output_padding[i]
                    + ((kernel_shape[i] - 1) * dilations[i] + 1)
                    - new_pads[i, :].sum()
                    for i in range(n_dims)
                ]

        kernel_shape = W.shape[2:]
        kernel_size = np.prod(kernel_shape)
        num_output_channels = W.shape[1] * group
        kernel_dim = num_output_channels // group * kernel_size

        C = X.shape[1]  # num_inputs_channels
        m = kernel_dim  # kernel_dim
        n = np.prod(X.shape[2:])  # input_image_size
        k = C // group
        w_reshaped = W.reshape((group, k, m))
        final = None

        # N x C x H x W = X.shape
        # C x M/group x k1 x k2 = W.shape
        if group == 1:
            for image_id in range(X.shape[0]):
                w_t = w_reshaped[0].T
                gemm = np.matmul(w_t, X[image_id].reshape((k, n)))
                gemmc = gemm.reshape((num_output_channels, -1, gemm.shape[-1]))
                for c in range(num_output_channels):
                    res = col2im_naive_implementation(
                        gemmc[c], output_shape, kernel_shape, dilations, pads, strides
                    )
                    if final is None:
                        final = np.empty(
                            X.shape[:1] + (num_output_channels,) + res.shape,
                            dtype=X.dtype,
                        )
                    if B is not None:
                        res += B[c]
                    final[image_id, c, ...] = res[...]
        else:
            raise NotImplementedError(
                f"Implementation for group={group} > 1 is not available yet."
            )

        return (final.astype(X.dtype),)  # type: ignore[union-attr]
