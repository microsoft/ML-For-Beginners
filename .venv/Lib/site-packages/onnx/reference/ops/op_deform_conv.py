# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _deform_conv_implementation(  # type: ignore
    X, W, offset, B, mask, dilations, group, kernel_shape, offset_group, pads, strides
):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    if group is None:
        group = 1
    if offset_group is None:
        offset_group = 1

    n, ic = X.shape[:2]
    oc = W.shape[0]
    output_shape = offset.shape[2:]

    if ic != W.shape[1] * group or oc % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}."
        )
    ics_per_group, ocs_per_group = W.shape[1], oc // group

    if ic % offset_group != 0:
        raise ValueError("Number of input channels must be divisible by offset_group.")
    ics_per_offset_group = ic // offset_group

    if offset_group * np.prod(kernel_shape) * len(kernel_shape) != offset.shape[1]:
        raise ValueError(
            f"Offset shape {offset.shape} is inconsistent with offset_group {offset_group} "
            f"and kernel shape {kernel_shape}."
        )
    offset = offset.reshape(
        (n, offset_group, *kernel_shape, len(kernel_shape), *output_shape)
    )

    if mask is None:
        mask = np.ones((n, offset_group * np.prod(kernel_shape), *output_shape))
    mask = mask.reshape((n, offset_group, *kernel_shape, *output_shape))

    from onnx.reference.ops._op_list import GridSample

    if len(X.shape) == 4:
        ih, iw = X.shape[2:]
        oh, ow = offset.shape[-2:]
        kh, kw = kernel_shape
        sth, stw = strides
        dh, dw = dilations
        kh_new, kw_new = (kh - 1) * dh + 1, (kw - 1) * dw + 1

        if oh != int(((ih - kh_new + pads[0] + pads[2]) / sth) + 1) or ow != int(
            ((iw - kw_new + pads[1] + pads[3]) / stw) + 1
        ):
            raise RuntimeError(
                "Padding, dilation, stride, and kernel shape incompatible with output shape."
            )

        bh, bw = -pads[0], -pads[1]

        res = np.zeros((n, oc, oh, ow), dtype=X.dtype)
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1))

        # Calculate coordinates of sampling points within kernel
        kernel_pos_w, kernel_pos_h = np.meshgrid(
            np.arange(0, kw_new, dw), np.arange(0, kh_new, dh)
        )
        kernel_pos_wrt_first_elem = np.stack(
            (kernel_pos_h, kernel_pos_w), axis=2
        )  # shape (kH, kW, 2)

        for batch_idx in range(n):
            for oc_idx in range(oc):
                for ic_idx in range(ic):
                    # Group convolution logic
                    if ic_idx // ics_per_group != oc_idx // ocs_per_group:
                        # Input channel and output channel don't belong to same group
                        continue

                    # Offset group logic
                    offset_group_idx = ic_idx // ics_per_offset_group

                    for i in range(oh):
                        h_coord = bh + sth * i
                        for j in range(ow):
                            w_coord = bw + stw * j
                            # (h_coord, w_coord) is coord of top left elem of kernel

                            kernel = np.copy(kernel_pos_wrt_first_elem).astype(float)
                            kernel[:, :, 0] += (
                                h_coord
                                + offset[batch_idx, offset_group_idx, :, :, 0, i, j]
                            )
                            kernel[:, :, 1] += (
                                w_coord
                                + offset[batch_idx, offset_group_idx, :, :, 1, i, j]
                            )

                            # GridSample expects normalized grid coordinates
                            kernel[:, :, 0] = kernel[:, :, 0] / (ih - 1) * 2 - 1
                            kernel[:, :, 1] = kernel[:, :, 1] / (iw - 1) * 2 - 1

                            kernel = np.expand_dims(kernel, 0)  # add batch dimension
                            kernel = np.flip(
                                kernel, 3
                            )  # spatial GridSample expects (x, y) input
                            grid_sample_output = GridSample.eval(
                                X[batch_idx : batch_idx + 1, ic_idx : ic_idx + 1],
                                kernel,
                                align_corners=1,
                            )

                            conv_value = np.multiply(
                                grid_sample_output,
                                W[oc_idx, ic_idx % ics_per_group, :, :],
                            )
                            conv_value = np.multiply(
                                conv_value,
                                mask[batch_idx, offset_group_idx, :, :, i, j],
                            )
                            res[batch_idx, oc_idx, i, j] += np.sum(conv_value)

        return res

    raise RuntimeError(
        f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
        f"kernel_shape={kernel_shape} is not implemented yet."
    )


class DeformConv(OpRun):
    def _run(  # type: ignore
        self,
        X,
        W,
        offset,
        B=None,
        mask=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        offset_group=None,
        pads=None,
        strides=None,
    ):
        if len(X.shape) < 3:
            raise ValueError(
                f"X must have at least 3 dimensions but its shape is {X.shape}."
            )
        return (
            _deform_conv_implementation(
                X,
                W,
                offset,
                B,
                mask,
                dilations,
                group,
                kernel_shape,
                offset_group,
                pads,
                strides,
            ),
        )
