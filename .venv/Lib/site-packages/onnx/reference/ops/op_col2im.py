# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_indices, _is_out


def _col2im_shape_check_2d(X, output_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    output_height, output_width = output_shape
    kernel_height, kernel_width = kernel_shape
    dilation_height, dilation_width = dilations
    stride_height, stride_width = strides

    ndim = len(X.shape)
    if not (
        (ndim == 2 and X.shape[0] != 0 and X.shape[1] != 0)
        or (ndim == 3 and X.shape[1] != 0 and X.shape[2] != 0)
    ):
        raise ValueError(
            "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input."
        )

    batch_dim = 0 if len(X.shape) == 3 else -1
    n_input_plane = X.shape[batch_dim + 1]

    if n_input_plane % (kernel_width * kernel_height) != 0:
        raise ValueError(
            f"Expected size of input's dimension 1 to be divisible by the "
            f"product of kernel_size, but got input.size(1)={n_input_plane} "
            f"and kernel_size={kernel_shape}."
        )

    input_length = X.shape[batch_dim + 2]
    n_blocks_height = (
        output_height + pads[0, :].sum() - dilation_height * (kernel_height - 1) - 1
    ) // stride_height + 1
    n_blocks_width = (
        output_width + pads[1, :].sum() - dilation_width * (kernel_width - 1) - 1
    ) // stride_width + 1

    if input_length != (n_blocks_height * n_blocks_width):
        raise ValueError(
            f"Given batch_dim={batch_dim}, n_input_plane={n_input_plane}, X.shape={X.shape}, "
            f"output_shape={output_shape}, kernel_shape={kernel_shape}, "
            f"dilations={dilations}, pads={pads}, strides={strides}, "
            f"expected size of input's dimension 2 to match the calculated number of ",
            f"sliding blocks {n_blocks_height} * {n_blocks_width} = {n_blocks_height * n_blocks_width}, "
            f"but got input.size(2)={input_length}.",
        )

    if not (n_blocks_height >= 1 and n_blocks_width >= 1):
        raise ValueError(
            f"Given batch_dim={batch_dim}, n_input_plane={n_input_plane}, X.shape={X.shape}, "
            f"output_shape={output_shape}, kernel_shape={kernel_shape}, "
            f"dilations={dilations}, pads={pads}, strides={strides}, "
            f"calculated shape of the array of sliding blocks as ({n_blocks_height}, {n_blocks_width}), "
            f"which is too small (non-positive)."
        )


def _col2im_naive_implementation_2d(res, image_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    # source: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/im2col.h

    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check_2d(res, image_shape, kernel_shape, dilations, new_pads, strides)

    data_col = res.ravel()
    data_im = np.zeros(image_shape, dtype=res.dtype).flatten()

    kernel_h, kernel_w = kernel_shape
    channels_col = kernel_h * kernel_w
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_h, pad_w = new_pads[:, 0]
    height, width = image_shape
    output_height, output_width = image_shape

    height_col = (
        output_height + new_pads[0, :].sum() - (dilation_h * (kernel_h - 1) + 1)
    ) // stride_h + 1
    width_col = (
        output_width + new_pads[1, :].sum() - (dilation_w * (kernel_w - 1) + 1)
    ) // stride_w + 1

    for c_col in range(channels_col):
        w_offset = c_col % kernel_w
        h_offset = (c_col // kernel_w) % kernel_h
        c_im = c_col // (kernel_h * kernel_w)

        for h_col in range(height_col):
            h_im = h_col * stride_h - pad_h + h_offset * dilation_h
            for w_col in range(width_col):
                w_im = w_col * stride_w - pad_w + w_offset * dilation_w
                if 0 <= h_im < height and 0 <= w_im < width:
                    i_im = (c_im * height + h_im) * width + w_im
                    i_col = (c_col * height_col + h_col) * width_col + w_col
                    if 0 <= i_col < data_col.shape[0]:
                        data_im[i_im] += data_col[i_col]

    return data_im.reshape(image_shape)


def _col2im_shape_check(X, output_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    n_input_plane = X.shape[0]

    kernel_size = np.prod(kernel_shape)

    if n_input_plane % kernel_size != 0:
        raise ValueError(
            f"Expected size of input's dimension 1 to be divisible by the "
            f"product of kernel_size={kernel_size}, "
            f"but got input.size(1)={n_input_plane} "
            f"and kernel_shape={kernel_shape}, X.shape={X.shape}, output_shape={output_shape}."
        )

    input_length = X.shape[1]
    n_dims = len(output_shape)
    n_blocks = []
    for i in range(n_dims):
        n_block = (
            output_shape[i]
            + pads[i, :].sum()
            - dilations[i] * (kernel_shape[i] - 1)
            - 1
        ) // strides[i] + 1
        n_blocks.append(n_block)

    block_size = np.prod(n_blocks)
    if input_length != block_size:
        raise ValueError(
            f"Given n_input_plane={n_input_plane}, X.shape={X.shape}, "
            f"output_shape={output_shape}, kernel_shape={kernel_shape}, "
            f"dilations={dilations}, pads={pads}, strides={strides}, "
            f"expected size of input's dimension 2 to match the calculated number of "
            f"sliding blocks {n_blocks} = {block_size}, "
            f"but got input.size(2)={input_length}.",
        )


def col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    """
    Naive implementation for `col2im`.
    """
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check(data, image_shape, kernel_shape, dilations, new_pads, strides)

    data_col = data
    data_im = np.zeros(image_shape, dtype=data.dtype)

    dim_col = []
    for i in range(n_dims):
        col = (
            image_shape[i]
            + new_pads[i, :].sum()
            - (dilations[i] * (kernel_shape[i] - 1) + 1)
        ) // strides[i] + 1
        dim_col.append(col)

    kernel_size = np.prod(kernel_shape)
    col_size = np.prod(dim_col)
    for c_col in range(kernel_size):
        offset = _get_indices(c_col, kernel_shape)

        for col in range(col_size):
            ind_col = _get_indices(col, dim_col)
            ind_im = []
            for i in range(n_dims):
                ind = (
                    ind_col[i] * strides[i] - new_pads[i, 0] + offset[i] * dilations[i]
                )
                ind_im.append(ind)

            if not _is_out(ind_im, data_im.shape):
                data_im[tuple(ind_im)] += data_col[c_col, col]

    return data_im


class Col2Im(OpRun):
    def _run(self, data, image_shape, block_shape, dilations=None, pads=None, strides=None):  # type: ignore
        if dilations is None:
            dilations = [1 for s in image_shape]
        if pads is None:
            pads = [0 for s in image_shape] * 2
        if strides is None:
            strides = [1 for s in image_shape]

        bl = np.prod(block_shape)
        C = data.shape[1] // bl
        data = data.reshape(data.shape[:1] + (C,) + (bl,) + data.shape[2:])

        ks = tuple(block_shape)
        res = None
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                out = col2im_naive_implementation(
                    data[n, c, ...], image_shape, ks, dilations, pads, strides
                )
                if res is None:
                    new_shape = data.shape[:2] + out.shape
                    res = np.empty(new_shape, dtype=data.dtype)
                res[n, c, ...] = out
        return (res,)  # type: ignore
