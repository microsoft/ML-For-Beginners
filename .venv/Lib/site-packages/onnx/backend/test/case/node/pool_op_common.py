# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Sequence

import numpy as np


def get_pad_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    output_spatial_shape: Sequence[int],
) -> Sequence[int]:
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass
    return pad_shape


def get_output_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
) -> Sequence[int]:
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i]))
            )
    elif auto_pad == "VALID":
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1))
                    / float(strides_spatial[i])
                )
            )
    return out_shape


def lp_pool(x: np.array, p: int) -> float:
    y = 0
    for v in np.nditer(x):
        y += abs(v) ** p
    return y ** (1.0 / p)


def pool(
    padded: np.ndarray,
    x_shape: Sequence[int],
    kernel_shape: Sequence[int],
    strides_shape: Sequence[int],
    out_shape: Sequence[int],
    pad_shape: Sequence[int],
    pooling_type: str,
    count_include_pad: int = 0,
    p: int = 1,
) -> np.ndarray:
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1], *list(out_shape)])

    def lp_pool_p(x):
        return lp_pool(x, p)

    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                    (x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i]
                    + 1
                )
            )
            for i in range(spatial_size)
        ],
    ):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [
                window[i]
                for i in list(
                    itertools.product(
                        *[
                            range(
                                strides_shape[i] * shape[i + 2],
                                strides_shape[i] * shape[i + 2] + kernel_shape[i],
                            )
                            for i in range(spatial_size)
                        ]
                    )
                )
            ]
        )
        if pooling_type == "AVG":
            f = np.average
        elif pooling_type == "MAX":
            f = np.max
        elif pooling_type == "LPPOOL":
            f = lp_pool_p
        else:
            raise NotImplementedError(
                f"Pooling type {pooling_type} does not support. Should be AVG, MAX"
            )

        if count_include_pad == 1 and (
            pooling_type == "AVG" or pooling_type == "LPPOOL"
        ):
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)
