# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    if num_axes * 2 != len(raw_pads):
        raise RuntimeError(
            "The number of elements in raw_pads should be 2 times the number of axes"
        )

    pad_width = [(0, 0)] * input_rank
    for i, axis in enumerate(axes):
        pad_begin = raw_pads[i]
        pad_end = raw_pads[num_axes + i]
        pad_width[axis] = (pad_begin, pad_end)

    if mode == "constant":
        return np.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_values
        ).astype(data.dtype)
    return np.pad(data, pad_width=pad_width, mode=mode).astype(data.dtype)


class Pad_1(OpRun):
    def _run(self, data, paddings=None, mode=None, value=None):  # type: ignore
        if value is None:
            value = 0
        return (_pad_impl(data, paddings, mode=mode, constant_values=value),)


class Pad_2(OpRun):
    def _run(self, data, pads=None, mode=None, value=None):  # type: ignore
        if value is None:
            value = 0
        return (_pad_impl(data, pads, mode=mode, constant_values=value),)


class Pad_11(OpRun):
    def _run(self, data, pads, constant_value=None, mode=None):  # type: ignore
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=None),
        )


class Pad_18(OpRun):
    def _run(self, data, pads, constant_value=None, axes=None, mode=None):  # type: ignore
        if constant_value is None:
            constant_value = 0
        return (
            _pad_impl(data, pads, mode=mode, constant_values=constant_value, axes=axes),
        )
