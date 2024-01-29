# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):  # type: ignore
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)

    if num_axes * 2 != raw_pads.size:
        raise Exception("The number of elements in raw_pads should be 2 * num_axes")

    pad_width = []
    for _ in range(input_rank):
        pad_width += [[0, 0]]  # init to zero

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]

    if mode == "constant":
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y


class Pad(Base):
    @staticmethod
    def export_constant_pad() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        y = pad_impl(x, pads, "constant", 1.2)

        expect(node, inputs=[x, pads, value], outputs=[y], name="test_constant_pad")

    @staticmethod
    def export_reflection_edge_and_wrap_pad() -> None:
        for mode in ("edge", "reflect", "wrap"):
            node = onnx.helper.make_node(
                "Pad", inputs=["x", "pads"], outputs=["y"], mode=mode
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.int32)
            pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(
                np.int64
            )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            y = pad_impl(x, pads, mode)

            expect(node, inputs=[x, pads], outputs=[y], name=f"test_{mode}_pad")

    @staticmethod
    def export_constant_pad_axes() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value", "axes"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 3, 0, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        axes = np.array([1, 3], dtype=np.int64)
        y = pad_impl(
            x,
            pads,
            "constant",
            1.2,
            [1, 3],
        )

        expect(
            node,
            inputs=[x, pads, value, axes],
            outputs=[y],
            name="test_constant_pad_axes",
        )

    @staticmethod
    def export_constant_pad_negative_axes() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value", "axes"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 3, 0, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        axes = np.array([-3, -1], dtype=np.int64)
        y = pad_impl(
            x,
            pads,
            "constant",
            1.2,
            [-3, -1],
        )

        expect(
            node,
            inputs=[x, pads, value, axes],
            outputs=[y],
            name="test_constant_pad_negative_axes",
        )
