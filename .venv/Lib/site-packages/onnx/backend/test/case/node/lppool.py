# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_pool_common import (
    get_output_shape_auto_pad,
    get_output_shape_explicit_padding,
    get_pad_shape,
    pool,
)


class LpPool(Base):
    @staticmethod
    def export_lppool_1d_default() -> None:
        """
        input_shape: [1, 3, 32]
        output_shape: [1, 3, 31]
        """
        p = 3
        kernel_shape = [2]
        strides = [1]
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            strides=strides,
            p=p,
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_1d_default")

    @staticmethod
    def export_lppool_2d_default() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        p = 4
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            p=p,
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_default")

    @staticmethod
    def export_lppool_3d_default() -> None:
        """
        input_shape: [1, 3, 32, 32, 32]
        output_shape: [1, 3, 31, 31, 31]
        """
        p = 3
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2, 2],
            p=p,
        )
        x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_3d_default")

    @staticmethod
    def export_lppool_2d_same_upper() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
        """
        p = 2
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_UPPER",
            p=p,
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape_auto_pad(
            "SAME_UPPER", x_shape[2:], kernel_shape, strides
        )
        pad_shape = get_pad_shape(
            "SAME_UPPER", x_shape[2:], kernel_shape, strides, out_shape
        )
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_same_upper")

    @staticmethod
    def export_lppool_2d_same_lower() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
        """
        p = 4
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_LOWER",
            p=p,
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape_auto_pad(
            "SAME_LOWER", x_shape[2:], kernel_shape, strides
        )
        pad_shape = get_pad_shape(
            "SAME_LOWER", x_shape[2:], kernel_shape, strides, out_shape
        )
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_same_lower")

    @staticmethod
    def export_lppool_2d_pads() -> None:
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        p = 3
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2],
            p=p,
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = pad_top = pad_right = pad_left = 2
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        out_shape, pads = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", pads, p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_pads")

    @staticmethod
    def export_lppool_2d_strides() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        p = 2
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[5, 5],
            strides=[3, 3],
            p=p,
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "LPPOOL", p=p)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_strides")

    @staticmethod
    def export_lppool_2d_dilations() -> None:
        """
        input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
        p = 2
        node = onnx.helper.make_node(
            "LpPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[1, 1],
            dilations=[2, 2],
            p=p,
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]
                ]
            ]
        ).astype(np.float32)

        y = np.array(
            [
                [
                    [
                        [14.560219778561036, 16.24807680927192],
                        [21.633307652783937, 23.49468024894146],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_lppool_2d_dilations")
