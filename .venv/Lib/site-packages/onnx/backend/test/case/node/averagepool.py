# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.backend.test.case.node.pool_op_common import (
    get_output_shape,
    get_pad_shape,
    pool,
)


class AveragePool(Base):
    @staticmethod
    def export_averagepool_2d_precomputed_pads() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array(
            [
                [
                    [
                        [7, 7.5, 8, 8.5, 9],
                        [9.5, 10, 10.5, 11, 11.5],
                        [12, 12.5, 13, 13.5, 14],
                        [14.5, 15, 15.5, 16, 16.5],
                        [17, 17.5, 18, 18.5, 19],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(
            node, inputs=[x], outputs=[y], name="test_averagepool_2d_precomputed_pads"
        )

    @staticmethod
    def export_averagepool_2d_precomputed_pads_count_include_pad() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
            count_include_pad=1,
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array(
            [
                [
                    [
                        [2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                        [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                        [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                        [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                        [6.1200, 8.4000, 10.8000, 8.8800, 6.8400],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_averagepool_2d_precomputed_pads_count_include_pad",
        )

    @staticmethod
    def export_averagepool_2d_precomputed_strides() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[4, 6], [14, 16]]]]).astype(np.float32)

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_averagepool_2d_precomputed_strides",
        )

    @staticmethod
    def export_averagepool_2d_precomputed_same_upper() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad="SAME_UPPER",
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[4, 5.5, 7], [11.5, 13, 14.5], [19, 20.5, 22]]]]).astype(
            np.float32
        )

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_averagepool_2d_precomputed_same_upper",
        )

    @staticmethod
    def export_averagepool_1d_default() -> None:
        """
        input_shape: [1, 3, 32]
        output_shape: [1, 3, 31]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2],
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2]
        strides = [1]
        out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_1d_default")

    @staticmethod
    def export_averagepool_2d_default() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_default")

    @staticmethod
    def export_averagepool_3d_default() -> None:
        """
        input_shape: [1, 3, 32, 32, 32]
        output_shape: [1, 3, 31, 31, 31]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2, 2],
        )
        x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_3d_default")

    @staticmethod
    def export_averagepool_2d_same_upper() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_UPPER",
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape("SAME_UPPER", x_shape[2:], kernel_shape, strides)
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
            constant_values=np.nan,
        )
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_same_upper")

    @staticmethod
    def export_averagepool_2d_same_lower() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_LOWER",
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape("SAME_LOWER", x_shape[2:], kernel_shape, strides)
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
            constant_values=np.nan,
        )
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_same_lower")

    @staticmethod
    def export_averagepool_2d_pads() -> None:
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2],
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape(
            "VALID", np.add(x_shape[2:], pad_shape), kernel_shape, strides
        )
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=np.nan,
        )
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_pads")

    @staticmethod
    def export_averagepool_2d_pads_count_include_pad() -> None:
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2],
            count_include_pad=1,
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape(
            "VALID", np.add(x_shape[2:], pad_shape), kernel_shape, strides
        )
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        y = pool(
            padded,
            x_shape,
            kernel_shape,
            strides,
            out_shape,
            pad_shape,
            "AVG",
            count_include_pad=1,
        )

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_averagepool_2d_pads_count_include_pad",
        )

    @staticmethod
    def export_averagepool_2d_strides() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[5, 5],
            strides=[3, 3],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), "AVG")

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_strides")

    @staticmethod
    def export_averagepool_2d_ceil() -> None:
        """
        input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            ceil_mode=True,
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
        y = np.array([[[[6, 7.5], [12, 13.5]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_ceil")

    @staticmethod
    def export_averagepool_2d_dilations() -> None:
        """
        input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "AveragePool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[1, 1],
            dilations=[2, 2],
            ceil_mode=True,
        )

        # input shape: [1, 1, 4, 4]
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

        y = np.array([[[[6, 7], [10, 11]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_dilations")
