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


class MaxPool(Base):
    @staticmethod
    def export_maxpool_2d_uint8() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
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
        ).astype(np.uint8)
        y = np.array(
            [
                [
                    [
                        [13, 14, 15, 15, 15],
                        [18, 19, 20, 20, 20],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                    ]
                ]
            ]
        ).astype(np.uint8)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_uint8")

    @staticmethod
    def export_maxpool_2d_precomputed_pads() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
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
                        [13, 14, 15, 15, 15],
                        [18, 19, 20, 20, 20],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_precomputed_pads")

    @staticmethod
    def export_maxpool_with_argmax_2d_precomputed_pads() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y", "z"],
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
                        [13, 14, 15, 15, 15],
                        [18, 19, 20, 20, 20],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                        [23, 24, 25, 25, 25],
                    ]
                ]
            ]
        ).astype(np.float32)
        z = np.array(
            [
                [
                    [
                        [12, 13, 14, 14, 14],
                        [17, 18, 19, 19, 19],
                        [22, 23, 24, 24, 24],
                        [22, 23, 24, 24, 24],
                        [22, 23, 24, 24, 24],
                    ]
                ]
            ]
        ).astype(np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[y, z],
            name="test_maxpool_with_argmax_2d_precomputed_pads",
        )

    @staticmethod
    def export_maxpool_2d_precomputed_strides() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "MaxPool", inputs=["x"], outputs=["y"], kernel_shape=[2, 2], strides=[2, 2]
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
        y = np.array([[[[7, 9], [17, 19]]]]).astype(np.float32)

        expect(
            node, inputs=[x], outputs=[y], name="test_maxpool_2d_precomputed_strides"
        )

    @staticmethod
    def export_maxpool_with_argmax_2d_precomputed_strides() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y", "z"],
            kernel_shape=[2, 2],
            strides=[2, 2],
            storage_order=1,
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
        y = np.array([[[[7, 9], [17, 19]]]]).astype(np.float32)
        z = np.array([[[[6, 16], [8, 18]]]]).astype(np.int64)

        expect(
            node,
            inputs=[x],
            outputs=[y, z],
            name="test_maxpool_with_argmax_2d_precomputed_strides",
        )

    @staticmethod
    def export_maxpool_2d_precomputed_same_upper() -> None:
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
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
        y = np.array([[[[7, 9, 10], [17, 19, 20], [22, 24, 25]]]]).astype(np.float32)

        expect(
            node, inputs=[x], outputs=[y], name="test_maxpool_2d_precomputed_same_upper"
        )

    @staticmethod
    def export_maxpool_1d_default() -> None:
        """
        input_shape: [1, 3, 32]
        output_shape: [1, 3, 31]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2],
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = [2]
        strides = [1]
        out_shape, _ = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX")

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_1d_default")

    @staticmethod
    def export_maxpool_2d_default() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
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
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX")

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_default")

    @staticmethod
    def export_maxpool_3d_default() -> None:
        """
        input_shape: [1, 3, 32, 32, 32]
        output_shape: [1, 3, 31, 31, 31]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2, 2],
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
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX")

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_3d_default")

    @staticmethod
    def export_maxpool_2d_same_upper() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_UPPER",
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
            constant_values=np.nan,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX", pads)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_same_upper")

    @staticmethod
    def export_maxpool_2d_same_lower() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            auto_pad="SAME_LOWER",
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
            constant_values=np.nan,
        )
        pads = [pad_top, pad_left, pad_bottom, pad_right]
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX", pads)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_same_lower")

    @staticmethod
    def export_maxpool_2d_pads() -> None:
        """
        input_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2],
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
            constant_values=np.nan,
        )

        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX", pads)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_pads")

    @staticmethod
    def export_maxpool_2d_strides() -> None:
        """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        node = onnx.helper.make_node(
            "MaxPool", inputs=["x"], outputs=["y"], kernel_shape=[5, 5], strides=[3, 3]
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        pads = None
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape, pads = get_output_shape_explicit_padding(
            pads, x_shape[2:], kernel_shape, strides
        )
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, "MAX")

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_strides")

    @staticmethod
    def export_maxpool_2d_ceil() -> None:
        """
        input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "MaxPool",
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
        y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_ceil")

    @staticmethod
    def export_maxpool_2d_dilations() -> None:
        """
        input_shape: [1, 1, 4, 4]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[1, 1],
            dilations=[2, 2],
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
        y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_2d_dilations")

    @staticmethod
    def export_maxpool_3d_dilations() -> None:
        """
        input_shape: [1, 1, 4, 4, 4]
        output_shape: [1, 1, 2, 2, 2]
        """
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2, 2],
            strides=[1, 1, 1],
            dilations=[2, 2, 2],
        )
        x = np.array(
            [
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[[11, 12], [15, 16]], [[11, 12], [15, 16]]]]]).astype(
            np.float32
        )

        expect(node, inputs=[x], outputs=[y], name="test_maxpool_3d_dilations")

    @staticmethod
    def export_maxpool_3d_dilations_use_ref_impl() -> None:
        """
        input_shape: [1, 1, 4, 4, 4]
        output_shape: [1, 1, 2, 2, 2]
        """
        dilations = [2, 2, 2]
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        ceil_mode = False
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[2, 2, 2],
            strides=[1, 1, 1],
            dilations=dilations,
        )
        x = np.array(
            [
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ],
                    ]
                ]
            ]
        ).astype(np.float32)

        x_shape = x.shape[2:]
        out_shape, pads = get_output_shape_explicit_padding(
            None, x_shape, kernel_shape, strides, dilations, ceil_mode=ceil_mode
        )
        padded = x
        y = pool(
            padded,
            (1, 1, *x_shape),
            kernel_shape,
            strides,
            out_shape,
            "MAX",
            pads,
            dilations=dilations,
        )

        expect(
            node, inputs=[x], outputs=[y], name="test_maxpool_3d_dilations_use_ref_impl"
        )

    @staticmethod
    def export_maxpool_3d_dilations_use_ref_impl_large() -> None:
        x_shape = (32, 32, 32)
        dilations = (2, 2, 2)
        kernel_shape = (5, 5, 5)
        strides = (3, 3, 3)
        ceil_mode = True

        node = onnx.helper.make_node(
            "MaxPool",
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            ceil_mode=ceil_mode,
        )

        x = np.random.randn(1, 1, *x_shape).astype(np.float32)
        out_shape, pads = get_output_shape_explicit_padding(
            None, x_shape, kernel_shape, strides, dilations, ceil_mode=ceil_mode
        )
        padded = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (pads[0], pads[3]),
                (pads[1], pads[4]),
                (pads[2], pads[5]),
            ),
            mode="constant",
            constant_values=0,
        )
        y = pool(
            padded,
            (1, 1, *x_shape),
            kernel_shape,
            strides,
            out_shape,
            "MAX",
            pads,
            dilations=dilations,
        )

        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_maxpool_3d_dilations_use_ref_impl_large",
        )
