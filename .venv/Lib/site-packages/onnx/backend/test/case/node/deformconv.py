# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DeformConv(Base):
    @staticmethod
    def export() -> None:
        X = np.arange(9).astype(np.float32)
        X.shape = (1, 1, 3, 3)
        W = np.ones((1, 1, 2, 2), dtype=np.float32)

        # Convolution with padding
        offset_with_padding = np.zeros((1, 8, 4, 4), dtype=np.float32)
        offset_with_padding[
            0, 0, 0, 0
        ] = 0.5  # h-coord of [0, 0] element of kernel, at output position [0, 0]
        offset_with_padding[
            0, 5, 1, 2
        ] = -0.1  # w-coord of [1, 0] element of kernel, at output position [1, 2]

        node_with_padding = onnx.helper.make_node(
            "DeformConv",
            inputs=["X", "W", "offset_with_padding"],
            outputs=["Y_with_padding"],
            kernel_shape=[2, 2],
            pads=[1, 1, 1, 1],
        )
        Y_with_padding = np.array(
            [
                [
                    [
                        [0.0, 1.0, 3.0, 2.0],  # (1, 1, 4, 4) output tensor
                        [3.0, 8.0, 11.9, 7.0],
                        [9.0, 20.0, 24.0, 13.0],
                        [6.0, 13.0, 15.0, 8.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        expect(
            node_with_padding,
            inputs=[X, W, offset_with_padding],
            outputs=[Y_with_padding],
            name="test_basic_deform_conv_with_padding",
        )

        # Convolution without padding
        offset_without_padding = np.zeros((1, 8, 2, 2), dtype=np.float32)
        offset_without_padding[
            0, 0, 0, 0
        ] = 0.5  # h-coord of [0, 0] element of kernel, at output position [0, 0]
        offset_without_padding[
            0, 5, 0, 1
        ] = -0.1  # w-coord of [1, 0] element of kernel, at output position [0, 1]

        node_without_padding = onnx.helper.make_node(
            "DeformConv",
            inputs=["X", "W", "offset_without_padding"],
            outputs=["Y_without_padding"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
        )
        Y_without_padding = np.array(
            [
                [
                    [
                        [9.5, 11.9],  # (1, 1, 2, 2) output tensor
                        [20.0, 24.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        expect(
            node_without_padding,
            inputs=[X, W, offset_without_padding],
            outputs=[Y_without_padding],
            name="test_basic_deform_conv_without_padding",
        )

    @staticmethod
    def export_deformconv_with_mask_bias() -> None:
        X = np.arange(9).astype(np.float32)
        X.shape = (1, 1, 3, 3)
        W = np.ones((1, 1, 2, 2), dtype=np.float32)
        B = np.ones((1,), dtype=np.float32)

        offset = np.zeros((1, 8, 2, 2), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  # h-coord of [0, 0] element of kernel, at output position [0, 0]
        offset[
            0, 5, 0, 1
        ] = -0.1  # w-coord of [1, 0] element of kernel, at output position [0, 1]

        mask = np.ones((1, 4, 2, 2), dtype=np.float32)
        mask[0, 2, 1, 1] = 0.2  # [1, 0] element of kernel at output position [1, 1]

        node = onnx.helper.make_node(
            "DeformConv",
            inputs=["X", "W", "offset", "B", "mask"],
            outputs=["Y"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
        )
        Y = np.array(
            [
                [
                    [
                        [10.5, 12.9],  # (1, 1, 2, 2) output tensor
                        [21.0, 19.4],
                    ]
                ]
            ]
        ).astype(np.float32)
        expect(
            node,
            inputs=[X, W, offset, B, mask],
            outputs=[Y],
            name="test_deform_conv_with_mask_bias",
        )

    @staticmethod
    def export_deformconv_with_multiple_offset_groups() -> None:
        X = np.zeros((1, 2, 3, 3), dtype=np.float32)
        X[0, 0] = np.reshape(np.arange(9).astype(np.float32), (3, 3))
        X[0, 1] = np.reshape(np.arange(8, -1, -1).astype(np.float32), (3, 3))
        X.shape = (1, 2, 3, 3)
        W = np.ones((1, 2, 2, 2), dtype=np.float32)

        offset = np.zeros((1, 16, 2, 2), dtype=np.float32)
        offset[
            0, 0, 0, 0
        ] = 0.5  # h-coord of [0, 0] element of kernel in channel 0, at output position [0, 0]
        offset[
            0, 13, 0, 1
        ] = (
            -0.1
        )  # w-coord of [1, 0] element of kernel in channel 1, at output position [0, 1]

        node = onnx.helper.make_node(
            "DeformConv",
            inputs=["X", "W", "offset"],
            outputs=["Y"],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            offset_group=2,
        )
        Y = np.array(
            [
                [
                    [
                        [33.5, 32.1],  # (1, 1, 2, 2) output tensor
                        [32.0, 32.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        expect(
            node,
            inputs=[X, W, offset],
            outputs=[Y],
            name="test_deform_conv_with_multiple_offset_groups",
        )
