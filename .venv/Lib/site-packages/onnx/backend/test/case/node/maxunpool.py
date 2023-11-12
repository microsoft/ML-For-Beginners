# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class MaxUnpool(Base):
    @staticmethod
    def export_without_output_shape() -> None:
        node = onnx.helper.make_node(
            "MaxUnpool",
            inputs=["xT", "xI"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
        xT = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        xI = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
        y = np.array(
            [[[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[xT, xI],
            outputs=[y],
            name="test_maxunpool_export_without_output_shape",
        )

    @staticmethod
    def export_with_output_shape() -> None:
        node = onnx.helper.make_node(
            "MaxUnpool",
            inputs=["xT", "xI", "output_shape"],
            outputs=["y"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        )
        xT = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
        xI = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
        output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
        y = np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 5, 0, 6, 0],
                        [0, 0, 0, 0, 0],
                        [0, 7, 0, 8, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[xT, xI, output_shape],
            outputs=[y],
            name="test_maxunpool_export_with_output_shape",
        )
