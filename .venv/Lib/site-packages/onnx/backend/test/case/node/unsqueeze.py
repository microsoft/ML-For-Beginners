# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Unsqueeze(Base):
    @staticmethod
    def export_unsqueeze_one_axis() -> None:
        x = np.random.randn(3, 4, 5).astype(np.float32)

        for i in range(x.ndim):
            axes = np.array([i]).astype(np.int64)
            node = onnx.helper.make_node(
                "Unsqueeze",
                inputs=["x", "axes"],
                outputs=["y"],
            )
            y = np.expand_dims(x, axis=i)

            expect(
                node,
                inputs=[x, axes],
                outputs=[y],
                name="test_unsqueeze_axis_" + str(i),
            )

    @staticmethod
    def export_unsqueeze_two_axes() -> None:
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([1, 4]).astype(np.int64)

        node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        y = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=4)

        expect(node, inputs=[x, axes], outputs=[y], name="test_unsqueeze_two_axes")

    @staticmethod
    def export_unsqueeze_three_axes() -> None:
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([2, 4, 5]).astype(np.int64)

        node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x, axes], outputs=[y], name="test_unsqueeze_three_axes")

    @staticmethod
    def export_unsqueeze_unsorted_axes() -> None:
        x = np.random.randn(3, 4, 5).astype(np.float32)
        axes = np.array([5, 4, 2]).astype(np.int64)

        node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x, axes], outputs=[y], name="test_unsqueeze_unsorted_axes")

    @staticmethod
    def export_unsqueeze_negative_axes() -> None:
        node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        x = np.random.randn(1, 3, 1, 5).astype(np.float32)
        axes = np.array([-2]).astype(np.int64)
        y = np.expand_dims(x, axis=-2)
        expect(node, inputs=[x, axes], outputs=[y], name="test_unsqueeze_negative_axes")
