# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class CumSum(Base):
    @staticmethod
    def export_cumsum_1d() -> None:
        node = onnx.helper.make_node("CumSum", inputs=["x", "axis"], outputs=["y"])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.int32(0)
        y = np.array([1.0, 3.0, 6.0, 10.0, 15.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_1d")

    @staticmethod
    def export_cumsum_1d_exclusive() -> None:
        node = onnx.helper.make_node(
            "CumSum", inputs=["x", "axis"], outputs=["y"], exclusive=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.int32(0)
        y = np.array([0.0, 1.0, 3.0, 6.0, 10.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_1d_exclusive")

    @staticmethod
    def export_cumsum_1d_reverse() -> None:
        node = onnx.helper.make_node(
            "CumSum", inputs=["x", "axis"], outputs=["y"], reverse=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.int32(0)
        y = np.array([15.0, 14.0, 12.0, 9.0, 5.0]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_1d_reverse")

    @staticmethod
    def export_cumsum_1d_reverse_exclusive() -> None:
        node = onnx.helper.make_node(
            "CumSum", inputs=["x", "axis"], outputs=["y"], reverse=1, exclusive=1
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float64)
        axis = np.int32(0)
        y = np.array([14.0, 12.0, 9.0, 5.0, 0.0]).astype(np.float64)
        expect(
            node, inputs=[x, axis], outputs=[y], name="test_cumsum_1d_reverse_exclusive"
        )

    @staticmethod
    def export_cumsum_2d_axis_0() -> None:
        node = onnx.helper.make_node(
            "CumSum",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.int32(0)
        y = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_2d_axis_0")

    @staticmethod
    def export_cumsum_2d_axis_1() -> None:
        node = onnx.helper.make_node(
            "CumSum",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.int32(1)
        y = np.array([1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_2d_axis_1")

    @staticmethod
    def export_cumsum_2d_negative_axis() -> None:
        node = onnx.helper.make_node(
            "CumSum",
            inputs=["x", "axis"],
            outputs=["y"],
        )
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64).reshape((2, 3))
        axis = np.int32(-1)
        y = np.array([1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y], name="test_cumsum_2d_negative_axis")
