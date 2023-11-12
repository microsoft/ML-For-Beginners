# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor


class QuantizeLinear(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = np.uint8(128)
        y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear",
        )

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array(
            [
                [
                    [[-162, 10], [-100, 232], [-20, -50]],
                    [[-76, 0], [0, 252], [32, -44]],
                    [[245, -485], [-960, -270], [-375, -470]],
                ],
            ],
            dtype=np.float32,
        )
        y_scale = np.array([2, 4, 5], dtype=np.float32)
        y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(
            np.uint8
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_axis",
        )

    @staticmethod
    def export_e4m3fn() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0.0, 1.0, 2.0, 100000.0, 200.0]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = make_tensor("zero_point", TensorProto.FLOAT8E4M3FN, [1], [0])
        y = make_tensor(
            "zero_point", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, 96]
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_e4m3fn",
        )

    @staticmethod
    def export_e5m2() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0.0, 1.0, 2.0, 100000.0, 200.0]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = make_tensor("zero_point", TensorProto.FLOAT8E5M2, [1], [0.0])
        y = make_tensor(
            "zero_point", TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 49152, 96]
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_e5m2",
        )
