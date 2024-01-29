# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor


class DequantizeLinear(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        # scalar zero point and scale
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.float32(2)
        x_zero_point = np.uint8(128)
        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear",
        )

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        # 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
        x = np.array(
            [
                [
                    [[3, 89], [34, 200], [74, 59]],
                    [[5, 24], [24, 87], [32, 13]],
                    [[245, 99], [4, 142], [121, 102]],
                ],
            ],
            dtype=np.uint8,
        )
        x_scale = np.array([2, 4, 5], dtype=np.float32)
        x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (
            x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)
        ) * x_scale.reshape(1, 3, 1, 1)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_axis",
        )

    @staticmethod
    def export_e4m3fn() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn",
        )

    @staticmethod
    def export_e4m3fn_float16() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        x_scale = np.float16(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float16)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn_float16",
        )

    @staticmethod
    def export_e4m3fn_zero_point() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "zero_point"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        zero_point = make_tensor("zero_point", TensorProto.FLOAT8E4M3FN, [1], [0])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, zero_point],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn_zero_point",
        )

    @staticmethod
    def export_e5m2() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 49152, -96])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 98304.0, -192.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e5m2",
        )
