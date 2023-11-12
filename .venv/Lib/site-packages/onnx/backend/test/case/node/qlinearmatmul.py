# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class QLinearMatMul(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "QLinearMatMul",
            inputs=[
                "a",
                "a_scale",
                "a_zero_point",
                "b",
                "b_scale",
                "b_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            outputs=["y"],
        )

        # 2D
        a = np.array(
            [
                [208, 236, 0, 238],
                [3, 214, 255, 29],
            ],
            dtype=np.uint8,
        )

        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113], dtype=np.uint8)

        b = np.array(
            [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
            dtype=np.uint8,
        )

        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114], dtype=np.uint8)

        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118], dtype=np.uint8)

        output = np.array(
            [
                [168, 115, 255],
                [1, 66, 151],
            ],
            dtype=np.uint8,
        )

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[output],
            name="test_qlinearmatmul_2D",
        )

        # 3D
        a = np.array(
            [
                [[208, 236, 0, 238], [3, 214, 255, 29]],
                [[208, 236, 0, 238], [3, 214, 255, 29]],
            ],
            dtype=np.uint8,
        )

        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113], dtype=np.uint8)

        b = np.array(
            [
                [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
                [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
            ],
            dtype=np.uint8,
        )

        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114], dtype=np.uint8)

        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118], dtype=np.uint8)

        output = np.array(
            [[[168, 115, 255], [1, 66, 151]], [[168, 115, 255], [1, 66, 151]]],
            dtype=np.uint8,
        )

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[output],
            name="test_qlinearmatmul_3D",
        )
