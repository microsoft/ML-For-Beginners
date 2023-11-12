# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class QLinearConv(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "QLinearConv",
            inputs=[
                "x",
                "x_scale",
                "x_zero_point",
                "w",
                "w_scale",
                "w_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            outputs=["y"],
        )

        x = np.array(
            [
                [255, 174, 162, 25, 203, 168, 58],
                [15, 59, 237, 95, 129, 0, 64],
                [56, 242, 153, 221, 168, 12, 166],
                [232, 178, 186, 195, 237, 162, 237],
                [188, 39, 124, 77, 80, 102, 43],
                [127, 230, 21, 83, 41, 40, 134],
                [255, 154, 92, 141, 42, 148, 247],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))

        x_scale = np.float32(0.00369204697)
        x_zero_point = np.uint8(132)

        w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))

        w_scale = np.array([0.00172794575], dtype=np.float32)
        w_zero_point = np.array([255], dtype=np.uint8)

        y_scale = np.float32(0.00162681262)
        y_zero_point = np.uint8(123)

        output = np.array(
            [
                [0, 81, 93, 230, 52, 87, 197],
                [240, 196, 18, 160, 126, 255, 191],
                [199, 13, 102, 34, 87, 243, 89],
                [23, 77, 69, 60, 18, 93, 18],
                [67, 216, 131, 178, 175, 153, 212],
                [128, 25, 234, 172, 214, 215, 121],
                [0, 101, 163, 114, 213, 107, 8],
            ],
            dtype=np.uint8,
        ).reshape((1, 1, 7, 7))

        expect(
            node,
            inputs=[
                x,
                x_scale,
                x_zero_point,
                w,
                w_scale,
                w_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[output],
            name="test_qlinearconv",
        )
