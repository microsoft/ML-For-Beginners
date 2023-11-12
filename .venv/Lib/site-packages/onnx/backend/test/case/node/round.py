# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Round(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Round",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array(
            [
                0.1,
                0.5,
                0.9,
                1.2,
                1.5,
                1.8,
                2.3,
                2.5,
                2.7,
                -1.1,
                -1.5,
                -1.9,
                -2.2,
                -2.5,
                -2.8,
            ]
        ).astype(np.float32)
        y = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                -1.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -3.0,
            ]
        ).astype(
            np.float32
        )  # expected output
        expect(node, inputs=[x], outputs=[y], name="test_round")
