# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Div(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Div",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([3, 4]).astype(np.float32)
        y = np.array([1, 2]).astype(np.float32)
        z = x / y  # expected output [3., 2.]
        expect(node, inputs=[x, y], outputs=[z], name="test_div_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z], name="test_div")

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8) + 1
        z = x // y
        expect(node, inputs=[x, y], outputs=[z], name="test_div_uint8")

    @staticmethod
    def export_div_broadcast() -> None:
        node = onnx.helper.make_node(
            "Div",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z], name="test_div_bcast")
