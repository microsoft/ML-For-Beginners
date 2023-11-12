# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Mul(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Mul",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = x * y  # expected output [4., 10., 18.]
        expect(node, inputs=[x, y], outputs=[z], name="test_mul_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name="test_mul")

        x = np.random.randint(4, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name="test_mul_uint8")

    @staticmethod
    def export_mul_broadcast() -> None:
        node = onnx.helper.make_node(
            "Mul",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z], name="test_mul_bcast")
