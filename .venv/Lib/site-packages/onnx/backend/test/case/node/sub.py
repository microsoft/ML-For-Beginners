# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Sub(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Sub",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([3, 2, 1]).astype(np.float32)
        z = x - y  # expected output [-2., 0., 2.]
        expect(node, inputs=[x, y], outputs=[z], name="test_sub_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z], name="test_sub")

        x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z], name="test_sub_uint8")

    @staticmethod
    def export_sub_broadcast() -> None:
        node = onnx.helper.make_node(
            "Sub",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z], name="test_sub_bcast")
