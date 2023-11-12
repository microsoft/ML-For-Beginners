# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Greater(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GreaterOrEqual",
            inputs=["x", "y"],
            outputs=["greater_equal"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal")

    @staticmethod
    def export_greater_broadcast() -> None:
        node = onnx.helper.make_node(
            "GreaterOrEqual",
            inputs=["x", "y"],
            outputs=["greater_equal"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal_bcast")
