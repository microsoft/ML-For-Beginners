# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Reciprocal(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Reciprocal",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.reciprocal(x)  # expected output [-0.25, 0.5],
        expect(node, inputs=[x], outputs=[y], name="test_reciprocal_example")

        x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
        y = np.reciprocal(x)
        expect(node, inputs=[x], outputs=[y], name="test_reciprocal")
