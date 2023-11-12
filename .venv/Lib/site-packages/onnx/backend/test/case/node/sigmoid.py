# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Sigmoid(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Sigmoid",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = 1.0 / (
            1.0 + np.exp(np.negative(x))
        )  # expected output [0.26894143, 0.5, 0.7310586]
        expect(node, inputs=[x], outputs=[y], name="test_sigmoid_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = 1.0 / (1.0 + np.exp(np.negative(x)))
        expect(node, inputs=[x], outputs=[y], name="test_sigmoid")
