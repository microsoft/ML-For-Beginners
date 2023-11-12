# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Exp(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Exp",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.exp(x)  # expected output [0.36787945, 1., 2.71828175]
        expect(node, inputs=[x], outputs=[y], name="test_exp_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.exp(x)
        expect(node, inputs=[x], outputs=[y], name="test_exp")
