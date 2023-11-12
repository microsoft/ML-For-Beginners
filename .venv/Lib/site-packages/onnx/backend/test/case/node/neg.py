# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Neg(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Neg",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.negative(x)  # expected output [4., -2.],
        expect(node, inputs=[x], outputs=[y], name="test_neg_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.negative(x)
        expect(node, inputs=[x], outputs=[y], name="test_neg")
