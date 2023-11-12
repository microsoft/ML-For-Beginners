# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Asin(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Asin",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y], name="test_asin_example")

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y], name="test_asin")
