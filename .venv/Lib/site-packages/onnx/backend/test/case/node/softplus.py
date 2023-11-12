# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Softplus(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Softplus",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.log(
            np.exp(x) + 1
        )  # expected output [0.31326166, 0.69314718, 1.31326163]
        expect(node, inputs=[x], outputs=[y], name="test_softplus_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.log(np.exp(x) + 1)
        expect(node, inputs=[x], outputs=[y], name="test_softplus")
