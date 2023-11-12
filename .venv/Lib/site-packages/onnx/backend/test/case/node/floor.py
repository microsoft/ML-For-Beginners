# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Floor(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Floor",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1.5, 1.2, 2]).astype(np.float32)
        y = np.floor(x)  # expected output [-2., 1., 2.]
        expect(node, inputs=[x], outputs=[y], name="test_floor_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.floor(x)
        expect(node, inputs=[x], outputs=[y], name="test_floor")
