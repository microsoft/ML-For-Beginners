# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LeakyRelu(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "LeakyRelu", inputs=["x"], outputs=["y"], alpha=0.1
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-0.1, 0., 1.]
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
        expect(node, inputs=[x], outputs=[y], name="test_leakyrelu_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
        expect(node, inputs=[x], outputs=[y], name="test_leakyrelu")

    @staticmethod
    def export_leakyrelu_default() -> None:
        default_alpha = 0.01
        node = onnx.helper.make_node(
            "LeakyRelu",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * default_alpha
        expect(node, inputs=[x], outputs=[y], name="test_leakyrelu_default")
