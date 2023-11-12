# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Selu(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Selu", inputs=["x"], outputs=["y"], alpha=2.0, gamma=3.0
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-3.79272318, 0., 3.]
        y = (
            np.clip(x, 0, np.inf) * 3.0
            + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
        )
        expect(node, inputs=[x], outputs=[y], name="test_selu_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = (
            np.clip(x, 0, np.inf) * 3.0
            + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
        )
        expect(node, inputs=[x], outputs=[y], name="test_selu")

    @staticmethod
    def export_selu_default() -> None:
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875
        node = onnx.helper.make_node(
            "Selu",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = (
            np.clip(x, 0, np.inf) * default_gamma
            + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
        )
        expect(node, inputs=[x], outputs=[y], name="test_selu_default")
