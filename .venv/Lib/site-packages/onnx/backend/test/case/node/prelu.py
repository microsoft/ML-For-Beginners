# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class PRelu(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "PRelu",
            inputs=["x", "slope"],
            outputs=["y"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y], name="test_prelu_example")

    @staticmethod
    def export_prelu_broadcast() -> None:
        node = onnx.helper.make_node(
            "PRelu",
            inputs=["x", "slope"],
            outputs=["y"],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y], name="test_prelu_broadcast")
