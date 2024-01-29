# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class HammingWindow(Base):
    @staticmethod
    def export() -> None:
        # Test periodic window
        node = onnx.helper.make_node(
            "HammingWindow",
            inputs=["x"],
            outputs=["y"],
        )
        size = np.int32(10)
        a0 = 25 / 46
        a1 = 1 - a0
        y = a0 - a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y], name="test_hammingwindow")

        # Test symmetric window
        node = onnx.helper.make_node(
            "HammingWindow", inputs=["x"], outputs=["y"], periodic=0
        )
        size = np.int32(10)
        a0 = 25 / 46
        a1 = 1 - a0
        y = a0 - a1 * np.cos(
            2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1)
        )
        expect(node, inputs=[size], outputs=[y], name="test_hammingwindow_symmetric")
