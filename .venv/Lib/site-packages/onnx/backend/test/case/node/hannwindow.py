# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class HannWindow(Base):
    @staticmethod
    def export() -> None:
        # Test periodic window
        node = onnx.helper.make_node(
            "HannWindow",
            inputs=["x"],
            outputs=["y"],
        )
        size = np.int32(10)
        a0 = 0.5
        a1 = 0.5
        y = a0 - a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y], name="test_hannwindow")

        # Test symmetric window
        node = onnx.helper.make_node(
            "HannWindow", inputs=["x"], outputs=["y"], periodic=0
        )
        size = np.int32(10)
        a0 = 0.5
        a1 = 0.5
        y = a0 - a1 * np.cos(
            2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1)
        )
        expect(node, inputs=[size], outputs=[y], name="test_hannwindow_symmetric")
