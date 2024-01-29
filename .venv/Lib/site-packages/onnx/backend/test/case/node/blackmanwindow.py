# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class BlackmanWindow(Base):
    @staticmethod
    def export() -> None:
        # Test periodic window
        node = onnx.helper.make_node(
            "BlackmanWindow",
            inputs=["x"],
            outputs=["y"],
        )
        size = np.int32(10)
        a0 = 0.42
        a1 = -0.5
        a2 = 0.08
        y = a0
        y += a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        y += a2 * np.cos(4 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y], name="test_blackmanwindow")

        # Test symmetric window
        node = onnx.helper.make_node(
            "BlackmanWindow", inputs=["x"], outputs=["y"], periodic=0
        )
        size = np.int32(10)
        a0 = 0.42
        a1 = -0.5
        a2 = 0.08
        y = a0
        y += a1 * np.cos(
            2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1)
        )
        y += a2 * np.cos(
            4 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1)
        )
        expect(node, inputs=[size], outputs=[y], name="test_blackmanwindow_symmetric")
