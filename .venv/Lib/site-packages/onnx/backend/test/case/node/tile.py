# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Tile(Base):
    @staticmethod
    def export_tile() -> None:
        node = onnx.helper.make_node("Tile", inputs=["x", "y"], outputs=["z"])

        x = np.random.rand(2, 3, 4, 5).astype(np.float32)

        repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)

        z = np.tile(x, repeats)

        expect(node, inputs=[x, repeats], outputs=[z], name="test_tile")

    @staticmethod
    def export_tile_precomputed() -> None:
        node = onnx.helper.make_node("Tile", inputs=["x", "y"], outputs=["z"])

        x = np.array([[0, 1], [2, 3]], dtype=np.float32)

        repeats = np.array([2, 2], dtype=np.int64)

        z = np.array(
            [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=np.float32
        )

        expect(node, inputs=[x, repeats], outputs=[z], name="test_tile_precomputed")
