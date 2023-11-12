# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class And(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "And",
            inputs=["x", "y"],
            outputs=["and"],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(bool)
        y = (np.random.randn(3, 4) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and2d")

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and3d")

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and4d")

    @staticmethod
    def export_and_broadcast() -> None:
        node = onnx.helper.make_node(
            "And",
            inputs=["x", "y"],
            outputs=["and"],
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(bool)
        y = (np.random.randn(5) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and_bcast3v1d")

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(bool)
        y = (np.random.randn(4, 5) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and_bcast3v2d")

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
        y = (np.random.randn(5, 6) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and_bcast4v2d")

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and_bcast4v3d")

        # 4d vs 4d
        x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
        y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_and_bcast4v4d")
