# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class BitShift(Base):
    @staticmethod
    def export_right_unit8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint8")

    @staticmethod
    def export_right_unit16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint16")

    @staticmethod
    def export_right_unit32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint32")

    @staticmethod
    def export_right_unit64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="RIGHT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x >> y  # expected output [8, 1, 0]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_right_uint64")

    @staticmethod
    def export_left_unit8() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint8)
        y = np.array([1, 2, 3]).astype(np.uint8)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint8")

    @staticmethod
    def export_left_unit16() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint16)
        y = np.array([1, 2, 3]).astype(np.uint16)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint16")

    @staticmethod
    def export_left_unit32() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint32)
        y = np.array([1, 2, 3]).astype(np.uint32)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint32")

    @staticmethod
    def export_left_unit64() -> None:
        node = onnx.helper.make_node(
            "BitShift", inputs=["x", "y"], outputs=["z"], direction="LEFT"
        )

        x = np.array([16, 4, 1]).astype(np.uint64)
        y = np.array([1, 2, 3]).astype(np.uint64)
        z = x << y  # expected output [32, 16, 8]
        expect(node, inputs=[x, y], outputs=[z], name="test_bitshift_left_uint64")
