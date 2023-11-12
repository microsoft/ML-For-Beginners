# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.numpy_helper import create_random_int


class BitwiseOr(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseOr",
            inputs=["x", "y"],
            outputs=["bitwiseor"],
        )
        # 2d
        x = create_random_int((3, 4), np.int32)
        y = create_random_int((3, 4), np.int32)
        z = np.bitwise_or(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_i32_2d")

        # 4d
        x = create_random_int((3, 4, 5, 6), np.int8)
        y = create_random_int((3, 4, 5, 6), np.int8)
        z = np.bitwise_or(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_i16_4d")

    @staticmethod
    def export_bitwiseor_broadcast() -> None:
        node = onnx.helper.make_node(
            "BitwiseOr",
            inputs=["x", "y"],
            outputs=["bitwiseor"],
        )

        # 3d vs 1d
        x = create_random_int((3, 4, 5), np.uint64)
        y = create_random_int((5,), np.uint64)
        z = np.bitwise_or(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_ui64_bcast_3v1d")

        # 4d vs 3d
        x = create_random_int((3, 4, 5, 6), np.uint8)
        y = create_random_int((4, 5, 6), np.uint8)
        z = np.bitwise_or(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_ui8_bcast_4v3d")
