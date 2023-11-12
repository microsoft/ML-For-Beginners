# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.numpy_helper import create_random_int


class BitwiseXor(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseXor",
            inputs=["x", "y"],
            outputs=["bitwisexor"],
        )

        # 2d
        x = create_random_int((3, 4), np.int32)
        y = create_random_int((3, 4), np.int32)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i32_2d")

        # 3d
        x = create_random_int((3, 4, 5), np.int16)
        y = create_random_int((3, 4, 5), np.int16)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i16_3d")

    @staticmethod
    def export_bitwiseor_broadcast() -> None:
        node = onnx.helper.make_node(
            "BitwiseXor",
            inputs=["x", "y"],
            outputs=["bitwisexor"],
        )

        # 3d vs 1d
        x = create_random_int((3, 4, 5), np.uint64)
        y = create_random_int((5,), np.uint64)
        z = np.bitwise_xor(x, y)
        expect(
            node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_ui64_bcast_3v1d"
        )

        # 4d vs 3d
        x = create_random_int((3, 4, 5, 6), np.uint8)
        y = create_random_int((4, 5, 6), np.uint8)
        z = np.bitwise_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_ui8_bcast_4v3d")
