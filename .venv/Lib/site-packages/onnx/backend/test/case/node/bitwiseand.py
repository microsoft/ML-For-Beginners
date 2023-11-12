# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.numpy_helper import create_random_int


class BitwiseAnd(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseAnd",
            inputs=["x", "y"],
            outputs=["bitwiseand"],
        )

        # 2d
        x = create_random_int((3, 4), np.int32)
        y = create_random_int((3, 4), np.int32)
        z = np.bitwise_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_and_i32_2d")

        # 3d
        x = create_random_int((3, 4, 5), np.int16)
        y = create_random_int((3, 4, 5), np.int16)
        z = np.bitwise_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_and_i16_3d")

    @staticmethod
    def export_bitwiseand_broadcast() -> None:
        node = onnx.helper.make_node(
            "BitwiseAnd",
            inputs=["x", "y"],
            outputs=["bitwiseand"],
        )

        # 3d vs 1d
        x = create_random_int((3, 4, 5), np.uint64)
        y = create_random_int((5,), np.uint64)
        z = np.bitwise_and(x, y)
        expect(
            node, inputs=[x, y], outputs=[z], name="test_bitwise_and_ui64_bcast_3v1d"
        )

        # 4d vs 3d
        x = create_random_int((3, 4, 5, 6), np.uint8)
        y = create_random_int((4, 5, 6), np.uint8)
        z = np.bitwise_and(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_and_ui8_bcast_4v3d")
