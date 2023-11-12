# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.numpy_helper import create_random_int


class BitwiseNot(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "BitwiseNot",
            inputs=["x"],
            outputs=["bitwise_not"],
        )

        # 2d
        x = create_random_int((3, 4), np.int32)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_2d")

        # 3d
        x = create_random_int((3, 4, 5), np.uint16)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_3d")

        # 4d
        x = create_random_int((3, 4, 5, 6), np.uint8)
        y = np.bitwise_not(x)
        expect(node, inputs=[x], outputs=[y], name="test_bitwise_not_4d")
