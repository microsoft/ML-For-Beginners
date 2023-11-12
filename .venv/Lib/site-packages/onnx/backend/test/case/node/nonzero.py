# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class NonZero(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "NonZero",
            inputs=["condition"],
            outputs=["result"],
        )

        condition = np.array([[1, 0], [1, 1]], dtype=bool)
        result = np.array(
            np.nonzero(condition), dtype=np.int64
        )  # expected output [[0, 1, 1], [0, 0, 1]]
        expect(node, inputs=[condition], outputs=[result], name="test_nonzero_example")
