# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Size(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Size",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ).astype(np.float32)
        y = np.array(6).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name="test_size_example")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.size).astype(np.int64)

        expect(node, inputs=[x], outputs=[y], name="test_size")
