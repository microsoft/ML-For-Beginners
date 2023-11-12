# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class GlobalMaxPool(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.random.randn(1, 3, 5, 5).astype(np.float32)
        y = np.max(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool")

    @staticmethod
    def export_globalmaxpool_precomputed() -> None:
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array(
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ]
            ]
        ).astype(np.float32)
        y = np.array([[[[9]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool_precomputed")
