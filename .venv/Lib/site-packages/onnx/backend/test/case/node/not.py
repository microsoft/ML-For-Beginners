# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Not(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Not",
            inputs=["x"],
            outputs=["not"],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_2d")

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_3d")

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
        expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_4d")
