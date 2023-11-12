# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Sign(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Sign",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array(range(-5, 6)).astype(np.float32)
        y = np.sign(x)
        expect(node, inputs=[x], outputs=[y], name="test_sign")
