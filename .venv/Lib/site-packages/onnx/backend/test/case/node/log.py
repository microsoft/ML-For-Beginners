# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Log(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Log",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([1, 10]).astype(np.float32)
        y = np.log(x)  # expected output [0., 2.30258512]
        expect(node, inputs=[x], outputs=[y], name="test_log_example")

        x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.log(x)
        expect(node, inputs=[x], outputs=[y], name="test_log")
