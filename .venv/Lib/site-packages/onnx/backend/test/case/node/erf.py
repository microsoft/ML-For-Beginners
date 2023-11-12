# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Erf(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Erf",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        y = np.vectorize(math.erf)(x).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_erf")
