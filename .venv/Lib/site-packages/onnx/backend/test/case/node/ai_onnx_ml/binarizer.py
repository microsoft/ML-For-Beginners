# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.aionnxml.op_binarizer import compute_binarizer


class Binarizer(Base):
    @staticmethod
    def export() -> None:
        threshold = 1.0
        node = onnx.helper.make_node(
            "Binarizer",
            inputs=["X"],
            outputs=["Y"],
            threshold=threshold,
            domain="ai.onnx.ml",
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = compute_binarizer(x, threshold)[0]

        expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_binarizer")
