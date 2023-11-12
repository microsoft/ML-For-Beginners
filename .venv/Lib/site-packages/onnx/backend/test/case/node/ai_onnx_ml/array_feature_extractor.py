# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ArrayFeatureExtractor(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "ArrayFeatureExtractor",
            inputs=["x", "y"],
            outputs=["z"],
            domain="ai.onnx.ml",
        )

        x = np.arange(12).reshape((3, 4)).astype(np.float32)
        y = np.array([0, 1], dtype=np.int64)
        z = np.array([[0, 4, 8], [1, 5, 9]], dtype=np.float32).T
        expect(
            node,
            inputs=[x, y],
            outputs=[z],
            name="test_ai_onnx_ml_array_feature_extractor",
        )
