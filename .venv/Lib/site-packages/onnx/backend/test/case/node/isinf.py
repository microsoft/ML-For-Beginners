# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class IsInf(Base):
    @staticmethod
    def export_infinity() -> None:
        node = onnx.helper.make_node(
            "IsInf",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
        y = np.isinf(x)
        expect(node, inputs=[x], outputs=[y], name="test_isinf")

    @staticmethod
    def export_positive_infinity_only() -> None:
        node = onnx.helper.make_node(
            "IsInf", inputs=["x"], outputs=["y"], detect_negative=0
        )

        x = np.array([-1.7, np.nan, np.inf, 3.6, np.NINF, np.inf], dtype=np.float32)
        y = np.isposinf(x)
        expect(node, inputs=[x], outputs=[y], name="test_isinf_positive")

    @staticmethod
    def export_negative_infinity_only() -> None:
        node = onnx.helper.make_node(
            "IsInf", inputs=["x"], outputs=["y"], detect_positive=0
        )

        x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf], dtype=np.float32)
        y = np.isneginf(x)
        expect(node, inputs=[x], outputs=[y], name="test_isinf_negative")
