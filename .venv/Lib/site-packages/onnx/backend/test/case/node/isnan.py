# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class IsNaN(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "IsNaN",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float32)
        y = np.isnan(x)
        expect(node, inputs=[x], outputs=[y], name="test_isnan")

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "IsNaN",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf], dtype=np.float16)
        y = np.isnan(x)
        expect(node, inputs=[x], outputs=[y], name="test_isnan_float16")
