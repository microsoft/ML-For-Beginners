# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DynamicQuantizeLinear(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
        )

        # expected scale 0.0196078438 and zero point 153
        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear",
        )

        # expected scale 0.0156862754 and zero point 255
        X = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_max_adjusted",
        )

        X = (
            np.array([1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345])
            .astype(np.float32)
            .reshape((3, 4))
        )

        # expected scale 0.0156862754 and zero point 0
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_min_adjusted",
        )
