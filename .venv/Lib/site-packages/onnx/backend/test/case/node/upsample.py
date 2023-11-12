# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Upsample(Base):
    @staticmethod
    def export_nearest() -> None:
        node = onnx.helper.make_node(
            "Upsample",
            inputs=["X", "scales"],
            outputs=["Y"],
            mode="nearest",
        )

        data = np.array(
            [
                [
                    [
                        [1, 2],
                        [3, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        output = np.array(
            [
                [
                    [
                        [1, 1, 1, 2, 2, 2],
                        [1, 1, 1, 2, 2, 2],
                        [3, 3, 3, 4, 4, 4],
                        [3, 3, 3, 4, 4, 4],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[data, scales],
            outputs=[output],
            name="test_upsample_nearest",
            opset_imports=[helper.make_opsetid("", 9)],
        )
