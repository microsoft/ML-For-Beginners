# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Celu(Base):
    @staticmethod
    def export() -> None:
        alpha = 2.0
        node = onnx.helper.make_node(
            "Celu",
            inputs=["X"],
            outputs=["Y"],
            alpha=alpha,
        )

        input_data = np.array(
            [
                [
                    [[0.8439683], [0.5665144], [0.05836735]],
                    [[0.02916367], [0.12964272], [0.5060197]],
                    [[0.79538304], [0.9411346], [0.9546573]],
                ],
                [
                    [[0.17730942], [0.46192095], [0.26480448]],
                    [[0.6746842], [0.01665257], [0.62473077]],
                    [[0.9240844], [0.9722341], [0.11965699]],
                ],
                [
                    [[0.41356155], [0.9129373], [0.59330076]],
                    [[0.81929934], [0.7862604], [0.11799799]],
                    [[0.69248444], [0.54119414], [0.07513223]],
                ],
            ],
            dtype=np.float32,
        )

        # Calculate expected output data
        positive_input = np.maximum(0, input_data)
        negative_input = np.minimum(0, alpha * (np.exp(input_data / alpha) - 1))
        expected_output = positive_input + negative_input

        expect(node, inputs=[input_data], outputs=[expected_output], name="test_celu")
