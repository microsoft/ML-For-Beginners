# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Concat(Base):
    @staticmethod
    def export() -> None:
        test_cases: Dict[str, Sequence[Any]] = {
            "1d": ([1, 2], [3, 4]),
            "2d": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            "3d": (
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
            ),
        }

        for test_case, values_ in test_cases.items():
            values = [np.asarray(v, dtype=np.float32) for v in values_]
            for i in range(len(values[0].shape)):
                in_args = ["value" + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    "Concat", inputs=list(in_args), outputs=["output"], axis=i
                )
                output = np.concatenate(values, i)
                expect(
                    node,
                    inputs=list(values),
                    outputs=[output],
                    name="test_concat_" + test_case + "_axis_" + str(i),
                )

            for i in range(-len(values[0].shape), 0):
                in_args = ["value" + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    "Concat", inputs=list(in_args), outputs=["output"], axis=i
                )
                output = np.concatenate(values, i)
                expect(
                    node,
                    inputs=list(values),
                    outputs=[output],
                    name="test_concat_" + test_case + "_axis_negative_" + str(abs(i)),
                )
