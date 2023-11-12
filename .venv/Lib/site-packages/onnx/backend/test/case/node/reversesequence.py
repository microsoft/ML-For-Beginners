# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReverseSequence(Base):
    @staticmethod
    def export_reversesequence_time() -> None:
        node = onnx.helper.make_node(
            "ReverseSequence",
            inputs=["x", "sequence_lens"],
            outputs=["y"],
            time_axis=0,
            batch_axis=1,
        )
        x = np.array(
            [
                [0.0, 4.0, 8.0, 12.0],
                [1.0, 5.0, 9.0, 13.0],
                [2.0, 6.0, 10.0, 14.0],
                [3.0, 7.0, 11.0, 15.0],
            ],
            dtype=np.float32,
        )
        sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)

        y = np.array(
            [
                [3.0, 6.0, 9.0, 12.0],
                [2.0, 5.0, 8.0, 13.0],
                [1.0, 4.0, 10.0, 14.0],
                [0.0, 7.0, 11.0, 15.0],
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[x, sequence_lens],
            outputs=[y],
            name="test_reversesequence_time",
        )

    @staticmethod
    def export_reversesequence_batch() -> None:
        node = onnx.helper.make_node(
            "ReverseSequence",
            inputs=["x", "sequence_lens"],
            outputs=["y"],
            time_axis=1,
            batch_axis=0,
        )
        x = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )
        sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)

        y = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [5.0, 4.0, 6.0, 7.0],
                [10.0, 9.0, 8.0, 11.0],
                [15.0, 14.0, 13.0, 12.0],
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[x, sequence_lens],
            outputs=[y],
            name="test_reversesequence_batch",
        )
