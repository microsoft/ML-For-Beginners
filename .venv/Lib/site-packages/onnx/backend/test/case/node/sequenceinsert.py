# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def sequence_insert_reference_implementation(
    sequence: List[Any], tensor: np.ndarray, position: np.ndarray = None
) -> List[Any]:
    # make a copy of input sequence
    seq = list(sequence)
    if position is not None:
        # In these cases, insert_position will be between [-len(sequence), len(sequence)]
        # The position argument will be in the format np.array([pos_index])
        insert_position = position[0]
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence.
        seq.append(tensor)
    return seq


class SequenceInsert(Base):
    @staticmethod
    def export() -> None:
        test_cases = {
            "at_back": [np.array([10, 11, 12]).astype(np.int64)],
            "at_front": [np.array([-2, -1, 0]), np.array([0]).astype(np.int64)],
        }
        sequence = [
            np.array([1, 2, 3, 4]).astype(np.int64),
            np.array([5, 6, 7]).astype(np.int64),
            np.array([8, 9]).astype(np.int64),
        ]

        for test_name, test_inputs in test_cases.items():
            tensor = test_inputs[0].astype(np.int64)

            if len(test_inputs) > 1:
                node = onnx.helper.make_node(
                    "SequenceInsert",
                    inputs=["sequence", "tensor", "position"],
                    outputs=["output_sequence"],
                )
                position = test_inputs[1]
                inserted = sequence_insert_reference_implementation(
                    sequence, tensor, position
                )
                expect(
                    node,
                    inputs=[sequence, tensor, position],
                    outputs=[inserted],
                    name="test_sequence_insert_" + test_name,
                )
            else:
                node = onnx.helper.make_node(
                    "SequenceInsert",
                    inputs=["sequence", "tensor"],
                    outputs=["output_sequence"],
                )
                inserted = sequence_insert_reference_implementation(sequence, tensor)
                expect(
                    node,
                    inputs=[sequence, tensor],
                    outputs=[inserted],
                    name="test_sequence_insert_" + test_name,
                )
