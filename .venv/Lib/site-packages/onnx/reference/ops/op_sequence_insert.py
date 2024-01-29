# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Any, List, Optional, Union

import numpy as np

from onnx.reference.op_run import OpRun


def sequence_insert_reference_implementation(
    sequence: Union[List[Any], np.ndarray],
    tensor: np.ndarray,
    position: Optional[np.ndarray] = None,
) -> List[Any]:
    # make a copy of input sequence
    seq: List[Any] = []
    if sequence is not None and (
        not isinstance(sequence, np.ndarray) or len(sequence.shape) > 0
    ):
        try:
            seq.extend(sequence)
        except TypeError as e:
            raise TypeError(
                f"Unable to iterate on type {type(sequence)}: {sequence}."
            ) from e
    if position is not None:
        # In these cases, insert_position will be between [-len(sequence), len(sequence)]
        # The position argument will be in the format np.array([pos_index])
        insert_position = (position[0] + len(seq)) % len(seq)
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence.
        seq.append(tensor)
    return seq


class SequenceInsert(OpRun):
    def _run(self, S, T, ind=None):  # type: ignore
        if ind is None:
            res = sequence_insert_reference_implementation(S, T)
        elif isinstance(ind, int):
            res = sequence_insert_reference_implementation(S, T, [ind])  # type: ignore[arg-type]
        elif len(ind.shape) > 0:
            res = sequence_insert_reference_implementation(S, T, ind)
        elif len(ind.shape) == 0:
            res = sequence_insert_reference_implementation(S, T, [int(ind)])  # type: ignore[arg-type]
        else:
            res = sequence_insert_reference_implementation(S, T)
        return (res,)
