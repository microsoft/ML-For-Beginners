# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class SequenceLength(OpRun):
    def _run(self, input_sequence):  # type: ignore
        if not isinstance(input_sequence, list):
            raise TypeError(
                f"input_sequence must be a list not {type(input_sequence)}."
            )
        return (np.array(len(input_sequence), dtype=np.int64),)
