# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class BitShift(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, np.right_shift, onnx_node, run_params)
        if self.direction not in ("LEFT", "RIGHT"):  # type: ignore
            raise ValueError(f"Unexpected value for direction ({self.direction!r}).")  # type: ignore
        if self.direction == "LEFT":  # type: ignore
            self.numpy_fct = np.left_shift
