# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class Max(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, np.maximum, onnx_node, run_params)

    def run(self, *data):  # type: ignore
        if len(data) == 2:
            return OpRunBinaryNumpy.run(self, *data)
        if len(data) == 1:
            return (data[0].copy(),)
        if len(data) > 2:
            a = data[0]
            for i in range(1, len(data)):
                a = np.maximum(a, data[i])
            return (a,)
        raise RuntimeError("Unexpected turn of events.")
