# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, np.divide, onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        res = OpRunBinaryNumpy._run(self, a, b)
        if res[0].dtype != a.dtype:
            return (res[0].astype(a.dtype),)
        return res
