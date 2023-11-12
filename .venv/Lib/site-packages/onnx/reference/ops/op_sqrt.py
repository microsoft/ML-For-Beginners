# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from warnings import catch_warnings, simplefilter

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Sqrt(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        with catch_warnings():
            simplefilter("ignore")
            return (np.sqrt(x).astype(x.dtype),)
