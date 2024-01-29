# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from warnings import catch_warnings, simplefilter

import numpy as np

from onnx.reference.op_run import OpRun


class Pow(OpRun):
    def _run(self, a, b):  # type: ignore
        with catch_warnings():
            simplefilter("ignore")
            return (np.power(a, b).astype(a.dtype),)
