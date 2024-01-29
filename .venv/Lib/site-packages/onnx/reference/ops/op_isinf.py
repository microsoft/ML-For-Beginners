# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class IsInf(OpRun):
    def _run(self, data, detect_negative=None, detect_positive=None):  # type: ignore
        if detect_negative:
            if detect_positive:
                return (np.isinf(data),)
            return (np.isneginf(data),)
        if detect_positive:
            return (np.isposinf(data),)
        res = np.full(data.shape, dtype=np.bool_, fill_value=False)
        return (res,)
