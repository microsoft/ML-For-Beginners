# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0622,W0622,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Clip_6(OpRun):
    def _run(self, data, min=None, max=None):  # type: ignore
        amin = min
        amax = max
        if amin is None and amax is None:
            return (data,)
        res = np.clip(data, amin, amax)  # type: ignore
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


class Clip_11(OpRun):
    def _run(self, data, *minmax):  # type: ignore
        le = len(minmax)
        amin = minmax[0] if le > 0 else None
        amax = minmax[1] if le > 1 else None
        if amin is None and amax is None:
            return (data,)
        res = np.clip(data, amin, amax)
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)
