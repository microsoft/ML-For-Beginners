# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Mod(OpRun):
    def _run(self, a, b, fmod=None):  # type: ignore
        fmod = fmod or self.fmod  # type: ignore
        if fmod == 1:  # type: ignore
            return (np.fmod(a, b),)
        if a.dtype in (np.float16, np.float32, np.float64):
            return (np.nan_to_num(np.fmod(a, b)),)
        return (np.nan_to_num(np.mod(a, b)),)
