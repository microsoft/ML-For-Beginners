# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Upsample(OpRun):
    def _run(self, x, scale, mode=None):  # type: ignore
        if mode == "nearest" and scale.astype(np.int64).tolist() == scale.tolist():
            r = x
            for axis, s in enumerate(scale):
                if s == 1:
                    continue
                r = np.repeat(r, int(s), axis=axis)
            return (r,)
        raise RuntimeError(f"Not implemented for mode={mode!r} and scale={scale!r}.")
