# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class SpaceToDepth(OpRun):
    def _run(self, data, blocksize=None):  # type: ignore
        if len(data.shape) != 4:
            raise RuntimeError(f"Unexpected shape {data.shape!r}.")
        b, C, H, W = data.shape
        tmpshape = (
            b,
            C,
            H // blocksize,
            blocksize,
            W // blocksize,
            blocksize,
        )
        reshaped = np.reshape(data, tmpshape)
        transposed = np.transpose(reshaped, [0, 3, 5, 1, 2, 4])
        finalshape = (
            b,
            C * blocksize * blocksize,
            H // blocksize,
            W // blocksize,
        )
        y = np.reshape(transposed, finalshape).astype(data.dtype)
        return (y,)
