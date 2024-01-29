# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class DepthToSpace(OpRun):
    def _run(self, data, blocksize=None, mode=None):  # type: ignore
        if len(data.shape) != 4:
            raise RuntimeError(f"Unexpected shape {data.shape!r}.")
        b, c, h, w = data.shape
        if mode == "DCR":
            tmpshape = (
                b,
                blocksize,
                blocksize,
                c // (blocksize * blocksize),
                h,
                w,
            )
            reshaped = data.reshape(tmpshape)
            transposed = np.transpose(reshaped, [0, 3, 4, 1, 5, 2])
        else:
            # assert mode == "CRD"
            tmpshape = (
                b,
                c // (blocksize * blocksize),
                blocksize,
                blocksize,
                h,
                w,
            )
            reshaped = data.reshape(tmpshape)
            transposed = np.transpose(reshaped, [0, 1, 4, 2, 5, 3])
        finalshape = (
            b,
            c // (blocksize * blocksize),
            h * blocksize,
            w * blocksize,
        )
        y = np.reshape(transposed, finalshape)
        return (y,)
