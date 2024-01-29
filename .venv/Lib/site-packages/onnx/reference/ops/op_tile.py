# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Tile(OpRun):
    def _run(self, x, repeats):  # type: ignore
        return (np.tile(x, repeats),)
