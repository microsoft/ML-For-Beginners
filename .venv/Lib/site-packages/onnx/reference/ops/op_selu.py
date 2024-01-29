# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Selu(OpRun):
    def _run(self, x, alpha=None, gamma=None):  # type: ignore
        return (
            (np.where(x > 0, x, np.exp(x) * alpha - alpha) * gamma).astype(x.dtype),
        )
