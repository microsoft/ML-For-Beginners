# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Size(OpRun):
    def _run(self, data):  # type: ignore
        return (np.array(data.size, dtype=np.int64),)
