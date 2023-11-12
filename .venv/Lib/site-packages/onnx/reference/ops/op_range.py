# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Range(OpRun):
    def _run(self, starts, ends, steps):  # type: ignore
        return (np.arange(starts, ends, steps).astype(starts.dtype),)
