# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Transpose(OpRun):
    def _run(self, data, perm=None):  # type: ignore
        perm_ = None if (perm is None or len(perm) == 0) else perm
        if perm_ is None:
            return (np.transpose(data),)
        if len(perm_) != len(data.shape):
            raise RuntimeError(
                f"Inconsistent permutation {perm_!r} with shape {data.shape!r}."
            )
        return (np.transpose(data, axes=perm_),)
