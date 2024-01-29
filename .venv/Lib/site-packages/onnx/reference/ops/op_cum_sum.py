# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class CumSum(OpRun):
    def _run(self, x, *axis, exclusive=None, reverse=None):  # type: ignore
        axis = None if not axis else axis[0]  # type: ignore
        if axis is None:  # type: ignore
            if reverse or exclusive:
                raise NotImplementedError("reverse=1 or exclusive=1 not implemented")
            return (np.cumsum(x),)
        if not isinstance(axis, (np.int32, np.int64)):
            if len(axis.shape) > 1 or (len(axis.shape) > 0 and axis.shape[0] != 1):  # type: ignore
                raise RuntimeError(
                    f"axis must be an array of one number not {axis} (shape {axis.shape})."  # type: ignore
                )
            if len(axis.shape) > 0:  # type: ignore
                axis = axis[0]
        if reverse:
            rev_indices = [slice(0, s) for s in x.shape]
            rev_indices[axis] = slice(None, None, -1)  # type: ignore
            x = x[tuple(rev_indices)]
        if exclusive:
            indices_c = [slice(0, s) for s in x.shape]
            indices_d = [slice(0, s) for s in x.shape]
            indices_c[axis] = slice(0, -1)  # type: ignore
            indices_d[axis] = slice(1, x.shape[axis])  # type: ignore
            res = np.zeros(x.shape, dtype=x.dtype)
            np.cumsum(x[tuple(indices_c)], axis=axis, out=res[tuple(indices_d)])  # type: ignore
        else:
            res = np.cumsum(x, axis=axis)  # type: ignore
        if reverse:
            res = res[tuple(rev_indices)]
        return (res,)
