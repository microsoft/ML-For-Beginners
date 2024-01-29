# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple

import numpy as np

from onnx.reference.op_run import OpRun


class SplitToSequence(OpRun):
    def common_run(
        self, mat: np.ndarray, split: Optional[np.ndarray], axis: int
    ) -> List[np.ndarray]:
        if split is None:
            split_length = [1 for _ in range(mat.shape[axis])]
        elif len(split.shape) == 0:
            # A scalar
            dim = mat.shape[axis]
            length = int(split)
            n = dim // int(length)
            split_length = [length] * n
            left = dim - length * n
            if left > 0:
                split_length.append(left)
        else:
            split_length = list(split)

        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split_length:
            sli[axis] = slice(pos, pos + spl)  # type: ignore
            pos += spl
            res.append(mat[tuple(sli)])
        return res

    def _run(
        self,
        mat: np.ndarray,
        split: Optional[np.ndarray] = None,
        axis: int = 0,
        keepdims: int = 1,
    ) -> Tuple[np.ndarray]:
        res = self.common_run(mat, split, axis=axis)
        if split is None and not keepdims:
            for i in range(len(res)):
                shape = list(res[i].shape)
                del shape[axis]
                res[i] = res[i].reshape(tuple(shape))
        return (res,)
