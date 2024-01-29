# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class FeatureVectorizer(OpRunAiOnnxMl):
    def _preprocess(self, a, cut):  # type: ignore
        if len(a.shape) == 1:
            a = a.reshape((-1, 1))
        if len(a.shape) != 2:
            raise ValueError(f"Every input must have 1 or 2 dimensions not {a.shape}.")
        if cut < a.shape[1]:
            return a[:, :cut]
        if cut > a.shape[1]:
            b = np.zeros((a.shape[0], cut), dtype=a.dtype)
            b[:, : a.shape[1]] = a
            return b
        return a

    def _run(self, *args, inputdimensions=None):  # type: ignore
        args = [  # type: ignore
            self._preprocess(a, axis) for a, axis in zip(args, inputdimensions)
        ]
        res = np.concatenate(args, axis=1)
        return (res,)
