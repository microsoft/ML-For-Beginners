# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Normalizer(OpRunAiOnnxMl):
    @staticmethod
    def norm_max(x):  # type: ignore
        "max normalization"
        div = np.abs(x).max(axis=1).reshape((x.shape[0], -1))
        return x / np.maximum(div, 1e-30)

    @staticmethod
    def norm_l1(x):  # type: ignore
        "L1 normalization"
        div = np.abs(x).sum(axis=1).reshape((x.shape[0], -1))
        return x / np.maximum(div, 1e-30)

    @staticmethod
    def norm_l2(x):  # type: ignore
        "L2 normalization"
        xn = np.square(x).sum(axis=1)
        np.sqrt(xn, out=xn)
        norm = np.maximum(xn.reshape((x.shape[0], -1)), 1e-30)
        return x / norm

    def _run(self, x, norm=None):  # type: ignore
        if norm == "MAX":
            _norm = Normalizer.norm_max
        elif norm == "L1":
            _norm = Normalizer.norm_l1
        elif norm == "L2":
            _norm = Normalizer.norm_l2
        else:
            raise ValueError(f"Unexpected value for norm='{norm}'.")
        return (_norm(x),)
