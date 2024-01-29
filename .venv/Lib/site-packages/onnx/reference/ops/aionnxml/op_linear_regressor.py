# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LinearRegressor(OpRunAiOnnxMl):
    def _run(  # type: ignore
        self, x, coefficients=None, intercepts=None, targets=1, post_transform=None
    ):
        coefficients = np.array(coefficients).astype(x.dtype)
        intercepts = np.array(intercepts).astype(x.dtype)
        n = coefficients.shape[0] // targets
        coefficients = coefficients.reshape(targets, n).T
        score = np.dot(x, coefficients)
        if intercepts is not None:
            score += intercepts
        if post_transform == "NONE":
            return (score,)
        raise NotImplementedError(
            f"post_transform: {post_transform!r} is not implemented."
        )
