# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining


def _apply_adagrad(r, t, x, g, h, norm_coefficient, epsilon, decay_factor):  # type: ignore
    # Compute adjusted learning-rate.
    r_ = r / (1 + t * decay_factor)
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Update squared accumulated gradient.
    h_new = h + g_regularized * g_regularized
    # Compute ADAGRAD's gradient scaling factors
    h_sqrt = np.sqrt(h_new) + epsilon
    # Apply ADAGRAD update rule.
    x_new = x - r_ * g_regularized / h_sqrt
    return (x_new, h_new)


class Adagrad(OpRunTraining):
    def _run(self, *data, decay_factor=None, epsilon=None, norm_coefficient=None):  # type: ignore
        if len(data) == 5:
            return self._run1(  # type: ignore
                *data,
                decay_factor=decay_factor,
                epsilon=epsilon,
                norm_coefficient=norm_coefficient,
            )
        n = (len(data) - 2) // 3
        xs = []
        hs = []
        for i in range(0, n):
            a, b = self._run1(  # type: ignore
                *data[:2],
                data[2 + i],
                data[2 + n + i],
                data[2 + n * 2 + i],
                decay_factor=decay_factor,
                epsilon=epsilon,
                norm_coefficient=norm_coefficient,
            )
            xs.append(a)
            hs.append(b)
        return tuple(xs + hs)

    def _run1(self, r, t, x, g, h, decay_factor=None, epsilon=None, norm_coefficient=None):  # type: ignore
        x_new, h_new = _apply_adagrad(
            r, t, x, g, h, norm_coefficient, epsilon, decay_factor  # type: ignore
        )
        return x_new, h_new
