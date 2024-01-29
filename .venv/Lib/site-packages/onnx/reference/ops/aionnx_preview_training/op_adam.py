# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining


def _apply_adam(  # type: ignore
    r, t, x, g, v, h, norm_coefficient, norm_coefficient_post, alpha, beta, epsilon
):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Update momentum.
    v_new = alpha * v + (1 - alpha) * g_regularized
    # Update second-order momentum.
    h_new = beta * h + (1 - beta) * (g_regularized * g_regularized)
    # Compute element-wise square root.
    h_sqrt = np.sqrt(h_new) + epsilon
    # Adjust learning rate.
    r_adjusted = None
    if t > 0:
        # Consider bias correction on momentums.
        r_adjusted = r * np.sqrt(1 - beta**t) / (1 - alpha**t)
    else:
        # No bias correction on momentums.
        r_adjusted = r
    # Apply Adam update rule.
    x_new = x - r_adjusted * (v_new / h_sqrt)
    # It's possible to apply regularization in the end.
    x_final = (1 - norm_coefficient_post) * x_new
    return x_final, v_new, h_new


class Adam(OpRunTraining):
    def _run(  # type: ignore
        self,
        *data,
        alpha=None,
        beta=None,
        epsilon=None,
        norm_coefficient=None,
        norm_coefficient_post=None,
    ):
        if len(data) == 6:
            return self._run1(  # type: ignore
                *data,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                norm_coefficient=norm_coefficient,
                norm_coefficient_post=norm_coefficient_post,
            )
        n = (len(data) - 2) // 4
        xs = []
        vs = []
        hs = []
        for i in range(0, n):
            a, b, c = self._run1(  # type: ignore
                *data[:2],
                data[2 + i],
                data[2 + n + i],
                data[2 + n * 2 + i],
                data[2 + n * 3 + i],
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                norm_coefficient=norm_coefficient,
                norm_coefficient_post=norm_coefficient_post,
            )
            xs.append(a.astype(np.float32))
            vs.append(b.astype(np.float32))
            hs.append(c.astype(np.float32))
        return tuple(xs + vs + hs)

    def _run1(  # type: ignore
        self,
        r,
        t,
        x,
        g,
        v,
        h,
        alpha=None,
        beta=None,
        epsilon=None,
        norm_coefficient=None,
        norm_coefficient_post=None,
    ):
        x_new, v_new, h_new = _apply_adam(
            r,
            t,
            x,
            g,
            v,
            h,
            norm_coefficient,
            norm_coefficient_post,
            alpha,
            beta,
            epsilon,
        )
        return x_new, v_new, h_new
