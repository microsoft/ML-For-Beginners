# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from onnx.reference.ops.aionnx_preview_training._op_run_training import OpRunTraining


def _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply SG with momentum update rule.
    x_new = x - r * v_new
    return x_new, v_new


def _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply Nesterov with momentum update rule.
    x_new = x - r * (g_regularized + alpha * v_new)
    return x_new, v_new


class Momentum(OpRunTraining):
    def _run(self, *data, alpha=None, beta=None, mode=None, norm_coefficient=None):  # type: ignore
        if len(data) == 5:
            r, t, x, g, v = data
            return self._run1(  # type: ignore
                r,
                t,
                x,
                g,
                v,
                norm_coefficient=norm_coefficient,
                alpha=alpha,
                beta=beta,
                mode=mode,
            )
        n = (len(data) - 2) // 3
        xs = []
        vs = []
        for i in range(0, n):
            a, b = self._run1(  # type: ignore
                *data[:2],
                data[2 + i],
                data[2 + n + i],
                data[2 + n * 2 + i],
                norm_coefficient=norm_coefficient,
                alpha=alpha,
                beta=beta,
                mode=mode,
            )
            xs.append(a)
            vs.append(b)
        return tuple(xs + vs)

    def _run1(self, r, t, x, g, v, mode="standard", norm_coefficient=None, alpha=None, beta=None):  # type: ignore
        if mode == "standard":
            x_new, v_new = _apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
        else:
            x_new, v_new = _apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)
        return x_new, v_new
