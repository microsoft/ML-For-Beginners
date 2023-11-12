# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN


def apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply SG with momentum update rule.
    x_new = x - r * v_new
    return x_new, v_new


def apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Coefficient of gradient should be 1 at the first iteration.
    beta_adjusted = beta if t > 0 else 1
    # Update momentum.
    v_new = alpha * v + beta_adjusted * g_regularized
    # Apply Nesterov with momentum update rule.
    x_new = x - r * (g_regularized + alpha * v_new)
    return x_new, v_new


class Momentum(Base):
    @staticmethod
    def export_momentum() -> None:
        # Define operator attributes.
        norm_coefficient = 0.001
        alpha = 0.95
        beta = 0.1

        # Create operator.
        node = onnx.helper.make_node(
            "Momentum",
            inputs=["R", "T", "X", "G", "V"],
            outputs=["X_new", "V_new"],
            norm_coefficient=norm_coefficient,
            alpha=alpha,
            beta=beta,
            mode="standard",
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar
        x = np.array([1.2, 2.8], dtype=np.float32)
        g = np.array([-0.94, -2.5], dtype=np.float32)
        v = np.array([1.7, 3.6], dtype=np.float32)

        # Compute expected outputs of Momentum.
        x_new, v_new = apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)

        # Check results.
        expect(
            node,
            inputs=[r, t, x, g, v],
            outputs=[x_new, v_new],
            name="test_momentum",
            opset_imports=[
                onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)
            ],
        )

    @staticmethod
    def export_nesterov_momentum() -> None:
        # Define operator attributes.
        norm_coefficient = 0.01
        alpha = 0.95
        beta = 1.0

        # Create operator.
        node = onnx.helper.make_node(
            "Momentum",
            inputs=["R", "T", "X", "G", "V"],
            outputs=["X_new", "V_new"],
            norm_coefficient=norm_coefficient,
            alpha=alpha,
            beta=beta,
            mode="nesterov",
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar
        x = np.array([1.2, 2.8], dtype=np.float32)
        g = np.array([-0.94, -2.5], dtype=np.float32)
        v = np.array([1.7, 3.6], dtype=np.float32)

        # Compute expected outputs of Momentum.
        x_new, v_new = apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta)

        # Check results.
        expect(
            node,
            inputs=[r, t, x, g, v],
            outputs=[x_new, v_new],
            name="test_nesterov_momentum",
            opset_imports=[
                onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)
            ],
        )

    @staticmethod
    def export_momentum_multiple() -> None:
        # Define operator attributes.
        norm_coefficient = 0.001
        alpha = 0.95
        beta = 0.85

        node = onnx.helper.make_node(
            "Momentum",
            inputs=["R", "T", "X1", "X2", "G1", "G2", "H1", "H2"],
            outputs=["X1_new", "X2_new", "V1_new", "V2_new"],
            norm_coefficient=norm_coefficient,
            alpha=alpha,
            beta=beta,
            mode="standard",
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar

        x1 = np.array([1.0], dtype=np.float32)
        g1 = np.array([-1.0], dtype=np.float32)
        v1 = np.array([2.0], dtype=np.float32)

        x2 = np.array([1.0, 2.0], dtype=np.float32)
        g2 = np.array([-1.0, -3.0], dtype=np.float32)
        v2 = np.array([4.0, 1.0], dtype=np.float32)

        # Compute expected outputs of Momentum.
        x1_new, v1_new = apply_momentum(r, t, x1, g1, v1, norm_coefficient, alpha, beta)
        x2_new, v2_new = apply_momentum(r, t, x2, g2, v2, norm_coefficient, alpha, beta)

        # Check results.
        expect(
            node,
            inputs=[r, t, x1, x2, g1, g2, v1, v2],
            outputs=[x1_new, x2_new, v1_new, v2_new],
            name="test_momentum_multiple",
            opset_imports=[
                onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)
            ],
        )
