# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN


def apply_adagrad(r, t, x, g, h, norm_coefficient, epsilon, decay_factor):  # type: ignore
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


class Adagrad(Base):
    @staticmethod
    def export_adagrad() -> None:
        # Define operator attributes.
        norm_coefficient = 0.001
        epsilon = 1e-5
        decay_factor = 0.1

        # Create operator.
        node = onnx.helper.make_node(
            "Adagrad",
            inputs=["R", "T", "X", "G", "H"],
            outputs=["X_new", "H_new"],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon,
            decay_factor=decay_factor,
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar
        x = np.array([1.0], dtype=np.float32)
        g = np.array([-1.0], dtype=np.float32)
        h = np.array([2.0], dtype=np.float32)

        # Compute expected outputs of Adagrad.
        x_new, h_new = apply_adagrad(
            r, t, x, g, h, norm_coefficient, epsilon, decay_factor
        )

        # Check results.
        expect(
            node,
            inputs=[r, t, x, g, h],
            outputs=[x_new, h_new],
            name="test_adagrad",
            opset_imports=[
                onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)
            ],
        )

    @staticmethod
    def export_adagrad_multiple() -> None:
        # Define operator attributes.
        norm_coefficient = 0.001
        epsilon = 1e-5
        decay_factor = 0.1

        node = onnx.helper.make_node(
            "Adagrad",
            inputs=["R", "T", "X1", "X2", "G1", "G2", "H1", "H2"],
            outputs=["X1_new", "X2_new", "H1_new", "H2_new"],
            norm_coefficient=norm_coefficient,
            epsilon=epsilon,
            decay_factor=decay_factor,
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
        )

        # Define operator inputs.
        r = np.array(0.1, dtype=np.float32)  # scalar
        t = np.array(0, dtype=np.int64)  # scalar

        x1 = np.array([1.0], dtype=np.float32)
        g1 = np.array([-1.0], dtype=np.float32)
        h1 = np.array([2.0], dtype=np.float32)

        x2 = np.array([1.0, 2.0], dtype=np.float32)
        g2 = np.array([-1.0, -3.0], dtype=np.float32)
        h2 = np.array([4.0, 1.0], dtype=np.float32)

        # Compute expected outputs of Adagrad.
        x1_new, h1_new = apply_adagrad(
            r, t, x1, g1, h1, norm_coefficient, epsilon, decay_factor
        )
        x2_new, h2_new = apply_adagrad(
            r, t, x2, g2, h2, norm_coefficient, epsilon, decay_factor
        )

        # Check results.
        expect(
            node,
            inputs=[r, t, x1, x2, g1, g2, h1, h2],
            outputs=[x1_new, x2_new, h1_new, h2_new],
            name="test_adagrad_multiple",
            opset_imports=[
                onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)
            ],
        )
