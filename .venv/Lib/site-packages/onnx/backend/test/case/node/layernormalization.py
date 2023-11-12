# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


# Layer normalization's reference implementation
def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev


def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]


class LayerNormalization(Base):
    @staticmethod
    def export() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)

            node = onnx.helper.make_node(
                "LayerNormalization",
                inputs=["X", "W", "B"],
                outputs=["Y", "Mean", "InvStdDev"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_layer_normalization_4d_axis_negative_{-axis}"
            else:
                name = f"test_layer_normalization_4d_axis{axis}"

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export_default_axis() -> None:
        X = np.random.randn(2, 3, 4, 5).astype(np.float32)

        # Default axis in LayerNormalization is -1.
        normalized_shape = calculate_normalized_shape(X.shape, -1)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        B = np.random.randn(*normalized_shape).astype(np.float32)
        # Axis is default to -1 in the reference implementation.
        Y, mean, inv_std_dev = _layer_normalization(X, W, B)

        # Not specifying axis attribute means -1.
        node = onnx.helper.make_node(
            "LayerNormalization",
            inputs=["X", "W", "B"],
            outputs=["Y", "Mean", "InvStdDev"],
        )

        expect(
            node,
            inputs=[X, W, B],
            outputs=[Y, mean, inv_std_dev],
            name="test_layer_normalization_default_axis",
        )

    @staticmethod
    def export2d() -> None:
        X = np.random.randn(3, 4).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis=axis)

            node = onnx.helper.make_node(
                "LayerNormalization",
                inputs=["X", "W", "B"],
                outputs=["Y", "Mean", "InvStdDev"],
                axis=axis,
            )

            if axis < 0:
                name = f"test_layer_normalization_2d_axis_negative_{-axis}"
            else:
                name = f"test_layer_normalization_2d_axis{axis}"

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))

    @staticmethod
    def export3d_epsilon() -> None:
        epsilon = 1e-1
        X = np.random.randn(2, 3, 5).astype(np.float32)

        def case(axis: int) -> None:
            normalized_shape = calculate_normalized_shape(X.shape, axis)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, epsilon)
            node = onnx.helper.make_node(
                "LayerNormalization",
                inputs=["X", "W", "B"],
                outputs=["Y", "Mean", "InvStdDev"],
                axis=axis,
                epsilon=epsilon,
            )

            if axis < 0:
                name = f"test_layer_normalization_3d_axis_negative_{-axis}_epsilon"
            else:
                name = f"test_layer_normalization_3d_axis{axis}_epsilon"

            expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name=name)

        for i in range(len(X.shape)):
            case(i)
            case(i - len(X.shape))
