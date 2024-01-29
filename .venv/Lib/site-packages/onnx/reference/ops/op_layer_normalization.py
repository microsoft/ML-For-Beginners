# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from onnx.reference.op_run import OpRun


def _layer_normalization(
    X: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    axis: int = -1,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    Y = np.reshape(y_mat, X_shape) * W
    if B is not None:
        Y = Y + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return (Y.astype(X.dtype), X_mean.astype(X.dtype), X_inv_std_dev.astype(X.dtype))


class LayerNormalization(OpRun):
    def _run(self, X, Scale, B=None, axis=None, epsilon=None, stash_type=None):  # type: ignore
        if stash_type != 1:
            raise NotImplementedError(
                f"LayerNormalization not implemented for stash_type={stash_type} != 1."
            )
        res = _layer_normalization(X, Scale, B, axis=axis, epsilon=epsilon)
        return res
