# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias


def _batchnorm_training_mode(x, s, bias, mean, var, momentum=0.9, epsilon=1e-5):  # type: ignore
    axis = tuple(np.delete(np.arange(len(x.shape)), 1))
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    output_mean = mean * momentum + saved_mean * (1 - momentum)
    output_var = var * momentum + saved_var * (1 - momentum)
    y = _batchnorm_test_mode(x, s, bias, saved_mean, saved_var, epsilon=epsilon)
    return y.astype(np.float32), output_mean, output_var


class BatchNormalization(Base):
    @staticmethod
    def export() -> None:
        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "s", "bias", "mean", "var"],
            outputs=["y"],
        )

        # output size: (2, 3, 4, 5)
        expect(
            node,
            inputs=[x, s, bias, mean, var],
            outputs=[y],
            name="test_batchnorm_example",
        )

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        epsilon = 1e-2
        y = _batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "s", "bias", "mean", "var"],
            outputs=["y"],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(
            node,
            inputs=[x, s, bias, mean, var],
            outputs=[y],
            name="test_batchnorm_epsilon",
        )

    @staticmethod
    def export_train() -> None:
        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        # using np.bool(1) while generating test data with "'bool' object has no attribute 'dtype'"
        # working around by using np.byte(1).astype(bool)
        training_mode = 1
        y, output_mean, output_var = _batchnorm_training_mode(x, s, bias, mean, var)

        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "s", "bias", "mean", "var"],
            outputs=["y", "output_mean", "output_var"],
            training_mode=training_mode,
        )

        # output size: (2, 3, 4, 5)
        expect(
            node,
            inputs=[x, s, bias, mean, var],
            outputs=[y, output_mean, output_var],
            name="test_batchnorm_example_training_mode",
        )

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        training_mode = 1
        momentum = 0.9
        epsilon = 1e-2
        y, output_mean, output_var = _batchnorm_training_mode(
            x, s, bias, mean, var, momentum, epsilon
        )

        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "s", "bias", "mean", "var"],
            outputs=["y", "output_mean", "output_var"],
            epsilon=epsilon,
            training_mode=training_mode,
        )

        # output size: (2, 3, 4, 5)
        expect(
            node,
            inputs=[x, s, bias, mean, var],
            outputs=[y, output_mean, output_var],
            name="test_batchnorm_epsilon_training_mode",
        )
