# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class InstanceNormalization(Base):
    @staticmethod
    def export() -> None:
        def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
            dims_x = len(x.shape)
            axis = tuple(range(2, dims_x))
            mean = np.mean(x, axis=axis, keepdims=True)
            var = np.var(x, axis=axis, keepdims=True)
            dim_ones = (1,) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)
            return s * (x - mean) / np.sqrt(var + epsilon) + bias

        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        y = _instancenorm_test_mode(x, s, bias).astype(np.float32)

        node = onnx.helper.make_node(
            "InstanceNormalization",
            inputs=["x", "s", "bias"],
            outputs=["y"],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias], outputs=[y], name="test_instancenorm_example")

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        epsilon = 1e-2
        y = _instancenorm_test_mode(x, s, bias, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            "InstanceNormalization",
            inputs=["x", "s", "bias"],
            outputs=["y"],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias], outputs=[y], name="test_instancenorm_epsilon")
