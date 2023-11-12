# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


class Softmax(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = softmax(x, axis=1)
        expect(node, inputs=[x], outputs=[y], name="test_softmax_example")

    @staticmethod
    def export_softmax_axis() -> None:
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output
        # [[0.032058604 0.08714432  0.23688284  0.6439143  ]
        # [0.032058604 0.08714432  0.23688284  0.6439143  ]]
        y = softmax(x)

        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_softmax_large_number")

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=0,
        )
        y = softmax(x, axis=0)
        expect(node, inputs=[x], outputs=[y], name="test_softmax_axis_0")

        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=1,
        )
        y = softmax(x, axis=1)
        expect(node, inputs=[x], outputs=[y], name="test_softmax_axis_1")

        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=2,
        )
        y = softmax(x, axis=2)
        expect(node, inputs=[x], outputs=[y], name="test_softmax_axis_2")

        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=-1,
        )
        y = softmax(x, axis=-1)
        expect(node, inputs=[x], outputs=[y], name="test_softmax_negative_axis")

        # default axis is -1
        node = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_softmax_default_axis")
