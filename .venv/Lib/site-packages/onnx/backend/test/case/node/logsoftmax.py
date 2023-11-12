# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def logsoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return (x - x_max) - np.log(s)


class LogSoftmax(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output
        # [[-2.4076061 -1.407606  -0.407606 ]]
        y = logsoftmax(x)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_example_1")

    @staticmethod
    def export_logsoftmax_axis() -> None:
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output
        # [[-3.4401896  -2.4401896  -1.4401896  -0.44018966]
        # [-3.4401896  -2.4401896  -1.4401896  -0.44018966]]
        y = logsoftmax(x)

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_large_number")

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=0,
        )
        y = logsoftmax(x, axis=0)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_0")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=1,
        )
        y = logsoftmax(x, axis=1)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_1")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=2,
        )
        y = logsoftmax(x, axis=2)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_2")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=-1,
        )
        y = logsoftmax(x, axis=-1)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_negative_axis")

        # default axis is -1
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_default_axis")
