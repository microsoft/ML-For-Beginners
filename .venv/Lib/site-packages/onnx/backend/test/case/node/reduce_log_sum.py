# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceLogSum(Base):
    @staticmethod
    def export_nokeepdims() -> None:
        shape = [3, 4, 5]
        axes = np.array([2, 1], dtype=np.int64)

        node = onnx.helper.make_node(
            "ReduceLogSum",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=0,
        )
        data = np.random.ranf(shape).astype(np.float32)
        reduced = np.log(np.sum(data, axis=tuple(axes), keepdims=False))
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_desc_axes",
        )

        axes = np.array([0, 1], dtype=np.int64)
        node = onnx.helper.make_node(
            "ReduceLogSum",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=0,
        )
        data = np.random.ranf(shape).astype(np.float32)
        reduced = np.log(np.sum(data, axis=tuple(axes), keepdims=False))
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_asc_axes",
        )

    @staticmethod
    def export_keepdims() -> None:
        node = onnx.helper.make_node(
            "ReduceLogSum", inputs=["data", "axes"], outputs=["reduced"]
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, keepdims=True))
        axes = np.array([], dtype=np.int64)
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_default",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        axes = np.array([-2], dtype=np.int64)
        node = onnx.helper.make_node(
            "ReduceLogSum", inputs=["data", "axes"], outputs=["reduced"]
        )
        data = np.random.ranf([3, 4, 5]).astype(np.float32)
        reduced = np.log(np.sum(data, axis=tuple(axes), keepdims=True))
        # print(reduced)
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_negative_axes",
        )

    @staticmethod
    def export_empty_set() -> None:
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceLogSum",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array([], dtype=np.float32).reshape(shape)
        axes = np.array([1], dtype=np.int64)
        zero = np.array(np.zeros(reduced_shape, dtype=np.float32))
        reduced = np.log(zero)  # -inf

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_empty_set",
        )
