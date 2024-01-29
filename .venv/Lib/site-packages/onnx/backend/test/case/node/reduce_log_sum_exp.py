# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceLogSumExp(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0
        node = onnx.helper.make_node(
            "ReduceLogSumExp",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double
        )
        reduced = np.log(np.sum(np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
        # print(reduced)
        # [[20., 2.31326175]
        # [40.00004578, 2.31326175]
        # [60.00671387, 2.31326175]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_do_not_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.double)
        reduced = np.log(np.sum(np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_do_not_keepdims_random",
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1
        node = onnx.helper.make_node(
            "ReduceLogSumExp",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double
        )
        reduced = np.log(np.sum(np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
        # print(reduced)
        # [[[20., 2.31326175]]
        # [[40.00004578, 2.31326175]]
        # [[60.00671387, 2.31326175]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.double)
        reduced = np.log(np.sum(np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_keepdims_random",
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceLogSumExp",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double
        )
        reduced = np.log(np.sum(np.exp(data), axis=None, keepdims=keepdims == 1))
        # print(reduced)
        # [[[60.00671387]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_default_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.double)
        reduced = np.log(np.sum(np.exp(data), axis=None, keepdims=keepdims == 1))
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_default_axes_keepdims_random",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1
        node = onnx.helper.make_node(
            "ReduceLogSumExp",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double
        )
        reduced = np.log(np.sum(np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
        # print(reduced)
        # [[[20., 2.31326175]]
        # [[40.00004578, 2.31326175]]
        # [[60.00671387, 2.31326175]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_negative_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.double)
        reduced = np.log(
            np.sum(np.exp(data), axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        )

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_log_sum_exp_negative_axes_keepdims_random",
        )

    @staticmethod
    def export_empty_set() -> None:
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceLogSumExp",
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
            name="test_reduce_log_sum_exp_empty_set",
        )
