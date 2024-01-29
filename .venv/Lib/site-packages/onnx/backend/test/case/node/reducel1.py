# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceL1(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([2], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            "ReduceL1",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        # print(data)
        # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[3., 7.], [11., 15.], [19., 23.]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_do_not_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_do_not_keepdims_random",
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceL1",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        # print(data)
        # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_keep_dims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_keep_dims_random",
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceL1", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        # print(data)
        # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=None, keepdims=keepdims == 1)
        # print(reduced)
        # [[[78.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_default_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=None, keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_default_axes_keepdims_random",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceL1",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        # print(data)
        # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_negative_axes_keep_dims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_negative_axes_keep_dims_random",
        )

    @staticmethod
    def export_empty_set() -> None:
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceL1",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array([], dtype=np.float32).reshape(shape)
        axes = np.array([1], dtype=np.int64)
        reduced = np.array(np.zeros(reduced_shape, dtype=np.float32))

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_l1_empty_set",
        )
