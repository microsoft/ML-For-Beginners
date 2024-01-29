# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceSum(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        # print(reduced)
        # [[4., 6.]
        # [12., 14.]
        # [20., 22.]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_do_not_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_do_not_keepdims_random",
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        # print(reduced)
        # [[[4., 6.]]
        # [[12., 14.]]
        # [[20., 22.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_keepdims_random",
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.sum(data, axis=None, keepdims=keepdims == 1)
        # print(reduced)
        # [[[78.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_default_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=None, keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_default_axes_keepdims_random",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        # print(reduced)
        # [[[4., 6.]]
        # [[12., 14.]]
        # [[20., 22.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_negative_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_negative_axes_keepdims_random",
        )

    @staticmethod
    def export_empty_axes_input_noop() -> None:
        shape = [3, 2, 2]
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
            noop_with_empty_axes=True,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        axes = np.array([], dtype=np.int64)
        reduced = np.array(data)
        # print(reduced)
        # [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_empty_axes_input_noop_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.array(data)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_negative_axes_keepdims_random",
        )

    @staticmethod
    def export_empty_set() -> None:
        """Test case with the reduced-axis of size zero."""
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceSum",
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
            name="test_reduce_sum_empty_set",
        )

    @staticmethod
    def export_non_reduced_axis_zero() -> None:
        """Test case with the non-reduced-axis of size zero."""
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 0, 1]

        node = onnx.helper.make_node(
            "ReduceSum",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array([], dtype=np.float32).reshape(shape)
        axes = np.array([2], dtype=np.int64)
        reduced = np.array([], dtype=np.float32).reshape(reduced_shape)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_sum_empty_set_non_reduced_axis_zero",
        )
