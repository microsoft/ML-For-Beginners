# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceProd(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            "ReduceProd",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[3., 8.]
        # [35., 48.]
        # [99., 120.]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_do_not_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_do_not_keepdims_random",
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceProd",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3., 8.]]
        # [[35., 48.]]
        # [[99., 120.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_keepdims_random",
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceProd", inputs=["data"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        # print(reduced)
        # [[[4.790016e+08]]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_prod_default_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_prod_default_axes_keepdims_random",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceProd",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
        )
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[3., 8.]]
        # [[35., 48.]]
        # [[99., 120.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_negative_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_negative_axes_keepdims_random",
        )

    @staticmethod
    def export_empty_set() -> None:
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceProd",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array([], dtype=np.float32).reshape(shape)
        axes = np.array([1], dtype=np.int64)
        reduced = np.array(np.ones(reduced_shape, dtype=np.float32))

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_prod_empty_set",
        )
