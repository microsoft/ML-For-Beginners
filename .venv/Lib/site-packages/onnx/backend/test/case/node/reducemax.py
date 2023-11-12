# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceMax(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            "ReduceMax",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[20., 2.]
        # [40., 2.]
        # [60., 2.]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_do_not_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_do_not_keepdims_random",
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMax",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[20., 2.]]
        # [[40., 2.]]
        # [[60., 2.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_keepdims_random",
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = None
        keepdims = 1
        node = onnx.helper.make_node(
            "ReduceMax", inputs=["data"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_max_default_axes_keepdim_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_max_default_axes_keepdims_random",
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMax",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[20., 2.]]
        # [[40., 2.]]
        # [[60., 2.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_negative_axes_keepdims_example",
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_max_negative_axes_keepdims_random",
        )
