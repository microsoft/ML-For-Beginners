# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ReduceMin(Base):
    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            "ReduceMin",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[5., 1.]
        # [30., 1.]
        # [55., 1.]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_do_not_keepdims_example",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_do_not_keepdims_random",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMin",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[5., 1.]]
        # [[30., 1.]]
        # [[55., 1.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_keepdims_example",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_keepdims_random",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMin", inputs=["data"], outputs=["reduced"], keepdims=keepdims
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
        # print(reduced)
        # [[[1.]]]

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_min_default_axes_keepdims_example",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data],
            outputs=[reduced],
            name="test_reduce_min_default_axes_keepdims_random",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMin",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
            dtype=np.float32,
        )
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        # print(reduced)
        # [[[5., 1.]]
        # [[30., 1.]]
        # [[55., 1.]]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_negative_axes_keepdims_example",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_negative_axes_keepdims_random",
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

    @staticmethod
    def export_bool_inputs() -> None:
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            "ReduceMin",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array(
            [[True, True], [True, False], [False, True], [False, False]],
        )
        reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=bool(keepdims))
        # print(reduced)
        # [[ True],
        #  [False],
        #  [False],
        #  [False]]

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_bool_inputs",
        )

    @staticmethod
    def export_empty_set() -> None:
        shape = [2, 0, 4]
        keepdims = 1
        reduced_shape = [2, 1, 4]

        node = onnx.helper.make_node(
            "ReduceMin",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

        data = np.array([], dtype=np.float32).reshape(shape)
        axes = np.array([1], dtype=np.int64)
        one = np.array(np.ones(reduced_shape, dtype=np.float32))
        zero = np.array(np.zeros(reduced_shape, dtype=np.float32))
        reduced = one / zero  # inf

        expect(
            node,
            inputs=[data, axes],
            outputs=[reduced],
            name="test_reduce_min_empty_set",
        )
