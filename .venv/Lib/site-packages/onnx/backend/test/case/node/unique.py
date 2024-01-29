# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def specify_int64(indices, inverse_indices, counts):  # type: ignore
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


class Unique(Base):
    @staticmethod
    def export_sorted_without_axis() -> None:
        node_sorted = onnx.helper.make_node(
            "Unique",
            inputs=["X"],
            outputs=["Y", "indices", "inverse_indices", "counts"],
        )

        x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)
        indices, inverse_indices, counts = specify_int64(
            indices, inverse_indices, counts
        )
        expect(
            node_sorted,
            inputs=[x],
            outputs=[y, indices, inverse_indices, counts],
            name="test_unique_sorted_without_axis",
        )

    @staticmethod
    def export_not_sorted_without_axis() -> None:
        node_not_sorted = onnx.helper.make_node(
            "Unique",
            inputs=["X"],
            outputs=["Y", "indices", "inverse_indices", "counts"],
            sorted=0,
        )
        # numpy unique does not retain original order (it sorts the output unique values)
        # https://github.com/numpy/numpy/issues/8621
        # we need to recover unsorted output and indices
        x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
        y, indices, inverse_indices, counts = np.unique(x, True, True, True)

        # prepare index mapping from sorted to unsorted
        argsorted_indices = np.argsort(indices)
        inverse_indices_map = dict(
            zip(argsorted_indices, np.arange(len(argsorted_indices)))
        )

        indices = indices[argsorted_indices]
        y = np.take(x, indices, axis=0)
        inverse_indices = np.asarray(
            [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64
        )
        counts = counts[argsorted_indices]
        indices, inverse_indices, counts = specify_int64(
            indices, inverse_indices, counts
        )
        # print(y)
        # [2.0, 1.0, 3.0, 4.0]
        # print(indices)
        # [0 1 3 4]
        # print(inverse_indices)
        # [0, 1, 1, 2, 3, 2]
        # print(counts)
        # [1, 2, 2, 1]

        expect(
            node_not_sorted,
            inputs=[x],
            outputs=[y, indices, inverse_indices, counts],
            name="test_unique_not_sorted_without_axis",
        )

    @staticmethod
    def export_sorted_with_axis() -> None:
        node_sorted = onnx.helper.make_node(
            "Unique",
            inputs=["X"],
            outputs=["Y", "indices", "inverse_indices", "counts"],
            sorted=1,
            axis=0,
        )

        x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]], dtype=np.float32)
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=0)
        indices, inverse_indices, counts = specify_int64(
            indices, inverse_indices, counts
        )
        # print(y)
        # [[1. 0. 0.]
        #  [2. 3. 4.]]
        # print(indices)
        # [0 2]
        # print(inverse_indices)
        # [0 0 1]
        # print(counts)
        # [2 1]

        expect(
            node_sorted,
            inputs=[x],
            outputs=[y, indices, inverse_indices, counts],
            name="test_unique_sorted_with_axis",
        )

    @staticmethod
    def export_sorted_with_axis_3d() -> None:
        node_sorted = onnx.helper.make_node(
            "Unique",
            inputs=["X"],
            outputs=["Y", "indices", "inverse_indices", "counts"],
            sorted=1,
            axis=1,
        )

        x = np.array(
            [
                [[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=1)
        indices, inverse_indices, counts = specify_int64(
            indices, inverse_indices, counts
        )
        # print(y)
        # [[[0. 1.]
        #  [1. 1.]
        #  [2. 1.]]
        # [[0. 1.]
        #  [1. 1.]
        #  [2. 1.]]]
        # print(indices)
        # [1 0 2]
        # print(inverse_indices)
        # [1 0 2 0]
        # print(counts)
        # [2 1 1]
        expect(
            node_sorted,
            inputs=[x],
            outputs=[y, indices, inverse_indices, counts],
            name="test_unique_sorted_with_axis_3d",
        )

    @staticmethod
    def export_sorted_with_negative_axis() -> None:
        node_sorted = onnx.helper.make_node(
            "Unique",
            inputs=["X"],
            outputs=["Y", "indices", "inverse_indices", "counts"],
            sorted=1,
            axis=-1,
        )

        x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 3]], dtype=np.float32)
        y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=-1)
        indices, inverse_indices, counts = specify_int64(
            indices, inverse_indices, counts
        )
        # print(y)
        # [[0. 1.]
        #  [0. 1.]
        #  [3. 2.]]
        # print(indices)
        # [1 0]
        # print(inverse_indices)
        # [1 0 0]
        # print(counts)
        # [2 1]

        expect(
            node_sorted,
            inputs=[x],
            outputs=[y, indices, inverse_indices, counts],
            name="test_unique_sorted_with_negative_axis",
        )
