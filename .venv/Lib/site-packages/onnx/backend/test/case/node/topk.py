# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    sorted_indices = np.argsort(X, axis=axis)
    sorted_values = np.sort(X, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    topk_sorted_indices = np.take(sorted_indices, np.arange(k), axis=axis)
    topk_sorted_values = np.take(sorted_values, np.arange(k), axis=axis)
    return topk_sorted_values, np.array(topk_sorted_indices, dtype=np.int64)


class TopK(Base):
    @staticmethod
    def export_top_k() -> None:
        axis = 1
        largest = 1

        k = 3
        node = onnx.helper.make_node(
            "TopK", inputs=["x", "k"], outputs=["values", "indices"], axis=axis
        )
        X = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=np.float32,
        )
        K = np.array([k], dtype=np.int64)
        values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

        # print(values_ref)
        # [[ 3.  2.  1.]
        # [ 7.  6.  5.]
        # [11. 10.  9.]]
        # print(indices_ref)
        # [[3 2 1]
        # [3 2 1]
        # [3 2 1]]

        expect(
            node, inputs=[X, K], outputs=[values_ref, indices_ref], name="test_top_k"
        )

    @staticmethod
    def export_top_k_smallest() -> None:
        axis = 1
        largest = 0
        sorted = 1
        k = 3

        node = onnx.helper.make_node(
            "TopK",
            inputs=["x", "k"],
            outputs=["values", "indices"],
            axis=axis,
            largest=largest,
            sorted=sorted,
        )

        X = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [11, 10, 9, 8],
            ],
            dtype=np.float32,
        )
        K = np.array([k], dtype=np.int64)
        values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

        # print(values_ref)
        # [[ 0.  1.  2.]
        # [ 4.  5.  6.]
        # [ 8.  9. 10.]]
        # print(indices_ref)
        # [[0 1 2]
        # [0 1 2]
        # [3 2 1]]

        expect(
            node,
            inputs=[X, K],
            outputs=[values_ref, indices_ref],
            name="test_top_k_smallest",
        )

    @staticmethod
    def export_top_k_negative_axis() -> None:
        axis = -1
        largest = 1

        k = 3
        node = onnx.helper.make_node(
            "TopK", inputs=["x", "k"], outputs=["values", "indices"], axis=axis
        )
        X = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=np.float32,
        )
        K = np.array([k], dtype=np.int64)
        values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

        # print(values_ref)
        # [[ 3.  2.  1.]
        # [ 7.  6.  5.]
        # [11. 10.  9.]]
        # print(indices_ref)
        # [[3 2 1]
        # [3 2 1]
        # [3 2 1]]

        expect(
            node,
            inputs=[X, K],
            outputs=[values_ref, indices_ref],
            name="test_top_k_negative_axis",
        )
