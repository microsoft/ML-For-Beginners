# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def scatter_nd_impl(data, indices, updates, reduction="none"):  # type: ignore
    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]

    # Compute output
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == "add":
            output[tuple(indices[i])] += updates[i]
        elif reduction == "mul":
            output[tuple(indices[i])] *= updates[i]
        elif reduction == "max":
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == "min":
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output


class ScatterND(Base):
    @staticmethod
    def export_scatternd() -> None:
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
        )
        data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0], [2]], dtype=np.int64)
        updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )
        # Expecting output as np.array(
        #    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        output = scatter_nd_impl(data, indices, updates)
        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[output],
            name="test_scatternd",
        )

    @staticmethod
    def export_scatternd_add() -> None:
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
            reduction="add",
        )
        data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0], [0]], dtype=np.int64)
        updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )
        # Expecting output as np.array(
        #    [[[7, 8, 9, 10], [13, 14, 15, 16], [18, 17, 16, 15], [16, 15, 14, 13]],
        #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        output = scatter_nd_impl(data, indices, updates, reduction="add")
        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[output],
            name="test_scatternd_add",
        )

    @staticmethod
    def export_scatternd_multiply() -> None:
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
            reduction="mul",
        )
        data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0], [0]], dtype=np.int64)
        updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )
        # Expecting output as np.array(
        #    [[[5, 10, 15, 20], [60, 72, 84, 96], [168, 147, 126, 105], [128, 96, 64, 32]],
        #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        output = scatter_nd_impl(data, indices, updates, reduction="mul")
        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[output],
            name="test_scatternd_multiply",
        )

    @staticmethod
    def export_scatternd_max() -> None:
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
            reduction="max",
        )
        data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0], [0]], dtype=np.int64)
        updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )
        # Expecting output as np.array(
        #    [[[5, 5, 5, 5], [6, 6, 7, 8], [8, 7, 7, 7], [8, 8 ,8, 8]],
        #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        output = scatter_nd_impl(data, indices, updates, reduction="max")
        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[output],
            name="test_scatternd_max",
        )

    @staticmethod
    def export_scatternd_min() -> None:
        node = onnx.helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
            reduction="min",
        )
        data = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.float32,
        )
        indices = np.array([[0], [0]], dtype=np.int64)
        updates = np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            ],
            dtype=np.float32,
        )
        # Expecting output as np.array(
        #    [[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 3, 2, 1]],
        #     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        #     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        #     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        output = scatter_nd_impl(data, indices, updates, reduction="min")
        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[output],
            name="test_scatternd_min",
        )
