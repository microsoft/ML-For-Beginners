# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


# The below Scatter's numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter(data, indices, updates, axis=0):  # type: ignore
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1 :]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of updates param. in data
    idx = [
        [
            unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0],
        ]
        for i in range(indices.shape[axis])
    ]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(
        axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape))
    )

    scattered = np.copy(data)
    scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered


class Scatter(Base):
    @staticmethod
    def export_scatter_without_axis() -> None:
        node = onnx.helper.make_node(
            "Scatter",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
        )
        data = np.zeros((3, 3), dtype=np.float32)
        indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
        updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

        y = scatter(data, indices, updates)
        # print(y) produces
        # [[2.0, 1.1, 0.0],
        #  [1.0, 0.0, 2.2],
        #  [0.0, 2.1, 1.2]]

        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[y],
            name="test_scatter_without_axis",
            opset_imports=[helper.make_opsetid("", 10)],
        )

    @staticmethod
    def export_scatter_with_axis() -> None:
        axis = 1
        node = onnx.helper.make_node(
            "Scatter",
            inputs=["data", "indices", "updates"],
            outputs=["y"],
            axis=axis,
        )
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        indices = np.array([[1, 3]], dtype=np.int64)
        updates = np.array([[1.1, 2.1]], dtype=np.float32)

        y = scatter(data, indices, updates, axis=axis)
        # print(y) produces
        # [[1.0, 1.1, 3.0, 2.1, 5.0]]

        expect(
            node,
            inputs=[data, indices, updates],
            outputs=[y],
            name="test_scatter_with_axis",
            opset_imports=[helper.make_opsetid("", 10)],
        )
