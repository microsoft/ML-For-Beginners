# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def gather_nd_impl(
    data: np.ndarray, indices: np.ndarray, batch_dims: int
) -> np.ndarray:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


class GatherND(Base):
    @staticmethod
    def export_int32() -> None:
        node = onnx.helper.make_node(
            "GatherND",
            inputs=["data", "indices"],
            outputs=["output"],
        )

        data = np.array([[0, 1], [2, 3]], dtype=np.int32)
        indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
        output = gather_nd_impl(data, indices, 0)
        expected_output = np.array([0, 3], dtype=np.int32)
        assert np.array_equal(output, expected_output)
        expect(
            node,
            inputs=[data, indices],
            outputs=[output],
            name="test_gathernd_example_int32",
        )

    @staticmethod
    def export_float32() -> None:
        node = onnx.helper.make_node(
            "GatherND",
            inputs=["data", "indices"],
            outputs=["output"],
        )

        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
        output = gather_nd_impl(data, indices, 0)
        expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
        assert np.array_equal(output, expected_output)
        expect(
            node,
            inputs=[data, indices],
            outputs=[output],
            name="test_gathernd_example_float32",
        )

    @staticmethod
    def export_int32_batchdim_1() -> None:
        node = onnx.helper.make_node(
            "GatherND",
            inputs=["data", "indices"],
            outputs=["output"],
            batch_dims=1,
        )

        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
        indices = np.array([[1], [0]], dtype=np.int64)
        output = gather_nd_impl(data, indices, 1)
        expected_output = np.array([[2, 3], [4, 5]], dtype=np.int32)
        assert np.array_equal(output, expected_output)
        expect(
            node,
            inputs=[data, indices],
            outputs=[output],
            name="test_gathernd_example_int32_batch_dim1",
        )
