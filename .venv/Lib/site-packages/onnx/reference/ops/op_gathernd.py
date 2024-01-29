# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from onnx.reference.op_run import OpRun


def _gather_nd_impl(
    data: np.ndarray, indices: np.ndarray, batch_dims: int
) -> Tuple[np.ndarray]:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # The list of data/indice shape of batch_dims.
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array.
    batch_dims_size = 1

    # Check the shape of indice and data are identical for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below.
    # Compute shape of output array.
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data.
    output_data_buffer = []

    # Flatten 'indices' to 2D array.
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape
    # (batch_dim_size, data.shape[batch_dimes:]).
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # Gather each scalar value from 'data'.
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return (np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape),)


class GatherND(OpRun):
    def _run(self, data, indices, batch_dims=None):  # type: ignore
        return _gather_nd_impl(data, indices, batch_dims)  # type: ignore
