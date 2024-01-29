# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def gather_numpy_2(self: np.ndarray, index: np.ndarray) -> np.ndarray:
    res = []
    for a, b in zip(self, index):
        res.append(a[b[0]])
    return np.array(res, dtype=self.dtype).reshape(index.shape)


def gather_numpy(self: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1 :]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(
            f"Except for dimension {dim!r}, all dimensions of "
            f"index and self should be the same size."
        )
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)

    try:
        gathered = np.choose(index_swaped, data_swaped, mode="wrap")
    except ValueError as e:
        if len(index_swaped.shape) == 2 and len(data_swaped.shape) == 2:
            return gather_numpy_2(self, index)
        raise e  # pragma: no cover

    return np.swapaxes(gathered, 0, dim)


class GatherElements(OpRun):
    def _run(self, data, indices, axis=None):  # type: ignore
        if indices.size == 0:
            return (np.empty((0,), dtype=data.dtype),)
        try:
            return (gather_numpy(data, axis, indices),)
        except TypeError:
            # distribution x86 requires int32.
            return (gather_numpy(data, axis, indices.astype(int)),)
