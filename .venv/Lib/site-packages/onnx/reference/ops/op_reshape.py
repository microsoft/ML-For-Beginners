# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def reshape_reference_implementation(
    data: np.ndarray, shape: np.ndarray, allowzero: int = 0
) -> np.ndarray:
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped


class CommonReshape(OpRun):
    def _run(self, data, shape):  # type: ignore
        return (reshape_reference_implementation(data, shape, 0),)


class Reshape_5(CommonReshape):
    pass


class Reshape_14(CommonReshape):
    def _run(self, data, shape, allowzero=None):  # type: ignore
        if allowzero is None:
            allowzero = getattr(self, "allowzero", 0) == 1
        else:
            allowzero = allowzero == 1
        return (reshape_reference_implementation(data, shape, allowzero),)
