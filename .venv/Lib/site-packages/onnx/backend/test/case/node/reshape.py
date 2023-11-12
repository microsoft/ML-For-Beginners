# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


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


class Reshape(Base):
    @staticmethod
    def export_reshape() -> None:
        original_shape = [2, 3, 4]
        test_cases = {
            "reordered_all_dims": np.array([4, 2, 3], dtype=np.int64),
            "reordered_last_dims": np.array([2, 4, 3], dtype=np.int64),
            "reduced_dims": np.array([2, 12], dtype=np.int64),
            "extended_dims": np.array([2, 3, 2, 2], dtype=np.int64),
            "one_dim": np.array([24], dtype=np.int64),
            "negative_dim": np.array([2, -1, 2], dtype=np.int64),
            "negative_extended_dims": np.array([-1, 2, 3, 4], dtype=np.int64),
            "zero_dim": np.array([2, 0, 4, 1], dtype=np.int64),
            "zero_and_negative_dim": np.array([2, 0, 1, -1], dtype=np.int64),
        }
        data = np.random.random_sample(original_shape).astype(np.float32)

        for test_name, shape in test_cases.items():
            node = onnx.helper.make_node(
                "Reshape",
                inputs=["data", "shape"],
                outputs=["reshaped"],
            )

            reshaped = reshape_reference_implementation(data, shape)

            expect(
                node,
                inputs=[data, shape],
                outputs=[reshaped],
                name="test_reshape_" + test_name,
            )

    @staticmethod
    def export_allowzero() -> None:
        original_shape = [0, 3, 4]
        test_cases = {
            "allowzero_reordered": np.array([3, 4, 0], dtype=np.int64),
        }
        data = np.random.random_sample(original_shape).astype(np.float32)

        for test_name, shape in test_cases.items():
            node = onnx.helper.make_node(
                "Reshape",
                inputs=["data", "shape"],
                outputs=["reshaped"],
                allowzero=1,  # if allowzero=1, final shape = (3, 4, 0)
                # if allowzero=0, final shape = (3, 4, 4)
            )

            reshaped = reshape_reference_implementation(data, shape, allowzero=1)

            expect(
                node,
                inputs=[data, shape],
                outputs=[reshaped],
                name="test_reshape_" + test_name,
            )
