# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.backend.test.case.utils import all_numeric_dtypes


class Min(Base):
    @staticmethod
    def export() -> None:
        data_0 = np.array([3, 2, 1]).astype(np.float32)
        data_1 = np.array([1, 4, 4]).astype(np.float32)
        data_2 = np.array([2, 5, 0]).astype(np.float32)
        result = np.array([1, 2, 0]).astype(np.float32)
        node = onnx.helper.make_node(
            "Min",
            inputs=["data_0", "data_1", "data_2"],
            outputs=["result"],
        )
        expect(
            node,
            inputs=[data_0, data_1, data_2],
            outputs=[result],
            name="test_min_example",
        )

        node = onnx.helper.make_node(
            "Min",
            inputs=["data_0"],
            outputs=["result"],
        )
        expect(node, inputs=[data_0], outputs=[data_0], name="test_min_one_input")

        result = np.minimum(data_0, data_1)
        node = onnx.helper.make_node(
            "Min",
            inputs=["data_0", "data_1"],
            outputs=["result"],
        )
        expect(
            node, inputs=[data_0, data_1], outputs=[result], name="test_min_two_inputs"
        )

    @staticmethod
    def export_min_all_numeric_types() -> None:
        for op_dtype in all_numeric_dtypes:
            data_0 = np.array([3, 2, 1]).astype(op_dtype)
            data_1 = np.array([1, 4, 4]).astype(op_dtype)
            result = np.array([1, 2, 1]).astype(op_dtype)
            node = onnx.helper.make_node(
                "Min",
                inputs=["data_0", "data_1"],
                outputs=["result"],
            )
            expect(
                node,
                inputs=[data_0, data_1],
                outputs=[result],
                name=f"test_min_{np.dtype(op_dtype).name}",
            )
