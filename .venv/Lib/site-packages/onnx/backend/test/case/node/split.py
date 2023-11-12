# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Split(Base):
    @staticmethod
    def export_1d_opset13() -> None:
        node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3"],
            axis=0,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_1d_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2"],
            axis=0,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_1d_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

    @staticmethod
    def export_2d_opset13() -> None:
        node_input = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "Split", inputs=["input"], outputs=["output_1", "output_2"], axis=1
        )

        expected_outputs = [
            np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
            np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
        ]

        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_2d_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2"],
            axis=1,
        )

        expected_outputs = [
            np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
            np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                np.float32
            ),
        ]

        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_2d_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

    @staticmethod
    def export_default_values_opset13() -> None:
        node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

        # If axis is not specified, split is applied on default axis 0
        node = onnx.helper.make_node(
            "Split", inputs=["input"], outputs=["output_1", "output_2", "output_3"]
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_default_axis_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output_1", "output_2"]
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_default_axis_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

    @staticmethod
    def export_zero_size_splits_opset13() -> None:
        # 1-dimensional tensor with dimension_size=0
        node_input = np.array([]).astype(np.float32)

        # Split emtpy tensor to tensors of size zero
        split = np.array([0, 0, 0]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2", "output_3"],
        )

        expected_outputs = [
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_zero_size_splits_opset13",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

    @staticmethod
    def export_1d_opset18() -> None:
        node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3"],
            axis=0,
            num_outputs=3,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_1d_opset18",
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2"],
            axis=0,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_1d_opset18",
        )

    @staticmethod
    def export_2d_opset18() -> None:
        node_input = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2"],
            axis=1,
            num_outputs=2,
        )

        expected_outputs = [
            np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
            np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
        ]

        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_2d",
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2"],
            axis=1,
        )

        expected_outputs = [
            np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
            np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(
                np.float32
            ),
        ]

        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_2d_opset18",
        )

    @staticmethod
    def export_default_values_opset18() -> None:
        node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

        # If axis is not specified, split is applied on default axis 0
        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3"],
            num_outputs=3,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_equal_parts_default_axis_opset18",
        )

        split = np.array([2, 4]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split", inputs=["input", "split"], outputs=["output_1", "output_2"]
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_variable_parts_default_axis_opset18",
        )

    @staticmethod
    def export_zero_size_splits_opset18() -> None:
        # 1-dimensional tensor with dimension_size=0
        node_input = np.array([]).astype(np.float32)

        # Split emtpy tensor to tensors of size zero
        split = np.array([0, 0, 0]).astype(np.int64)
        node = onnx.helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=["output_1", "output_2", "output_3"],
        )

        expected_outputs = [
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
            np.array([]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input, split],
            outputs=expected_outputs,
            name="test_split_zero_size_splits_opset18",
        )

    @staticmethod
    def export_1d_uneven_split_opset18() -> None:
        node_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)

        # If axis is not specified, split is applied on default axis 0
        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3", "output_4"],
            num_outputs=4,
        )

        expected_outputs = [
            np.array([1.0, 2.0]).astype(np.float32),
            np.array([3.0, 4.0]).astype(np.float32),
            np.array([5.0, 6.0]).astype(np.float32),
            np.array([7.0]).astype(np.float32),
        ]
        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_1d_uneven_split_opset18",
        )

    @staticmethod
    def export_2d_uneven_split_opset18() -> None:
        node_input = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ]
        ).astype(np.float32)

        node = onnx.helper.make_node(
            "Split",
            inputs=["input"],
            outputs=["output_1", "output_2", "output_3"],
            axis=1,
            num_outputs=3,
        )

        expected_outputs = [
            np.array([[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]).astype(np.float32),
            np.array([[4.0, 5.0, 6.0], [12.0, 13.0, 14.0]]).astype(np.float32),
            np.array([[7.0, 8.0], [15.0, 16.0]]).astype(np.float32),
        ]

        expect(
            node,
            inputs=[node_input],
            outputs=expected_outputs,
            name="test_split_2d_uneven_split_opset18",
        )
