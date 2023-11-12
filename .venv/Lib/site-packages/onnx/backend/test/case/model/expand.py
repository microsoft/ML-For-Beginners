# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect


class ExpandDynamicShape(Base):
    @staticmethod
    def export() -> None:
        def make_graph(
            node: onnx.helper.NodeProto,
            input_shape: Sequence[int],
            shape_shape: Sequence[int],
            output_shape: Sequence[int],
        ) -> onnx.helper.GraphProto:
            graph = onnx.helper.make_graph(
                nodes=[node],
                name="Expand",
                inputs=[
                    onnx.helper.make_tensor_value_info(
                        "X", onnx.TensorProto.FLOAT, input_shape
                    ),
                    onnx.helper.make_tensor_value_info(
                        "shape", onnx.TensorProto.INT64, shape_shape
                    ),
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info(
                        "Y", onnx.TensorProto.FLOAT, output_shape
                    )
                ],
            )
            return graph

        node = onnx.helper.make_node("Expand", ["X", "shape"], ["Y"], name="test")
        input_shape = [1, 3, 1]
        x = np.ones(input_shape, dtype=np.float32)

        # 1st testcase
        shape = np.array([3, 1], dtype=np.int64)
        y = x * np.ones(shape, dtype=np.float32)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )
        expect(model, inputs=[x, shape], outputs=[y], name="test_expand_shape_model1")

        # 2nd testcase
        shape = np.array([1, 3], dtype=np.int64)
        y = x * np.ones(shape, dtype=np.float32)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )
        expect(model, inputs=[x, shape], outputs=[y], name="test_expand_shape_model2")

        # 3rd testcase
        shape = np.array([3, 1, 3], dtype=np.int64)
        y = x * np.ones(shape, dtype=np.float32)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )
        expect(model, inputs=[x, shape], outputs=[y], name="test_expand_shape_model3")

        # 4th testcase
        shape = np.array([3, 3, 1, 3], dtype=np.int64)
        y = x * np.ones(shape, dtype=np.float32)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )
        expect(model, inputs=[x, shape], outputs=[y], name="test_expand_shape_model4")
