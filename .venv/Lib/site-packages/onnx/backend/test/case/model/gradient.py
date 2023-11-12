# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN, ONNX_DOMAIN


class Gradient(Base):
    @staticmethod
    def export_gradient_scalar_add() -> None:
        add_node = onnx.helper.make_node("Add", ["a", "b"], ["c"], name="my_add")
        gradient_node = onnx.helper.make_node(
            "Gradient",
            ["a", "b"],
            ["dc_da", "dc_db"],
            name="my_gradient",
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
            xs=["a", "b"],
            y="c",
        )

        a = np.array(1.0).astype(np.float32)
        b = np.array(2.0).astype(np.float32)
        c = a + b
        # dc / da = d(a+b) / da = 1
        dc_da = np.array(1).astype(np.float32)
        # db / db = d(a+b) / db = 1
        dc_db = np.array(1).astype(np.float32)

        graph = onnx.helper.make_graph(
            nodes=[add_node, gradient_node],
            name="GradientOfAdd",
            inputs=[
                onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, []),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("dc_da", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("dc_db", onnx.TensorProto.FLOAT, []),
            ],
        )
        opsets = [
            onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
            onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
        ]
        model = onnx.helper.make_model_gen_version(
            graph, producer_name="backend-test", opset_imports=opsets
        )
        expect(
            model, inputs=[a, b], outputs=[c, dc_da, dc_db], name="test_gradient_of_add"
        )

    @staticmethod
    def export_gradient_scalar_add_and_mul() -> None:
        add_node = onnx.helper.make_node("Add", ["a", "b"], ["c"], name="my_add")
        mul_node = onnx.helper.make_node("Mul", ["c", "a"], ["d"], name="my_mul")
        gradient_node = onnx.helper.make_node(
            "Gradient",
            ["a", "b"],
            ["dd_da", "dd_db"],
            name="my_gradient",
            domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
            xs=["a", "b"],
            y="d",
        )

        a = np.array(1.0).astype(np.float32)
        b = np.array(2.0).astype(np.float32)
        c = a + b
        # d = a * c = a * (a + b)
        d = a * c
        # dd / da = d(a*a+a*b) / da = 2 * a + b
        dd_da = (2 * a + b).astype(np.float32)
        # dd / db = d(a*a+a*b) / db = a
        dd_db = a

        graph = onnx.helper.make_graph(
            nodes=[add_node, mul_node, gradient_node],
            name="GradientOfTwoOperators",
            inputs=[
                onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, []),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("d", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("dd_da", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("dd_db", onnx.TensorProto.FLOAT, []),
            ],
        )

        opsets = [
            onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
            onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
        ]
        model = onnx.helper.make_model_gen_version(
            graph, producer_name="backend-test", opset_imports=opsets
        )
        expect(
            model,
            inputs=[a, b],
            outputs=[d, dd_da, dd_db],
            name="test_gradient_of_add_and_mul",
        )
