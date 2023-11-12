# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect


class ShrinkTest(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Shrink",
            ["x"],
            ["y"],
            lambd=1.5,
            bias=1.5,
        )
        graph = onnx.helper.make_graph(
            nodes=[node],
            name="Shrink",
            inputs=[
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [5])
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [5])
            ],
        )
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([-0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

        expect(model, inputs=[x], outputs=[y], name="test_shrink")
