# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Constant(Base):
    @staticmethod
    def export() -> None:
        values = np.random.randn(5, 5).astype(np.float32)
        node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["values"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

        expect(node, inputs=[], outputs=[values], name="test_constant")
