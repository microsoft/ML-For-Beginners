# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Equal(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Equal",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_equal")

    @staticmethod
    def export_equal_broadcast() -> None:
        node = onnx.helper.make_node(
            "Equal",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_equal_bcast")

    @staticmethod
    def export_equal_string() -> None:
        node = onnx.helper.make_node(
            "Equal",
            inputs=["x", "y"],
            outputs=["z"],
        )
        x = np.array(["string1", "string2"], dtype=np.dtype(object))
        y = np.array(["string1", "string3"], dtype=np.dtype(object))
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_equal_string")

    @staticmethod
    def export_equal_string_broadcast() -> None:
        node = onnx.helper.make_node(
            "Equal",
            inputs=["x", "y"],
            outputs=["z"],
        )
        x = np.array(["string1", "string2"], dtype=np.dtype(object))
        y = np.array(["string1"], dtype=np.dtype(object))
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z], name="test_equal_string_broadcast")
