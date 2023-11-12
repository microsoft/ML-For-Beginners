# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Gather(Base):
    @staticmethod
    def export_gather_0() -> None:
        node = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=0,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=0)

        expect(
            node,
            inputs=[data, indices.astype(np.int64)],
            outputs=[y],
            name="test_gather_0",
        )

    @staticmethod
    def export_gather_1() -> None:
        node = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=1,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=1)

        expect(
            node,
            inputs=[data, indices.astype(np.int64)],
            outputs=[y],
            name="test_gather_1",
        )

    @staticmethod
    def export_gather_2d_indices() -> None:
        node = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=1,
        )
        data = np.random.randn(3, 3).astype(np.float32)
        indices = np.array([[0, 2]])
        y = np.take(data, indices, axis=1)

        expect(
            node,
            inputs=[data, indices.astype(np.int64)],
            outputs=[y],
            name="test_gather_2d_indices",
        )

    @staticmethod
    def export_gather_negative_indices() -> None:
        node = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=0,
        )
        data = np.arange(10).astype(np.float32)
        indices = np.array([0, -9, -10])
        y = np.take(data, indices, axis=0)

        # print(y)
        # [0. 1. 0.]

        expect(
            node,
            inputs=[data, indices.astype(np.int64)],
            outputs=[y],
            name="test_gather_negative_indices",
        )
