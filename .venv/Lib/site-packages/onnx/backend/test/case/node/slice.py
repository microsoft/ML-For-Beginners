# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Slice(Base):
    @staticmethod
    def export_slice() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[0:3, 0:10]
        starts = np.array([0, 0], dtype=np.int64)
        ends = np.array([3, 10], dtype=np.int64)
        axes = np.array([0, 1], dtype=np.int64)
        steps = np.array([1, 1], dtype=np.int64)

        expect(
            node, inputs=[x, starts, ends, axes, steps], outputs=[y], name="test_slice"
        )

    @staticmethod
    def export_slice_neg() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0], dtype=np.int64)
        ends = np.array([-1], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        steps = np.array([1], dtype=np.int64)
        y = x[:, 0:-1]

        expect(
            node,
            inputs=[x, starts, ends, axes, steps],
            outputs=[y],
            name="test_slice_neg",
        )

    @staticmethod
    def export_slice_start_out_of_bounds() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([1000], dtype=np.int64)
        ends = np.array([1000], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        steps = np.array([1], dtype=np.int64)
        y = x[:, 1000:1000]

        expect(
            node,
            inputs=[x, starts, ends, axes, steps],
            outputs=[y],
            name="test_slice_start_out_of_bounds",
        )

    @staticmethod
    def export_slice_end_out_of_bounds() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([1000], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        steps = np.array([1], dtype=np.int64)
        y = x[:, 1:1000]

        expect(
            node,
            inputs=[x, starts, ends, axes, steps],
            outputs=[y],
            name="test_slice_end_out_of_bounds",
        )

    @staticmethod
    def export_slice_default_axes() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0, 0, 3], dtype=np.int64)
        ends = np.array([20, 10, 4], dtype=np.int64)
        y = x[:, :, 3:4]

        expect(
            node, inputs=[x, starts, ends], outputs=[y], name="test_slice_default_axes"
        )

    @staticmethod
    def export_slice_default_steps() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0, 0, 3], dtype=np.int64)
        ends = np.array([20, 10, 4], dtype=np.int64)
        axes = np.array([0, 1, 2], dtype=np.int64)
        y = x[:, :, 3:4]

        expect(
            node,
            inputs=[x, starts, ends, axes],
            outputs=[y],
            name="test_slice_default_steps",
        )

    @staticmethod
    def export_slice_neg_steps() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([20, 10, 4], dtype=np.int64)
        ends = np.array([0, 0, 1], dtype=np.int64)
        axes = np.array([0, 1, 2], dtype=np.int64)
        steps = np.array([-1, -3, -2]).astype(np.int64)
        y = x[20:0:-1, 10:0:-3, 4:1:-2]

        expect(
            node,
            inputs=[x, starts, ends, axes, steps],
            outputs=[y],
            name="test_slice_neg_steps",
        )

    @staticmethod
    def export_slice_negative_axes() -> None:
        node = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes"],
            outputs=["y"],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0, 0, 3], dtype=np.int64)
        ends = np.array([20, 10, 4], dtype=np.int64)
        axes = np.array([0, -2, -1], dtype=np.int64)
        y = x[:, :, 3:4]

        expect(
            node,
            inputs=[x, starts, ends, axes],
            outputs=[y],
            name="test_slice_negative_axes",
        )
