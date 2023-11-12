# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class CenterCropPad(Base):
    @staticmethod
    def export_center_crop_pad_crop() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
        )

        # First dim is even diff, second is uneven
        x = np.random.randn(20, 10, 3).astype(np.float32)
        shape = np.array([10, 7, 3], dtype=np.int64)
        y = x[5:15, 1:8, :]

        expect(node, inputs=[x, shape], outputs=[y], name="test_center_crop_pad_crop")

    @staticmethod
    def export_center_crop_pad_pad() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
        )

        # First dim is even diff, second is uneven
        x = np.random.randn(10, 7, 3).astype(np.float32)
        shape = np.array([20, 10, 3], dtype=np.int64)
        y = np.zeros([20, 10, 3], dtype=np.float32)
        y[5:15, 1:8, :] = x

        expect(node, inputs=[x, shape], outputs=[y], name="test_center_crop_pad_pad")

    @staticmethod
    def export_center_crop_pad_crop_and_pad() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
        )

        # Cropping on first dim, padding on second, third stays the same
        x = np.random.randn(20, 8, 3).astype(np.float32)
        shape = np.array([10, 10, 3], dtype=np.int64)
        y = np.zeros([10, 10, 3], dtype=np.float32)
        y[:, 1:9, :] = x[5:15, :, :]

        expect(
            node,
            inputs=[x, shape],
            outputs=[y],
            name="test_center_crop_pad_crop_and_pad",
        )

    @staticmethod
    def export_center_crop_pad_crop_axes_hwc() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
            axes=[0, 1],
        )

        # Cropping on first dim, padding on second, third stays the same
        x = np.random.randn(20, 8, 3).astype(np.float32)
        shape = np.array([10, 9], dtype=np.int64)
        y = np.zeros([10, 9, 3], dtype=np.float32)
        y[:, :8, :] = x[5:15, :, :]

        expect(
            node,
            inputs=[x, shape],
            outputs=[y],
            name="test_center_crop_pad_crop_axes_hwc",
        )

    @staticmethod
    def export_center_crop_pad_crop_negative_axes_hwc() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
            axes=[-3, -2],
        )

        # Cropping on first dim, padding on second, third stays the same
        x = np.random.randn(20, 8, 3).astype(np.float32)
        shape = np.array([10, 9], dtype=np.int64)
        y = np.zeros([10, 9, 3], dtype=np.float32)
        y[:, :8, :] = x[5:15, :, :]

        expect(
            node,
            inputs=[x, shape],
            outputs=[y],
            name="test_center_crop_pad_crop_negative_axes_hwc",
        )

    @staticmethod
    def export_center_crop_pad_crop_axes_chw() -> None:
        node = onnx.helper.make_node(
            "CenterCropPad",
            inputs=["x", "shape"],
            outputs=["y"],
            axes=[1, 2],
        )

        # Cropping on second dim, padding on third, first stays the same
        x = np.random.randn(3, 20, 8).astype(np.float32)
        shape = np.array([10, 9], dtype=np.int64)
        y = np.zeros([3, 10, 9], dtype=np.float32)
        y[:, :, :8] = x[:, 5:15, :]

        expect(
            node,
            inputs=[x, shape],
            outputs=[y],
            name="test_center_crop_pad_crop_axes_chw",
        )
