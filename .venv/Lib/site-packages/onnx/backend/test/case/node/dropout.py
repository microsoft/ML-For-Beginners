# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def dropout(X, drop_probability=0.5, seed=0, training_mode=False, return_mask=False):  # type: ignore
    if drop_probability == 0 or training_mode is False:
        if return_mask is True:
            return X, np.ones(X.shape, dtype=bool)
        else:
            return X

    np.random.seed(seed)
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = 1 / (1 - drop_probability)
    if return_mask:
        return mask * X * scale, mask.astype(bool)
    return mask * X * scale


class Dropout(Base):
    # Inferencing tests.
    @staticmethod
    def export_default() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node("Dropout", inputs=["x"], outputs=["y"], seed=seed)

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = dropout(x)
        expect(node, inputs=[x], outputs=[y], name="test_dropout_default")

    @staticmethod
    def export_default_ratio() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r"], outputs=["y"], seed=seed
        )

        r = np.float32(0.1)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = dropout(x, r)
        expect(node, inputs=[x, r], outputs=[y], name="test_dropout_default_ratio")

    @staticmethod
    def export_default_mask() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x"], outputs=["y", "z"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y, z = dropout(x, return_mask=True)
        expect(node, inputs=[x], outputs=[y, z], name="test_dropout_default_mask")

    @staticmethod
    def export_default_mask_ratio() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r"], outputs=["y", "z"], seed=seed
        )

        r = np.float32(0.1)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y, z = dropout(x, r, return_mask=True)
        expect(
            node, inputs=[x, r], outputs=[y, z], name="test_dropout_default_mask_ratio"
        )

    # Training tests.

    @staticmethod
    def export_training_default() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.5)
        t = np.bool_(True)
        y = dropout(x, r, training_mode=t)
        expect(
            node, inputs=[x, r, t], outputs=[y], name="test_training_dropout_default"
        )

    @staticmethod
    def export_training_default_ratio_mask() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y", "z"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.5)
        t = np.bool_(True)
        y, z = dropout(x, r, training_mode=t, return_mask=True)
        expect(
            node,
            inputs=[x, r, t],
            outputs=[y, z],
            name="test_training_dropout_default_mask",
        )

    @staticmethod
    def export_training() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.75)
        t = np.bool_(True)
        y = dropout(x, r, training_mode=t)
        expect(node, inputs=[x, r, t], outputs=[y], name="test_training_dropout")

    @staticmethod
    def export_training_ratio_mask() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y", "z"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.75)
        t = np.bool_(True)
        y, z = dropout(x, r, training_mode=t, return_mask=True)
        expect(
            node, inputs=[x, r, t], outputs=[y, z], name="test_training_dropout_mask"
        )

    @staticmethod
    def export_training_default_zero_ratio() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.0)
        t = np.bool_(True)
        y = dropout(x, r, training_mode=t)
        expect(
            node, inputs=[x, r, t], outputs=[y], name="test_training_dropout_zero_ratio"
        )

    @staticmethod
    def export_training_default_zero_ratio_mask() -> None:
        seed = np.int64(0)
        node = onnx.helper.make_node(
            "Dropout", inputs=["x", "r", "t"], outputs=["y", "z"], seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.0)
        t = np.bool_(True)
        y, z = dropout(x, r, training_mode=t, return_mask=True)
        expect(
            node,
            inputs=[x, r, t],
            outputs=[y, z],
            name="test_training_dropout_zero_ratio_mask",
        )

    # Old dropout tests

    @staticmethod
    def export_default_old() -> None:
        node = onnx.helper.make_node(
            "Dropout",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = x
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dropout_default_old",
            opset_imports=[helper.make_opsetid("", 11)],
        )

    @staticmethod
    def export_random_old() -> None:
        node = onnx.helper.make_node(
            "Dropout",
            inputs=["x"],
            outputs=["y"],
            ratio=0.2,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = x
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dropout_random_old",
            opset_imports=[helper.make_opsetid("", 11)],
        )
