# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Clip(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", "min", "max"],
            outputs=["y"],
        )

        x = np.array([-2, 0, 2]).astype(np.float32)
        min_val = np.float32(-1)
        max_val = np.float32(1)
        y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]
        expect(
            node, inputs=[x, min_val, max_val], outputs=[y], name="test_clip_example"
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, min_val, max_val)
        expect(node, inputs=[x, min_val, max_val], outputs=[y], name="test_clip")
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", "min", "max"],
            outputs=["y"],
        )

        min_val = np.float32(-5)
        max_val = np.float32(5)

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(
            node, inputs=[x, min_val, max_val], outputs=[y], name="test_clip_inbounds"
        )

        x = np.array([-6, 0, 6]).astype(np.float32)
        y = np.array([-5, 0, 5]).astype(np.float32)
        expect(
            node, inputs=[x, min_val, max_val], outputs=[y], name="test_clip_outbounds"
        )

        x = np.array([-1, 0, 6]).astype(np.float32)
        y = np.array([-1, 0, 5]).astype(np.float32)
        expect(
            node,
            inputs=[x, min_val, max_val],
            outputs=[y],
            name="test_clip_splitbounds",
        )

    @staticmethod
    def export_clip_default() -> None:
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", "min"],
            outputs=["y"],
        )
        min_val = np.float32(0)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, min_val, np.inf)
        expect(node, inputs=[x, min_val], outputs=[y], name="test_clip_default_min")

        no_min = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", no_min, "max"],
            outputs=["y"],
        )
        max_val = np.float32(0)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, -np.inf, max_val)
        expect(node, inputs=[x, max_val], outputs=[y], name="test_clip_default_max")

        no_max = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", no_min, no_max],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_clip_default_inbounds")

    @staticmethod
    def export_clip_default_int8() -> None:
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", "min"],
            outputs=["y"],
        )
        min_val = np.int8(0)
        x = np.random.randn(3, 4, 5).astype(np.int8)
        y = np.clip(x, min_val, np.iinfo(np.int8).max)
        expect(
            node, inputs=[x, min_val], outputs=[y], name="test_clip_default_int8_min"
        )

        no_min = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", no_min, "max"],
            outputs=["y"],
        )
        max_val = np.int8(0)
        x = np.random.randn(3, 4, 5).astype(np.int8)
        y = np.clip(x, np.iinfo(np.int8).min, max_val)
        expect(
            node, inputs=[x, max_val], outputs=[y], name="test_clip_default_int8_max"
        )

        no_max = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "Clip",
            inputs=["x", no_min, no_max],
            outputs=["y"],
        )

        x = np.array([-1, 0, 1]).astype(np.int8)
        y = np.array([-1, 0, 1]).astype(np.int8)
        expect(node, inputs=[x], outputs=[y], name="test_clip_default_int8_inbounds")
