# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Mod(Base):
    @staticmethod
    def export_mod_mixed_sign_float64() -> None:
        node = onnx.helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=1)

        x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float64)
        y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float64)
        z = np.fmod(x, y)  # expected output [-0.1,  0.4,  5. ,  0.1, -0.4,  3.]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_float64")

    @staticmethod
    def export_mod_mixed_sign_float32() -> None:
        node = onnx.helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=1)

        x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
        y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
        z = np.fmod(
            x, y
        )  # expected output [-0.10000038, 0.39999962, 5. , 0.10000038, -0.39999962, 3.]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_float32")

    @staticmethod
    def export_mod_mixed_sign_float16() -> None:
        node = onnx.helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=1)

        x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float16)
        y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float16)
        z = np.fmod(
            x, y
        )  # expected output [-0.10156, 0.3984 , 5. , 0.10156, -0.3984 ,  3.]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_float16")

    @staticmethod
    def export_mod_mixed_sign_int64() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
        z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_int64")

    @staticmethod
    def export_mod_mixed_sign_int32() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int32)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int32)
        z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_int32")

    @staticmethod
    def export_mod_mixed_sign_int16() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int16)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int16)
        z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_int16")

    @staticmethod
    def export_mod_mixed_sign_int8() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int8)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int8)
        z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_mixed_sign_int8")

    @staticmethod
    def export_mod_uint8() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([4, 7, 5]).astype(np.uint8)
        y = np.array([2, 3, 8]).astype(np.uint8)
        z = np.mod(x, y)  # expected output [0, 1, 5]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_uint8")

    @staticmethod
    def export_mod_uint16() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([4, 7, 5]).astype(np.uint16)
        y = np.array([2, 3, 8]).astype(np.uint16)
        z = np.mod(x, y)  # expected output [0, 1, 5]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_uint16")

    @staticmethod
    def export_mod_uint32() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([4, 7, 5]).astype(np.uint32)
        y = np.array([2, 3, 8]).astype(np.uint32)
        z = np.mod(x, y)  # expected output [0, 1, 5]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_uint32")

    @staticmethod
    def export_mod_uint64() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.array([4, 7, 5]).astype(np.uint64)
        y = np.array([2, 3, 8]).astype(np.uint64)
        z = np.mod(x, y)  # expected output [0, 1, 5]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_uint64")

    @staticmethod
    def export_mod_int64_fmod() -> None:
        node = onnx.helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=1)

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
        z = np.fmod(x, y)  # expected output [ 0,  1,  5,  0, -1,  3]
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_int64_fmod")

    @staticmethod
    def export_mod_broadcast() -> None:
        node = onnx.helper.make_node(
            "Mod",
            inputs=["x", "y"],
            outputs=["z"],
        )

        x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
        y = np.array([7]).astype(np.int32)
        z = np.mod(x, y)
        #   array([[[0, 1, 2, 3, 4],
        #     [5, 6, 0, 1, 2]],

        #    [[3, 4, 5, 6, 0],
        #     [1, 2, 3, 4, 5]],

        #    [[6, 0, 1, 2, 3],
        #     [4, 5, 6, 0, 1]]], dtype=int32)
        expect(node, inputs=[x, y], outputs=[z], name="test_mod_broadcast")
