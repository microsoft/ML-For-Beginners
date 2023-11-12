# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def triu_reference_implementation(x, k=0):  # type: ignore
    return np.triu(x, k)


def tril_reference_implementation(x, k=0):  # type: ignore
    return np.tril(x, k)


class Trilu(Base):
    @staticmethod
    def export_triu() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 7, 3, 7, 9],
        #   [0, 2, 8, 6, 9],
        #   [0, 0, 0, 8, 7],
        #   [0, 0, 0, 2, 4]]
        y = triu_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_triu")

    @staticmethod
    def export_triu_neg() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(-1).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [0, 4, 0, 8, 7],
        #   [0, 0, 4, 2, 4]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_neg")

    @staticmethod
    def export_triu_out_neg_out() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(-7).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_out_neg_out")

    @staticmethod
    def export_triu_pos() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(2).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[0, 0, 3, 7, 9],
        #   [0, 0, 0, 6, 9],
        #   [0, 0, 0, 0, 7],
        #   [0, 0, 0, 0, 0]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_pos")

    @staticmethod
    def export_triu_out_pos() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(6).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 0, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_out_pos")

    @staticmethod
    def export_triu_square() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
        y = triu_reference_implementation(x)
        # X:
        # [[[4, 6, 9],
        #   [7, 5, 4],
        #   [8, 1, 2]],
        #
        #  [[1, 4, 9],
        #   [9, 6, 3],
        #   [8, 9, 8]]]
        # expect result:
        # [[[4, 6, 9],
        #   [0, 5, 4],
        #   [0, 0, 2]],
        #
        #  [[1, 4, 9],
        #   [0, 6, 3],
        #   [0, 0, 8]]]
        expect(node, inputs=[x], outputs=[y], name="test_triu_square")

    @staticmethod
    def export_triu_square_neg() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
        k = np.array(-1).astype(np.int64)
        # X:
        # [[[4, 6, 9],
        #   [7, 5, 4],
        #   [8, 1, 2]],
        #
        #  [[1, 4, 9],
        #   [9, 6, 3],
        #   [8, 9, 8]]]
        # expect result:
        # [[[4, 6, 9],
        #   [7, 5, 4],
        #   [0, 1, 2]],
        #
        #  [[1, 4, 9],
        #   [9, 6, 3],
        #   [0, 9, 8]]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_square_neg")

    @staticmethod
    def export_triu_one_row() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
        k = np.array(1).astype(np.int64)
        # X:
        # [[[1, 4, 9, 7, 1]],
        #
        #  [[9, 2, 8, 8, 4]],
        #
        #  [[3, 9, 7, 4, 2]]]
        # expect result:
        # [[[0, 4, 9, 7, 1]],
        #
        #  [[0, 2, 8, 8, 4]],
        #
        #  [[0, 9, 7, 4, 2]]]
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_one_row")

    @staticmethod
    def export_triu_zero() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
        )

        x = np.random.randint(10, size=(0, 5)).astype(np.int64)
        k = np.array(6).astype(np.int64)
        # X:
        # []
        # expect result:
        # []
        y = triu_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_zero")

    @staticmethod
    def export_tril() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 0, 0, 0, 0],
        #   [1, 2, 0, 0, 0],
        #   [9, 4, 1, 0, 0],
        #   [4, 3, 4, 2, 0]]
        y = tril_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_tril")

    @staticmethod
    def export_tril_neg() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(-1).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[0, 0, 0, 0, 0],
        #   [1, 0, 0, 0, 0],
        #   [9, 4, 0, 0, 0],
        #   [4, 3, 4, 0, 0]]
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_neg")

    @staticmethod
    def export_tril_out_neg() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(-7).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0]]
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_out_neg")

    @staticmethod
    def export_tril_pos() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(2).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 7, 3, 0, 0],
        #   [1, 2, 8, 6, 0],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_pos")

    @staticmethod
    def export_tril_out_pos() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )
        x = np.random.randint(10, size=(4, 5)).astype(np.int64)
        k = np.array(6).astype(np.int64)
        # X:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        # expect result:
        #  [[4, 7, 3, 7, 9],
        #   [1, 2, 8, 6, 9],
        #   [9, 4, 1, 8, 7],
        #   [4, 3, 4, 2, 4]]
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_out_pos")

    @staticmethod
    def export_tril_square() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
        # X:
        # [[[0, 4, 3],
        #   [2, 0, 9],
        #   [8, 2, 5]],
        #
        #  [[2, 7, 2],
        #   [2, 6, 0],
        #   [2, 6, 5]]]
        # expect result:
        # [[[0, 0, 0],
        #   [2, 0, 0],
        #   [8, 2, 5]],
        #
        #  [[2, 0, 0],
        #   [2, 6, 0],
        #   [2, 6, 5]]]
        y = tril_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_tril_square")

    @staticmethod
    def export_tril_square_neg() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
        k = np.array(-1).astype(np.int64)
        # X:
        # [[[0, 4, 3],
        #   [2, 0, 9],
        #   [8, 2, 5]],
        #
        #  [[2, 7, 2],
        #   [2, 6, 0],
        #   [2, 6, 5]]]
        # expect result:
        # [[[0, 0, 0],
        #   [2, 0, 0],
        #   [8, 2, 0]],
        #
        #  [[0, 0, 0],
        #   [2, 0, 0],
        #   [2, 6, 0]]]
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_square_neg")

    @staticmethod
    def export_tril_one_row() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
        # X:
        # [[[6, 2, 4, 1, 6]],
        #
        #  [[8, 3, 8, 7, 0]],
        #
        #  [[2, 2, 9, 5, 9]]]
        # expect result:
        # [[[6, 0, 0, 0, 0]],
        #
        #  [[8, 0, 0, 0, 0]],
        #
        #  [[2, 0, 0, 0, 0]]]
        y = tril_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_tril_one_row_neg")

    @staticmethod
    def export_tril_zero() -> None:
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
        )

        x = np.random.randint(10, size=(3, 0, 5)).astype(np.int64)
        k = np.array(6).astype(np.int64)
        # X:
        # []
        # expect result:
        # []
        y = tril_reference_implementation(x, int(k))
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_zero")
