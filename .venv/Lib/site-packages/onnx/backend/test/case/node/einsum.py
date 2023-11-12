# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def einsum_reference_implementation(
    Eqn: str, Operands: Tuple[np.ndarray, ...]
) -> np.ndarray:
    Z = np.einsum(Eqn, *Operands)
    return Z


class Einsum(Base):
    @staticmethod
    def export_einsum_transpose() -> None:
        Eqn = "ij->ji"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 4)
        Y = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Y], name="test_einsum_transpose")

    @staticmethod
    def export_einsum_sum() -> None:
        Eqn = "ij->i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 4)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_sum")

    @staticmethod
    def export_einsum_batch_diagonal() -> None:
        Eqn = "...ii ->...i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x"], outputs=["y"], equation=Eqn
        )

        X = np.random.randn(3, 5, 5)
        Z = einsum_reference_implementation(Eqn, (X,))

        expect(node, inputs=[X], outputs=[Z], name="test_einsum_batch_diagonal")

    @staticmethod
    def export_einsum_inner_prod() -> None:
        Eqn = "i,i"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x", "y"], outputs=["z"], equation=Eqn
        )

        X = np.random.randn(5)
        Y = np.random.randn(5)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[X, Y], outputs=[Z], name="test_einsum_inner_prod")

    @staticmethod
    def export_einsum_batch_matmul() -> None:
        Eqn = "bij, bjk -> bik"
        node = onnx.helper.make_node(
            "Einsum", inputs=["x", "y"], outputs=["z"], equation=Eqn
        )

        X = np.random.randn(5, 2, 3)
        Y = np.random.randn(5, 3, 4)
        Z = einsum_reference_implementation(Eqn, (X, Y))

        expect(node, inputs=[X, Y], outputs=[Z], name="test_einsum_batch_matmul")
