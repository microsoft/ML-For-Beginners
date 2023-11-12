# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Transpose(Base):
    @staticmethod
    def export_default() -> None:
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)

        node = onnx.helper.make_node(
            "Transpose", inputs=["data"], outputs=["transposed"]
        )

        transposed = np.transpose(data)
        expect(node, inputs=[data], outputs=[transposed], name="test_transpose_default")

    @staticmethod
    def export_all_permutations() -> None:
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)
        permutations = list(itertools.permutations(np.arange(len(shape))))

        for i, permutation in enumerate(permutations):
            node = onnx.helper.make_node(
                "Transpose",
                inputs=["data"],
                outputs=["transposed"],
                perm=permutation,
            )
            transposed = np.transpose(data, permutation)
            expect(
                node,
                inputs=[data],
                outputs=[transposed],
                name=f"test_transpose_all_permutations_{i}",
            )
