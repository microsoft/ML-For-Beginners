# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class SpaceToDepth(Base):
    @staticmethod
    def export() -> None:
        b, c, h, w = shape = (2, 2, 6, 6)
        blocksize = 2
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=blocksize,
        )
        x = np.random.random_sample(shape).astype(np.float32)
        tmp = np.reshape(
            x, [b, c, h // blocksize, blocksize, w // blocksize, blocksize]
        )
        tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize, w // blocksize])
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth")

    @staticmethod
    def export_example() -> None:
        node = onnx.helper.make_node(
            "SpaceToDepth",
            inputs=["x"],
            outputs=["y"],
            blocksize=2,
        )

        # (1, 1, 4, 6) input tensor
        x = np.array(
            [
                [
                    [
                        [0, 6, 1, 7, 2, 8],
                        [12, 18, 13, 19, 14, 20],
                        [3, 9, 4, 10, 5, 11],
                        [15, 21, 16, 22, 17, 23],
                    ]
                ]
            ]
        ).astype(np.float32)

        # (1, 4, 2, 3) output tensor
        y = np.array(
            [
                [
                    [[0, 1, 2], [3, 4, 5]],
                    [[6, 7, 8], [9, 10, 11]],
                    [[12, 13, 14], [15, 16, 17]],
                    [[18, 19, 20], [21, 22, 23]],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_spacetodepth_example")
