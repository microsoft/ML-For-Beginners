# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DepthToSpace(Base):
    @staticmethod
    def export_default_mode_example() -> None:
        node = onnx.helper.make_node(
            "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=2, mode="DCR"
        )

        # (1, 8, 2, 3) input tensor
        x = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
                ]
            ]
        ).astype(np.float32)

        # (1, 2, 4, 6) output tensor
        y = np.array(
            [
                [
                    [
                        [0.0, 18.0, 1.0, 19.0, 2.0, 20.0],
                        [36.0, 54.0, 37.0, 55.0, 38.0, 56.0],
                        [3.0, 21.0, 4.0, 22.0, 5.0, 23.0],
                        [39.0, 57.0, 40.0, 58.0, 41.0, 59.0],
                    ],
                    [
                        [9.0, 27.0, 10.0, 28.0, 11.0, 29.0],
                        [45.0, 63.0, 46.0, 64.0, 47.0, 65.0],
                        [12.0, 30.0, 13.0, 31.0, 14.0, 32.0],
                        [48.0, 66.0, 49.0, 67.0, 50.0, 68.0],
                    ],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_depthtospace_example")

    @staticmethod
    def export_crd_mode_example() -> None:
        node = onnx.helper.make_node(
            "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=2, mode="CRD"
        )

        # (1, 8, 2, 3) input tensor
        x = np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                    [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                    [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                    [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                    [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                    [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
                ]
            ]
        ).astype(np.float32)

        # (1, 2, 4, 6) output tensor
        y = np.array(
            [
                [
                    [
                        [0.0, 9.0, 1.0, 10.0, 2.0, 11.0],
                        [18.0, 27.0, 19.0, 28.0, 20.0, 29.0],
                        [3.0, 12.0, 4.0, 13.0, 5.0, 14.0],
                        [21.0, 30.0, 22.0, 31.0, 23.0, 32.0],
                    ],
                    [
                        [36.0, 45.0, 37.0, 46.0, 38.0, 47.0],
                        [54.0, 63.0, 55.0, 64.0, 56.0, 65.0],
                        [39.0, 48.0, 40.0, 49.0, 41.0, 50.0],
                        [57.0, 66.0, 58.0, 67.0, 59.0, 68.0],
                    ],
                ]
            ]
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_depthtospace_crd_mode_example")
