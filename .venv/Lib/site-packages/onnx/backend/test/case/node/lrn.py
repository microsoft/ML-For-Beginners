# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LRN(Base):
    @staticmethod
    def export() -> None:
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        nsize = 3
        node = onnx.helper.make_node(
            "LRN",
            inputs=["x"],
            outputs=["y"],
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=nsize,
        )
        x = np.random.randn(5, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[
                    n,
                    max(0, c - int(math.floor((nsize - 1) / 2))) : min(
                        5, c + int(math.ceil((nsize - 1) / 2)) + 1
                    ),
                    h,
                    w,
                ]
                ** 2
            )
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y], name="test_lrn")

    @staticmethod
    def export_default() -> None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        nsize = 3
        node = onnx.helper.make_node("LRN", inputs=["x"], outputs=["y"], size=3)
        x = np.random.randn(5, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[
                    n,
                    max(0, c - int(math.floor((nsize - 1) / 2))) : min(
                        5, c + int(math.ceil((nsize - 1) / 2)) + 1
                    ),
                    h,
                    w,
                ]
                ** 2
            )
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y], name="test_lrn_default")
