# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class GRUHelper:
    def __init__(self, **params: Any) -> None:
        # GRU Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        LBR = "linear_before_reset"
        LAYOUT = "layout"
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params[LAYOUT] if LAYOUT in params else 0
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = (
                params[B]
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size)
            )
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                np.dot(x, np.transpose(w_h))
                + np.dot(r * H_t, np.transpose(r_h))
                + w_bh
                + r_bh
            )
            h_linear = self.g(
                np.dot(x, np.transpose(w_h))
                + r * (np.dot(H_t, np.transpose(r_h)) + r_bh)
                + w_bh
            )
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h


class GRU(Base):
    @staticmethod
    def export_defaults() -> None:
        input = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 5
        weight_scale = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            "GRU", inputs=["X", "W", "R"], outputs=["", "Y_h"], hidden_size=hidden_size
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        gru = GRUHelper(X=input, W=W, R=R)
        _, Y_h = gru.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y_h.astype(np.float32)],
            name="test_gru_defaults",
        )

    @staticmethod
    def export_initial_bias() -> None:
        input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )

        input_size = 3
        hidden_size = 3
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 3

        node = onnx.helper.make_node(
            "GRU",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
            np.float32
        )
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        gru = GRUHelper(X=input, W=W, R=R, B=B)
        _, Y_h = gru.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name="test_gru_with_initial_bias",
        )

    @staticmethod
    def export_seq_length() -> None:
        input = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ).astype(np.float32)

        input_size = 3
        hidden_size = 5
        number_of_gates = 3

        node = onnx.helper.make_node(
            "GRU",
            inputs=["X", "W", "R", "B"],
            outputs=["", "Y_h"],
            hidden_size=hidden_size,
        )

        W = np.random.randn(1, number_of_gates * hidden_size, input_size).astype(
            np.float32
        )
        R = np.random.randn(1, number_of_gates * hidden_size, hidden_size).astype(
            np.float32
        )

        # Adding custom bias
        W_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
        R_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)

        gru = GRUHelper(X=input, W=W, R=R, B=B)
        _, Y_h = gru.step()
        expect(
            node,
            inputs=[input, W, R, B],
            outputs=[Y_h.astype(np.float32)],
            name="test_gru_seq_length",
        )

    @staticmethod
    def export_batchwise() -> None:
        input = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 6
        number_of_gates = 3
        weight_scale = 0.2
        layout = 1

        node = onnx.helper.make_node(
            "GRU",
            inputs=["X", "W", "R"],
            outputs=["Y", "Y_h"],
            hidden_size=hidden_size,
            layout=layout,
        )

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float32)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float32)

        gru = GRUHelper(X=input, W=W, R=R, layout=layout)
        Y, Y_h = gru.step()
        expect(
            node,
            inputs=[input, W, R],
            outputs=[Y.astype(np.float32), Y_h.astype(np.float32)],
            name="test_gru_batchwise",
        )
