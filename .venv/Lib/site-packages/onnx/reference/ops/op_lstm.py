# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from onnx.reference.op_run import OpRun


class CommonLSTM(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)
        self.n_gates = 3

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _step(
        self,
        X: np.ndarray,
        R: np.ndarray,
        B: np.ndarray,
        W: np.ndarray,
        H_0: np.ndarray,
        C_0: np.ndarray,
        P: np.ndarray,
        num_directions: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_length = X.shape[0]
        hidden_size = H_0.shape[-1]
        batch_size = X.shape[1]

        Y = np.empty([seq_length, num_directions, batch_size, hidden_size])
        h_list = []

        [p_i, p_o, p_f] = np.split(P, 3)
        H_t = H_0
        C_t = C_0
        for x in np.split(X, X.shape[0], axis=0):
            gates = (
                np.dot(x, np.transpose(W))
                + np.dot(H_t, np.transpose(R))
                + np.add(*np.split(B, 2))
            )
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C

        concatenated = np.concatenate(h_list)
        if num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.layout == 0:  # type: ignore
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h  # type: ignore

    def _run(  # type: ignore
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        initial_c=None,
        P=None,
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction=None,
        hidden_size=None,
        input_forget=None,
        layout=None,
    ):
        # TODO: support overridden attributes.
        n_gates = 4
        number_of_peepholes = 3

        num_directions = W.shape[0]

        if num_directions == 1:
            R = np.squeeze(R, axis=0)
            W = np.squeeze(W, axis=0)
            if B is not None and len(B.shape) > 0 and B.shape[0] == 1:
                B = np.squeeze(B, axis=0)
            if (
                sequence_lens is not None
                and len(sequence_lens.shape) > 0
                and sequence_lens.shape[0] == 1
            ):
                sequence_lens = np.squeeze(sequence_lens, axis=0)
            if (
                initial_h is not None
                and len(initial_h.shape) > 0
                and initial_h.shape[0] == 1
            ):
                initial_h = np.squeeze(initial_h, axis=0)
            if (
                initial_c is not None
                and len(initial_c.shape) > 0
                and initial_c.shape[0] == 1
            ):
                initial_c = np.squeeze(initial_c, axis=0)
            if P is not None and len(P.shape) > 0 and P.shape[0] == 1:
                P = np.squeeze(P, axis=0)

            hidden_size = R.shape[-1]
            batch_size = X.shape[1]

            if self.layout != 0:  # type: ignore
                X = np.swapaxes(X, 0, 1)
            if B is None:
                B = np.zeros(2 * n_gates * hidden_size, dtype=np.float32)
            if P is None:
                P = np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            if initial_h is None:
                initial_h = np.zeros((batch_size, hidden_size), dtype=np.float32)
            if initial_c is None:
                initial_c = np.zeros((batch_size, hidden_size), dtype=np.float32)
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Unsupported value {num_directions!r} for num_directions "
                f"and operator {self.__class__.__name__!r}."
            )

        Y, Y_h = self._step(
            X, R, B, W, initial_h, initial_c, P, num_directions=num_directions
        )
        Y = Y.astype(X.dtype)
        return (Y,) if self.n_outputs == 1 else (Y, Y_h.astype(X.dtype))  # type: ignore


class LSTM(CommonLSTM):
    def __init__(self, onnx_node, run_params):  # type: ignore
        CommonLSTM.__init__(self, onnx_node, run_params)
