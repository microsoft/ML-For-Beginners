# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class CommonGRU(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)
        self.number_of_gates = 3

    def f(self, x):  # type: ignore
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: ignore
        return np.tanh(x)

    def _step(self, X, R, B, W, H_0, num_directions):  # type: ignore
        seq_length = X.shape[0]
        hidden_size = H_0.shape[-1]
        batch_size = X.shape[1]

        Y = np.empty([seq_length, num_directions, batch_size, hidden_size])
        h_list = []

        [w_z, w_r, w_h] = np.split(W, 3)
        [r_z, r_r, r_h] = np.split(R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = H_0
        for x in np.split(X, X.shape[0], axis=0):
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
            h = h_linear if self.linear_before_reset else h_default  # type: ignore
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H

        concatenated = np.concatenate(h_list)
        if num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.layout == 0:  # type: ignore
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h

    def _run(  # type: ignore
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction=None,
        hidden_size=None,
        layout=None,
        linear_before_reset=None,
    ):
        # TODO: support overridden attributes.
        num_directions = W.shape[0]

        if num_directions == 1:
            R = np.squeeze(R, axis=0)
            W = np.squeeze(W, axis=0)
            if B is not None:
                B = np.squeeze(B, axis=0)
            if sequence_lens is not None:
                sequence_lens = np.squeeze(sequence_lens, axis=0)
            if initial_h is not None:
                initial_h = np.squeeze(initial_h, axis=0)

            hidden_size = R.shape[-1]
            batch_size = X.shape[1]

            X = X if layout == 0 else np.swapaxes(X, 0, 1)
            b = (
                B
                if B is not None
                else np.zeros(2 * self.number_of_gates * hidden_size, dtype=X.dtype)
            )
            h_0 = (
                initial_h
                if initial_h is not None
                else np.zeros((batch_size, hidden_size), dtype=X.dtype)
            )

            B = b
            H_0 = h_0
        else:
            raise NotImplementedError(
                f"Unsupported value {num_directions} for num_directions and operator "
                f"{self.__class__.__name__!r}."
            )

        Y, Y_h = self._step(X, R, B, W, H_0, num_directions=num_directions)
        Y = Y.astype(X.dtype)
        return (Y,) if self.n_outputs == 1 else (Y, Y_h.astype(X.dtype))


class GRU(CommonGRU):
    def __init__(self, onnx_node, run_params):  # type: ignore
        CommonGRU.__init__(self, onnx_node, run_params)  # type: ignore
