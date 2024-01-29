# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op_common_window import _CommonWindow


class BlackmanWindow(_CommonWindow):
    """
    Returns
    :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) +
    0.08 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `blackman_window
    <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_
    """

    def _run(self, size, output_datatype=None, periodic=None):  # type: ignore
        ni, N_1 = np.arange(size), size
        if periodic == 0:
            N_1 = N_1 - 1
        alpha = 0.42
        beta = 0.08
        pi = np.pi
        y = np.cos((ni * (pi * 2)) / N_1) * (-0.5)
        y += np.cos((ni * (pi * 4)) / N_1) * beta
        y += alpha
        return self._end(size, y, output_datatype)
