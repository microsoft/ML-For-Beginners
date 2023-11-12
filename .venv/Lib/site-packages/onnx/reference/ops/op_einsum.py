# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E0203,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Einsum(OpRun):
    def _run(self, *args, equation=None):  # type: ignore
        if not isinstance(equation, str):
            raise TypeError(f"equation must be string but is {type(equation)!r}.")
        equation = equation.strip()
        if len(equation) == 0:
            raise TypeError("equation is empty.")
        try:
            return (np.einsum(equation, *args, optimize=True),)
        except TypeError:
            return (np.einsum(equation, *args),)
