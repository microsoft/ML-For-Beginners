# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Einsum(OpRun):
    def _run(self, *args, equation=None):  # type: ignore
        if not isinstance(equation, str):
            raise TypeError(f"equation must be string but is {type(equation)!r}.")
        equation = equation.strip()
        if not equation:
            raise TypeError("equation is empty.")
        try:
            return (np.einsum(equation, *args, optimize=True),)
        except TypeError:
            return (np.einsum(equation, *args),)
