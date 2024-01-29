# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops._op import OpRunUnaryNum


class Identity(OpRunUnaryNum):
    def _run(self, a):  # type: ignore
        if a is None:
            return (None,)
        return (a.copy(),)
