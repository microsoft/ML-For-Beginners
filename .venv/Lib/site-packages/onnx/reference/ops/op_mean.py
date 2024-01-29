# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class Mean(OpRun):
    def _run(self, *args):  # type: ignore
        res = args[0].copy()
        for m in args[1:]:
            res += m
        return ((res / len(args)).astype(args[0].dtype),)
