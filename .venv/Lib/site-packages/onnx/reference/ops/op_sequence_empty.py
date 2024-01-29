# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class SequenceEmpty(OpRun):
    def _run(self, dtype=None):  # type: ignore
        return ([],)
