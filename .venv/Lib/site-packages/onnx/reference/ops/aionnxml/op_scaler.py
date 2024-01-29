# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Scaler(OpRunAiOnnxMl):
    def _run(self, x, offset=None, scale=None):  # type: ignore
        dx = x - offset
        return ((dx * scale).astype(x.dtype),)
