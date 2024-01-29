# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class OpRunTraining(OpRun):
    op_domain = "ai.onnx.preview.training"
