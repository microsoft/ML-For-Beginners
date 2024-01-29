# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class SequenceErase(OpRun):
    def _run(self, S, ind=None):  # type: ignore
        if ind is None:
            ind = -1
        else:
            ind = int(ind)
        S2 = S.copy()
        del S2[ind]
        return (S2,)
