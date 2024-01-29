# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class ReverseSequence(OpRun):
    def _run(self, data, sequence_lens, batch_axis=None, time_axis=None):  # type: ignore
        index = [slice(0, s) for s in data.shape]
        index_data = [slice(0, s) for s in data.shape]
        result = data.copy()
        for i, sl in enumerate(sequence_lens):
            index[batch_axis] = i  # type: ignore
            index[time_axis] = slice(0, sl)
            index_data[batch_axis] = i  # type: ignore
            index_data[time_axis] = slice(sl - 1, None, -1)  # type: ignore
            result[tuple(index)] = data[tuple(index_data)]
        return (result,)
