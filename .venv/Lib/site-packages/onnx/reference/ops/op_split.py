# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class CommonSplit(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def common_run(self, mat, split, axis, num_outputs):  # type: ignore
        n_outputs = num_outputs or self.n_outputs
        if split is None:
            if mat.shape[axis] % n_outputs == 0:
                div = mat.shape[axis] // n_outputs
                split = [div] * n_outputs
            else:
                div = mat.shape[axis] // n_outputs + 1
                split = [div] * n_outputs
                split[-1] += mat.shape[axis] - sum(split)  # type: ignore

        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split:
            sli[axis] = slice(pos, pos + spl)  # type: ignore
            pos += spl
            res.append(mat[tuple(sli)])
        return tuple(res)


class Split_2(CommonSplit):
    def _run(self, mat, axis=None, split=None):  # type: ignore
        return self.common_run(mat, split, axis=axis, num_outputs=None)  # type: ignore


class Split_11(Split_2):
    pass


class Split_13(CommonSplit):
    def _run(self, mat, split=None, axis=None):  # type: ignore
        return self.common_run(mat, split, axis=axis, num_outputs=None)


class Split_18(CommonSplit):
    def _run(self, mat, split=None, axis=None, num_outputs=None):  # type: ignore
        return self.common_run(mat, split, axis=axis, num_outputs=num_outputs)
