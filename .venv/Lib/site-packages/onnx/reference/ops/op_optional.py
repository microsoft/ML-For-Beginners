# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


class Optional(OpRun):
    def _run(self, x=None, type=None):  # type: ignore
        if x is not None and type is not None:
            dt = tensor_dtype_to_np_dtype(type)
            if dt != x.dtype:
                raise TypeError(
                    f"Input dtype {x.dtype} ({dt}) and parameter type_proto {type} disagree"
                )
        return ([x],)
