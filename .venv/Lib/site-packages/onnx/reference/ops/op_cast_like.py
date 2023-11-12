# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from onnx.helper import np_dtype_to_tensor_dtype
from onnx.onnx_pb import TensorProto
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_cast import (
    bfloat16,
    cast_to,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)


def _cast_like(x, y, saturate):
    if y.dtype == bfloat16:
        to = TensorProto.BFLOAT16
    elif y.dtype == float8e4m3fn and y.dtype.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
    elif y.dtype == float8e4m3fnuz and y.dtype.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
    elif y.dtype == float8e5m2 and y.dtype.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
    elif y.dtype == float8e5m2fnuz and y.dtype.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
    else:
        to = np_dtype_to_tensor_dtype(y.dtype)  # type: ignore
    return (cast_to(x, to, saturate),)


class CastLike_15(OpRun):
    def _run(self, x, y):  # type: ignore
        return _cast_like(x, y, True)


class CastLike_19(OpRun):
    def _run(self, x, y, saturate=None):  # type: ignore
        return _cast_like(x, y, saturate)
