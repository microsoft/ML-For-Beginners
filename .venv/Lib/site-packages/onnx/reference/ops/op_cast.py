# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,W0221

import numpy as np

from onnx.helper import (
    float32_to_bfloat16,
    float32_to_float8e4m3,
    float32_to_float8e5m2,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import (
    bfloat16_to_float32,
    float8e4m3_to_float32,
    float8e5m2_to_float32,
)
from onnx.onnx_pb import TensorProto
from onnx.reference.custom_element_types import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun


def cast_to(x, to, saturate):
    if x.dtype == bfloat16 and x.dtype.descr[0][0] == "bfloat16":
        if to == TensorProto.BFLOAT16:
            return x
        xr = x.ravel()
        xf = np.empty(xr.shape[0], dtype=np.float32)
        for i in range(xr.shape[0]):
            el = bfloat16_to_float32(xr[i])
            xf[i] = el
        dtype = tensor_dtype_to_np_dtype(to)
        return xf.astype(dtype).reshape(x.shape)

    f8 = {
        (float8e4m3fn, "e4m3fn", TensorProto.FLOAT8E4M3FN): float8e4m3_to_float32,
        (
            float8e4m3fnuz,
            "e4m3fnuz",
            TensorProto.FLOAT8E4M3FNUZ,
        ): lambda *args: float8e4m3_to_float32(*args, uz=True),
        (float8e5m2, "e5m2", TensorProto.FLOAT8E5M2): float8e5m2_to_float32,
        (
            float8e5m2fnuz,
            "e5m2fnuz",
            TensorProto.FLOAT8E5M2FNUZ,
        ): lambda *args: float8e5m2_to_float32(*args, fn=True, uz=True),
    }

    for (dt, st, proto_type), cvt in f8.items():
        if x.dtype == dt and x.dtype.descr[0][0] == st:
            if to == proto_type:
                return x
            xr = x.ravel()
            xf = np.empty(xr.shape[0], dtype=np.float32)
            for i in range(xr.shape[0]):
                el = cvt(xr[i])
                xf[i] = el
            dtype = tensor_dtype_to_np_dtype(to)
            return xf.astype(dtype).reshape(x.shape)

    if to == TensorProto.BFLOAT16:
        xf = x.astype(np.float32).ravel()
        y = np.empty(xf.shape, dtype=bfloat16).ravel()
        for i in range(y.shape[0]):
            el = float32_to_bfloat16(xf[i], truncate=True)  # type: ignore[assignment]
            y[i] = el
        return y.reshape(x.shape)

    f8back = {
        TensorProto.FLOAT8E4M3FN: (
            float8e4m3fn,
            lambda *args: float32_to_float8e4m3(*args, saturate=saturate),
        ),
        TensorProto.FLOAT8E4M3FNUZ: (
            float8e4m3fnuz,
            lambda *args: float32_to_float8e4m3(*args, uz=True, saturate=saturate),
        ),
        TensorProto.FLOAT8E5M2: (
            float8e5m2,
            lambda *args: float32_to_float8e5m2(*args, saturate=saturate),
        ),
        TensorProto.FLOAT8E5M2FNUZ: (
            float8e5m2fnuz,
            lambda *args: float32_to_float8e5m2(
                *args, fn=True, uz=True, saturate=saturate
            ),
        ),
    }
    for dt, (npdt, cvt) in f8back.items():
        if to == dt:
            xf = x.astype(np.float32).ravel()
            y = np.empty(xf.shape, dtype=npdt).ravel()
            for i in range(y.shape[0]):
                el = cvt(xf[i])  # type: ignore[assignment]
                y[i] = el
            return y.reshape(x.shape)

    if to == TensorProto.STRING:
        return x.astype(np.str_)

    dtype = tensor_dtype_to_np_dtype(to)
    return x.astype(dtype)


class Cast_1(OpRun):
    def _run(self, x, to=None):  # type: ignore
        return (cast_to(x, to, saturate=True),)


class Cast_19(OpRun):
    def _run(self, x, to=None, saturate=None):  # type: ignore
        return (cast_to(x, to, saturate),)
