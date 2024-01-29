# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def _gemm00(a, b, c, alpha, beta):  # type: ignore
    o = np.dot(a, b) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm01(a, b, c, alpha, beta):  # type: ignore
    o = np.dot(a, b.T) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm10(a, b, c, alpha, beta):  # type: ignore
    o = np.dot(a.T, b) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm11(a, b, c, alpha, beta):  # type: ignore
    o = np.dot(a.T, b.T) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


class Gemm_6(OpRun):
    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None, broadcast=None):  # type: ignore
        if broadcast == 0:
            if transA:
                _meth = _gemm11 if transB else _gemm10
            else:
                _meth = _gemm01 if transB else _gemm00
            res = _meth(a, b, None, alpha, beta)
            if c is None:
                return (res.astype(a.dtype),)
            if c.shape != res.shape:
                raise ValueError(
                    f"Unable to add shape {c.shape} to shape {res.shape} without broadcast."
                )
            return (res + c,)
        if transA:
            _meth = _gemm11 if transB else _gemm10
        else:
            _meth = _gemm01 if transB else _gemm00
        return (_meth(a, b, c, alpha, beta).astype(a.dtype),)


class Gemm_7(OpRun):
    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None):  # type: ignore
        if transA:
            _meth = _gemm11 if transB else _gemm10
        else:
            _meth = _gemm01 if transB else _gemm00
        return (_meth(a, b, c, alpha, beta).astype(a.dtype),)
