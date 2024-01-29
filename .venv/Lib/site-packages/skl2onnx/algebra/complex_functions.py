# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from collections import OrderedDict
import numpy as np
from ..common.data_types import FloatTensorType, DoubleTensorType
from ..common.utils import get_unique_subgraph
from .onnx_ops import (
    OnnxAbs,
    OnnxDiv,
    OnnxIdentity,
    OnnxMatMul,
    OnnxPow,
    OnnxScan,
    OnnxSqrt,
    OnnxSub,
    OnnxReduceSumApi11,
    OnnxReduceSumSquareApi18,
    OnnxTranspose,
)

logger = getLogger("skl2onnx")


def onnx_squareform_pdist(
    X, metric="sqeuclidean", dtype=None, op_version=None, **kwargs
):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric=metric))``.
    """
    if metric == "sqeuclidean":
        return _onnx_squareform_pdist_sqeuclidean(
            X, dtype=dtype, op_version=op_version, **kwargs
        )
    if metric == "euclidean":
        res = _onnx_squareform_pdist_sqeuclidean(X, dtype=dtype, op_version=op_version)
        return OnnxSqrt(res, op_version=op_version, **kwargs)
    raise NotImplementedError("metric='{}' is not implemented.".format(metric))


def _onnx_squareform_pdist_sqeuclidean(X, dtype=None, op_version=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric='sqeuclidean'))``.
    """
    unique = get_unique_subgraph()
    diff = OnnxSub("next_in", "next", op_version=op_version)
    id_next = OnnxIdentity("next_in", output_names=["next_out"], op_version=op_version)
    flat = OnnxReduceSumSquareApi18(
        diff, axes=[1], op_version=op_version, output_names=["scan_out"], keepdims=0
    )
    flat.set_onnx_name_prefix("cflat_%d" % unique)
    id_next.set_onnx_name_prefix("pdistsqe_%d" % unique)
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    scan_body = id_next.to_onnx(
        OrderedDict(
            [("next_in", tensor_type([None, None])), ("next", tensor_type([None]))]
        ),
        outputs=[
            ("next_out", tensor_type([None, None])),
            ("scan_out", tensor_type([None])),
        ],
        other_outputs=[flat],
        target_opset=op_version,
    )

    node = OnnxScan(
        X,
        X,
        output_names=["u(scan0)", "u(scan1)"],
        num_scan_inputs=1,
        body=(scan_body.graph, [id_next, flat]),
        op_version=op_version,
        **kwargs,
    )
    logger.debug("[_onnx_squareform_pdist_sqeuclidean] +Scan dtype=%r", dtype)
    return node[1]


def onnx_cdist(
    XA,
    XB,
    metric="sqeuclidean",
    dtype=None,
    op_version=None,
    dim_in=None,
    dim_out=None,
    **kwargs,
):
    """
    Returns the ONNX graph which computes
    ``cdist(XA, XB, metric=metric)``.

    :param XA: array or OnnxOperatorMixin
    :param XB: array or OnnxOperatorMixin
    :param metric: distance type
    :param dtype: *np.float32* or *np.float64*
    :param op_version: opset version
    :param dim_in: dimension of the input vectorial space
        (if known)
    :param dim_out: dimension of the output vectorial space
        (if known)
    :param kwargs: addition parameter
    :return: OnnxOperatorMixin
    """
    if metric == "sqeuclidean":
        return _onnx_cdist_sqeuclidean(
            XA,
            XB,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs,
        )
    if metric == "euclidean":
        res = _onnx_cdist_sqeuclidean(
            XA, XB, dtype=dtype, op_version=op_version, dim_in=dim_in, dim_out=dim_out
        )
        return OnnxSqrt(res, op_version=op_version, **kwargs)
    if metric == "minkowski":
        p = kwargs.pop("p")
        res = _onnx_cdist_minkowski(
            XA,
            XB,
            dtype=dtype,
            op_version=op_version,
            p=p,
            dim_in=dim_in,
            dim_out=dim_out,
        )
        return OnnxPow(
            res, np.array([1.0 / p], dtype=dtype), op_version=op_version, **kwargs
        )
    if metric in ("manhattan", "cityblock"):
        return _onnx_cdist_manhattan(
            XA,
            XB,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs,
        )
    if metric == "cosine":
        return _onnx_cdist_cosine(
            XA,
            XB,
            dtype=dtype,
            op_version=op_version,
            dim_in=dim_in,
            dim_out=dim_out,
            **kwargs,
        )
    raise NotImplementedError(f"metric={metric!r} is not implemented.")


def _onnx_cdist_begin(op_version):
    diff = OnnxSub("next_in", "next", op_version=op_version)
    id_next = OnnxIdentity("next_in", output_names=["next_out"], op_version=op_version)
    return diff, id_next


def _onnx_cdist_end(
    XA, XB, id_next, flat, dtype, op_version, dim_in=None, dim_out=None, **kwargs
):
    unique = get_unique_subgraph()
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix("cdistd_%d" % unique)
    flat.set_onnx_name_prefix("cdistdf_%d" % unique)
    shape_in = (
        tensor_type([None, None]) if dim_in is None else tensor_type([None, dim_in])
    )
    scan_body = id_next.to_onnx(
        OrderedDict([("next_in", shape_in), ("next", tensor_type([None]))]),
        outputs=[
            ("next_out", tensor_type([None, None])),
            ("scan_out", tensor_type([None])),
        ],
        other_outputs=[flat],
        target_opset=op_version,
    )
    logger.debug(
        "[_onnx_cdist_end] + Scan dim_in=%r dim_out=%r dtype=%r", dim_in, dim_out, dtype
    )

    node = OnnxScan(
        XA,
        XB,
        output_names=["u(scan0)", "u(scan1)"],
        num_scan_inputs=1,
        body=(scan_body.graph, [id_next, flat]),
        op_version=op_version,
    )
    return OnnxTranspose(node[1], perm=[1, 0], op_version=op_version, **kwargs)


def _onnx_cdist_sqeuclidean(
    XA, XB, dtype=None, op_version=None, dim_in=None, dim_out=None, **kwargs
):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='sqeuclidean')``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    norm = OnnxReduceSumSquareApi18(diff, axes=[1], keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=["scan_out"], op_version=op_version)
    return _onnx_cdist_end(
        XA,
        XB,
        id_next,
        flat,
        dtype,
        op_version,
        dim_in=dim_in,
        dim_out=dim_out,
        **kwargs,
    )


def _onnx_cdist_minkowski(
    XA, XB, dtype=None, op_version=None, p=2, dim_in=None, dim_out=None, **kwargs
):
    """
    Returns the ONNX graph which computes the Minkowski distance
    or ``minkowski(XA, XB, p)``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    diff_pow = OnnxPow(
        OnnxAbs(diff, op_version=op_version),
        np.array([p], dtype=dtype),
        op_version=op_version,
    )
    norm = OnnxReduceSumApi11(diff_pow, axes=[1], keepdims=0, op_version=op_version)
    norm.set_onnx_name_prefix("norm_%d" % id(norm))
    flat = OnnxIdentity(norm, output_names=["scan_out"], op_version=op_version)
    return _onnx_cdist_end(
        XA,
        XB,
        id_next,
        flat,
        dtype,
        op_version,
        dim_in=dim_in,
        dim_out=dim_out,
        **kwargs,
    )


def _onnx_cdist_manhattan(
    XA, XB, dtype=None, op_version=None, dim_in=None, dim_out=None, **kwargs
):
    """
    Returns the ONNX graph which computes the Manhattan distance
    or ``Manhattan(X, Y)``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    diff_pow = OnnxAbs(diff, op_version=op_version)
    norm = OnnxReduceSumApi11(diff_pow, axes=[1], keepdims=0, op_version=op_version)
    norm.set_onnx_name_prefix("norm_%d" % id(norm))
    flat = OnnxIdentity(norm, output_names=["scan_out"], op_version=op_version)
    return _onnx_cdist_end(
        XA,
        XB,
        id_next,
        flat,
        dtype,
        op_version,
        dim_in=dim_in,
        dim_out=dim_out,
        **kwargs,
    )


def _onnx_cdist_cosine(
    XA, XB, dtype=None, op_version=None, dim_in=None, dim_out=None, **kwargs
):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='cosine')``.
    """
    txb = OnnxTranspose(XB, perm=[1, 0], op_version=op_version)
    scal = OnnxMatMul(XA, txb, op_version=op_version)
    norma = OnnxSqrt(
        OnnxReduceSumSquareApi18(XA, axes=[1], keepdims=1, op_version=op_version),
        op_version=op_version,
    )
    normb = OnnxSqrt(
        OnnxReduceSumSquareApi18(txb, axes=[0], keepdims=1, op_version=op_version),
        op_version=op_version,
    )
    return OnnxSub(
        np.array([1], dtype=dtype),
        OnnxDiv(
            scal,
            OnnxMatMul(norma, normb, op_version=op_version),
            op_version=op_version,
        ),
        op_version=op_version,
        **kwargs,
    )
