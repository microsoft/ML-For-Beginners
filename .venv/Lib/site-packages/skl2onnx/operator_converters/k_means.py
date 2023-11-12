# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.utils.extmath import row_norms
from ..common.data_types import Int64TensorType, guess_numpy_type
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import (
    OnnxReduceSumSquareApi18,
    OnnxGemm,
    OnnxMatMul,
    OnnxAdd,
    OnnxArgMin,
    OnnxCast,
    OnnxSqrt,
    OnnxMul,
)


def convert_sklearn_kmeans(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Computation graph of distances to all centroids for a batch of examples.
    Note that a centriod is just the center of a cluster. We use ``[]`` to
    denote the dimension of a variable; for example, ``X[3, 2]`` means that
    *X* is a *3-by-2* tensor. In addition, for a matrix *X*, $X'$ denotes its
    transpose.

    Symbols:

    * *l*: # of examples.
    * *n*: # of features per input example.
    * *X*: input examples, l-by-n tensor.
    * *C*: centroids, k-by-n tensor.
    * :math:`C^2`: 2-norm of all centriod vectors, its shape is ``[k]``.
    * *Y*: 2-norm of difference between examples and centroids,
      *l-by-k* tensor. The value at i-th row and k-th column row,
      ``Y[i,k]``,is the distance from example *i* to centroid *k*.
    * *L*: the id of the nearest centroid for each input example,
      its shape is ``[l]``.

    ::

         .------------------------------------------------------.
         |                                                      |
         |                                                      v
        X [l, n] --> ReduceSumSquare -> X^2 [l]   Gemm (alpha=-2, transB=1)
                                         |                  |  |- C [k, n]
                                         |                  |
                                         |                  v
                                         `------> Add <-- -2XC' [l, k]
                                                   |
                                                   v
                     C^2 [k] --------> Add <----- Z [l, k]
                                        |
                                        v
                 L [l] <-- ArgMin <--  Y2 [l, k] --> Sqrt --> Y2 [l, k]

    *scikit-learn* code:

    ::

        X = data
        Y = model.cluster_centers_
        XX = row_norms(X, squared=True)
        YY = row_norms(Y, squared=True)
        distances = safe_sparse_dot(X, Y.T, dense_output=True)
        distances *= -2
        distances += XX[:, numpy.newaxis]
        distances += YY[numpy.newaxis, :]
        numpy.sqrt(distances, out=distances)
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    options = container.get_options(op, dict(gemm=True))
    opv = container.target_opset
    C = op.cluster_centers_
    input_name = X
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32

    if isinstance(X.type, Int64TensorType):
        x_cast = OnnxCast(X, to=np.float32, op_version=opv)
        input_name = x_cast

    C2 = row_norms(C, squared=True).astype(dtype)
    C = C.astype(dtype)
    rs = OnnxReduceSumSquareApi18(input_name, axes=[1], keepdims=1, op_version=opv)

    if options["gemm"]:
        N = X.get_first_dimension()
        if isinstance(N, int):
            zeros = np.zeros((N,), dtype=dtype)
        else:
            zeros = OnnxMul(rs, np.array([0], dtype=dtype), op_version=opv)
        gemm_out = OnnxGemm(input_name, C, zeros, alpha=-2.0, transB=1, op_version=opv)
    else:
        gemm_out = OnnxMatMul(input_name, (C.T * (-2)).astype(dtype), op_version=opv)

    z = OnnxAdd(rs, gemm_out, op_version=opv)
    y2 = OnnxAdd(C2, z, op_version=opv)
    ll = OnnxArgMin(y2, axis=1, keepdims=0, output_names=out[:1], op_version=opv)
    y2s = OnnxSqrt(y2, output_names=out[1:], op_version=opv)
    ll.add_to(scope, container)
    y2s.add_to(scope, container)


register_converter(
    "SklearnKMeans", convert_sklearn_kmeans, options={"gemm": [True, False]}
)
register_converter(
    "SklearnMiniBatchKMeans", convert_sklearn_kmeans, options={"gemm": [True, False]}
)
