# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.preprocessing import normalize
from ..algebra.complex_functions import onnx_cdist
from ..algebra.onnx_ops import (
    OnnxMatMul,
    OnnxTranspose,
    OnnxDiv,
    OnnxSub,
    OnnxAdd,
    OnnxMul,
    OnnxPow,
    OnnxTanh,
    OnnxSqrt,
    OnnxExp,
    OnnxReduceSumApi11,
)
from ..algebra.onnx_operator import OnnxSubEstimator
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type


def kernel_centerer_converter(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    op_version = container.target_opset
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)

    N = np.array([op.K_fit_rows_.shape[0]], dtype=dtype)
    K_pred_cols = OnnxDiv(
        OnnxReduceSumApi11(X, axes=[1], op_version=op_version), N, op_version=op_version
    )

    # K -= self.K_fit_rows_
    # K -= K_pred_cols
    # K += self.K_fit_all_
    K1 = OnnxSub(X, op.K_fit_rows_.astype(dtype), op_version=op_version)
    K2 = OnnxSub(K1, K_pred_cols, op_version=op_version)
    final = OnnxAdd(
        K2,
        np.array([op.K_fit_all_], dtype=dtype),
        op_version=op_version,
        output_names=operator.outputs[:1],
    )
    final.add_to(scope, container)


def kernel_pca_converter(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    op_version = container.target_opset
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    options = container.get_options(op, dict(optim=None))
    optim = options["optim"]

    # def _get_kernel(self, X, Y=None):
    # return pairwise_kernels(
    #         X, Y, metric=self.kernel, filter_params=True, **params)
    if callable(op.kernel):
        raise RuntimeError(
            "Unable to converter KernelPCA with a custom kernel %r." "" % op.kernel
        )
    if op.kernel == "precomputed":
        raise RuntimeError(
            "The converter is not implemented when kernel=%r for "
            "type=%r." % (op.kernel, type(op))
        )

    kernel = op.kernel
    params = {"gamma": op.gamma, "degree": op.degree, "coef0": op.coef0}

    if kernel == "linear":
        Y = op.X_fit_.astype(dtype)
        dist = OnnxMatMul(
            X,
            OnnxTranspose(Y, perm=[1, 0], op_version=op_version),
            op_version=op_version,
        )
    elif kernel == "cosine":
        yn = normalize(op.X_fit_, copy=True)
        ynt = yn.astype(dtype)
        norm = OnnxSqrt(
            OnnxReduceSumApi11(
                OnnxPow(X, np.array([2], dtype=np.int64), op_version=op_version),
                axes=[1],
                op_version=op_version,
                keepdims=1,
            ),
            op_version=op_version,
        )
        dist = OnnxMatMul(
            OnnxDiv(X, norm, op_version=op_version),
            OnnxTranspose(ynt, perm=[1, 0], op_version=op_version),
            op_version=op_version,
        )
    elif kernel in ("poly", "sigmoid"):
        Y = op.X_fit_.astype(dtype)
        dot = OnnxMatMul(
            X,
            OnnxTranspose(Y, perm=[1, 0], op_version=op_version),
            op_version=op_version,
        )
        if params["gamma"] is None:
            gamma = np.array([1.0 / Y.shape[1]], dtype=dtype)
        else:
            gamma = np.array([params["gamma"]], dtype=dtype)
        dot_g = OnnxMul(dot, gamma, op_version=op_version)
        dot_c = OnnxAdd(
            dot_g, np.array([params["coef0"]], dtype=dtype), op_version=op_version
        )
        if kernel == "poly":
            dist = OnnxPow(
                dot_c,
                np.array([params["degree"]], dtype=np.int64),
                op_version=op_version,
            )
        else:
            dist = OnnxTanh(dot_c, op_version=op_version)
    elif kernel == "rbf":
        if optim == "cdist":
            from skl2onnx.algebra.custom_ops import OnnxCDist

            Y = op.X_fit_.astype(dtype)
            pair = OnnxCDist(X, Y, metric="sqeuclidean", op_version=op_version)
        elif optim is None:
            Y = op.X_fit_.astype(dtype)
            dim_in = Y.shape[1] if hasattr(Y, "shape") else None
            dim_out = Y.shape[0] if hasattr(Y, "shape") else None
            pair = onnx_cdist(
                X,
                Y,
                metric="sqeuclidean",
                dtype=dtype,
                op_version=op_version,
                dim_in=dim_in,
                dim_out=dim_out,
            )
        else:
            raise ValueError("Unknown optimisation '{}'.".format(optim))
        if params["gamma"] is None:
            gamma = np.array([-1.0 / Y.shape[1]], dtype=dtype)
        else:
            gamma = np.array([-params["gamma"]], dtype=dtype)
        pair_g = OnnxMul(pair, gamma, op_version=op_version)
        dist = OnnxExp(pair_g, op_version=op_version)
    else:
        raise ValueError("Unknown kernel '{}'.".format(kernel))

    #  K = self._centerer.transform(self._get_kernel(X, self.X_fit_))
    K = OnnxSubEstimator(op._centerer, dist, op_version=op_version)

    if hasattr(op, "eigenvalues_"):
        # scikit-learn>=1.0
        non_zeros = np.flatnonzero(op.eigenvalues_)
        scaled_alphas = np.zeros_like(op.eigenvectors_)
        scaled_alphas[:, non_zeros] = op.eigenvectors_[:, non_zeros] / np.sqrt(
            op.eigenvalues_[non_zeros]
        )
    else:
        # scikit-learn<1.0
        non_zeros = np.flatnonzero(op.lambdas_)
        scaled_alphas = np.zeros_like(op.alphas_)
        scaled_alphas[:, non_zeros] = op.alphas_[:, non_zeros] / np.sqrt(
            op.lambdas_[non_zeros]
        )

    # np.dot(K, scaled_alphas)
    output = OnnxMatMul(
        K,
        scaled_alphas.astype(dtype),
        op_version=op_version,
        output_names=operator.outputs[:1],
    )

    # register the output
    output.add_to(scope, container)


register_converter("SklearnKernelCenterer", kernel_centerer_converter)
register_converter(
    "SklearnKernelPCA", kernel_pca_converter, options={"optim": [None, "cdist"]}
)
