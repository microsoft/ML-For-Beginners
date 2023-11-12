# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.special import digamma
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

try:
    from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
except ImportError:
    # scikit-learn < 0.22
    from sklearn.mixture.gaussian_mixture import _compute_log_det_cholesky
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxSub,
    OnnxMul,
    OnnxGemm,
    OnnxReduceSumSquareApi18,
    OnnxReduceLogSumExpApi18,
    OnnxExp,
    OnnxArgMax,
    OnnxConcat,
    OnnxReduceSumApi11,
    OnnxLog,
    OnnxReduceMaxApi18,
    OnnxEqual,
    OnnxCast,
)
from ..proto import onnx_proto


def _estimate_log_gaussian_prob(
    X, means, precisions_chol, covariance_type, dtype, op_version, combined_reducesum
):
    """
    Converts the same function into ONNX.
    Returns log probabilities.
    """
    n_components = means.shape[0]
    n_features = means.shape[1]
    opv = op_version

    # self._estimate_log_prob(X)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features
    ).astype(dtype)

    if covariance_type == "full":
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        #     y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        ys = []
        for c in range(n_components):
            prec_chol = precisions_chol[c, :, :]
            cst = -np.dot(means[c, :], prec_chol)
            y = OnnxGemm(
                X,
                prec_chol.astype(dtype),
                cst.astype(dtype),
                alpha=1.0,
                beta=1.0,
                op_version=opv,
            )
            if combined_reducesum:
                y2s = OnnxReduceSumApi11(
                    OnnxMul(y, y, op_version=opv), axes=[1], op_version=opv
                )
            else:
                y2s = OnnxReduceSumSquareApi18(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif covariance_type == "tied":
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_features, n_features)

        # log_prob = np.empty((n_samples, n_components))
        # for k, mu in enumerate(means):
        #     y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
        #     log_prob[:, k] = np.sum(np.square(y), axis=1)

        ys = []
        for f in range(n_components):
            cst = -np.dot(means[f, :], precisions_chol)
            y = OnnxGemm(
                X,
                precisions_chol.astype(dtype),
                cst.astype(dtype),
                alpha=1.0,
                beta=1.0,
                op_version=opv,
            )
            if combined_reducesum:
                y2s = OnnxReduceSumApi11(
                    OnnxMul(y, y, op_version=opv), axes=[1], op_version=opv
                )
            else:
                y2s = OnnxReduceSumSquareApi18(y, axes=[1], op_version=opv)
            ys.append(y2s)
        log_prob = OnnxConcat(*ys, axis=1, op_version=opv)

    elif covariance_type == "diag":
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) =
        #   (n_components, n_features)

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum((means ** 2 * precisions), 1) -
        #             2. * np.dot(X, (means * precisions).T) +
        #             np.dot(X ** 2, precisions.T))

        precisions = (precisions_chol**2).astype(dtype)
        mp = np.sum((means**2 * precisions), 1).astype(dtype)
        zeros = np.zeros((n_components,), dtype=dtype)
        xmp = OnnxGemm(
            X,
            (means * precisions).T.astype(dtype),
            zeros,
            alpha=-2.0,
            beta=0.0,
            op_version=opv,
        )
        term = OnnxGemm(
            OnnxMul(X, X, op_version=opv),
            precisions.T.astype(dtype),
            zeros,
            alpha=1.0,
            beta=0.0,
            op_version=opv,
        )
        log_prob = OnnxAdd(
            OnnxAdd(mp.astype(dtype), xmp, op_version=opv), term, op_version=opv
        )

    elif covariance_type == "spherical":
        # shape(op.means_) = (n_components, n_features)
        # shape(op.precisions_cholesky_) = (n_components, )

        # precisions = precisions_chol ** 2
        # log_prob = (np.sum(means ** 2, 1) * precisions -
        #             2 * np.dot(X, means.T * precisions) +
        #             np.outer(row_norms(X, squared=True), precisions))

        zeros = np.zeros((n_components,), dtype=dtype)
        precisions = (precisions_chol**2).astype(dtype)
        if combined_reducesum:
            normX = OnnxReduceSumApi11(
                OnnxMul(X, X, op_version=opv), axes=[1], op_version=opv
            )
        else:
            normX = OnnxReduceSumSquareApi18(X, axes=[1], op_version=opv)
        outer = OnnxGemm(
            normX,
            precisions[np.newaxis, :].astype(dtype),
            zeros.astype(dtype),
            alpha=1.0,
            beta=1.0,
            op_version=opv,
        )
        xmp = OnnxGemm(
            X,
            (means.T * precisions).astype(dtype),
            zeros,
            alpha=-2.0,
            beta=0.0,
            op_version=opv,
        )
        mp = (np.sum(means**2, 1) * precisions).astype(dtype)
        log_prob = OnnxAdd(mp, OnnxAdd(xmp, outer, op_version=opv), op_version=opv)
    else:
        raise RuntimeError(
            "Unknown op.covariance_type='{}'. Upgrade "
            "to a more recent version of skearn-onnx "
            "or raise an issue.".format(covariance_type)
        )
    # -.5 * (cst + log_prob) + log_det
    cst = np.array([n_features * np.log(2 * np.pi)]).astype(dtype)
    add = OnnxAdd(cst, log_prob, op_version=opv)
    mul = OnnxMul(add, np.array([-0.5], dtype=dtype), op_version=opv)
    if isinstance(log_det, (np.float32, np.float64, float)):
        log_det = np.array([log_det], dtype=dtype)

    return OnnxAdd(mul, log_det.astype(dtype), op_version=opv)


def convert_sklearn_gaussian_mixture(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for *GaussianMixture*,
    *BayesianGaussianMixture*.
    """
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    elif operator.target_opset < 11:
        raise RuntimeError(
            "Some needed operators are not available below opset 11"
            " to convert model %r" % type(operator.raw_operator)
        )
    out = operator.outputs
    op = operator.raw_operator
    n_components = op.means_.shape[0]
    opv = container.target_opset
    options = container.get_options(op, dict(score_samples=None))
    add_score = options.get("score_samples", False)
    combined_reducesum = not container.is_allowed(
        {"ReduceLogSumExp", "ReduceSumSquare"}
    )
    if add_score and len(out) != 3:
        raise RuntimeError("3 outputs are expected.")

    if X.type is not None:
        if X.type.shape[1] != op.means_.shape[1]:
            raise RuntimeError(
                "Dimension mismath between expected number of features {} "
                "and ONNX graphs expectations {}.".format(
                    op.means_.shape[1], X.type.shape[1]
                )
            )
    n_features = op.means_.shape[1]

    # All comments come from scikit-learn code and tells
    # which functions is being onnxified.
    # def _estimate_weighted_log_prob(self, X):
    log_weights = op._estimate_log_weights().astype(dtype)

    log_gauss = _estimate_log_gaussian_prob(
        X,
        op.means_,
        op.precisions_cholesky_,
        op.covariance_type,
        dtype,
        opv,
        combined_reducesum,
    )

    if isinstance(op, BayesianGaussianMixture):
        # log_gauss = (_estimate_log_gaussian_prob(
        #   X, self.means_, self.precisions_cholesky_, self.covariance_type) -
        #   .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5 * (op.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )
        cst_log_lambda = 0.5 * (log_lambda - n_features / op.mean_precision_)
        cst = cst_log_lambda - 0.5 * n_features * np.log(op.degrees_of_freedom_)
        if isinstance(cst, np.ndarray):
            cst_array = cst.astype(dtype)
        else:
            cst_array = np.array([cst], dtype=dtype)
        log_gauss = OnnxAdd(log_gauss, cst_array, op_version=opv)
    elif not isinstance(op, GaussianMixture):
        raise RuntimeError("The converter does not support type {}.".format(type(op)))

    # self._estimate_log_prob(X) + self._estimate_log_weights()
    weighted_log_prob = OnnxAdd(log_gauss, log_weights, op_version=opv)

    # labels
    if container.is_allowed("ArgMax"):
        labels = OnnxArgMax(
            weighted_log_prob, axis=1, output_names=out[:1], op_version=opv
        )
    else:
        mxlabels = OnnxReduceMaxApi18(weighted_log_prob, axes=[1], op_version=opv)
        zeros = OnnxEqual(
            OnnxSub(weighted_log_prob, mxlabels, op_version=opv),
            np.array([0], dtype=dtype),
            op_version=opv,
        )
        toint = OnnxCast(zeros, to=onnx_proto.TensorProto.INT64, op_version=opv)
        mulind = OnnxMul(
            toint, np.arange(n_components).astype(np.int64), op_version=opv
        )
        labels = OnnxReduceMaxApi18(
            mulind, axes=[1], output_names=out[:1], op_version=opv
        )

    # def _estimate_log_prob_resp():
    # np.exp(log_resp)
    # weighted_log_prob = self._estimate_weighted_log_prob(X)
    # log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    # with np.errstate(under='ignore'):
    #    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    if add_score:
        outnames = out[2:3]
    else:
        outnames = None

    if combined_reducesum:
        max_weight = OnnxReduceMaxApi18(weighted_log_prob, axes=[1], op_version=opv)
        log_prob_norm_demax = OnnxLog(
            OnnxReduceSumApi11(
                OnnxExp(
                    OnnxSub(weighted_log_prob, max_weight, op_version=opv),
                    op_version=opv,
                ),
                axes=[1],
                op_version=opv,
            ),
            op_version=opv,
        )
        log_prob_norm = OnnxAdd(
            log_prob_norm_demax, max_weight, op_version=opv, output_names=outnames
        )
    else:
        log_prob_norm = OnnxReduceLogSumExpApi18(
            weighted_log_prob, axes=[1], op_version=opv, output_names=outnames
        )
    log_resp = OnnxSub(weighted_log_prob, log_prob_norm, op_version=opv)

    # probabilities
    probs = OnnxExp(log_resp, output_names=out[1:2], op_version=opv)

    # final
    labels.add_to(scope, container)
    probs.add_to(scope, container)
    if add_score:
        log_prob_norm.add_to(scope, container)


register_converter(
    "SklearnGaussianMixture",
    convert_sklearn_gaussian_mixture,
    options={"score_samples": [True, False]},
)
register_converter(
    "SklearnBayesianGaussianMixture",
    convert_sklearn_gaussian_mixture,
    options={"score_samples": [True, False]},
)
