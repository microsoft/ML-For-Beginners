# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

try:
    from sklearn.gaussian_process._gpc import LAMBDAS, COEFS
except ImportError:
    LAMBDAS, COEFS = None, None
from ..proto import onnx_proto
from ..common.data_types import guess_numpy_type
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxSqrt,
    OnnxMatMul,
    OnnxSub,
    OnnxReduceSumApi11,
    OnnxMul,
    OnnxMax,
    OnnxReshapeApi13,
    OnnxDiv,
    OnnxNot,
    OnnxReciprocal,
    OnnxCast,
    OnnxLess,
    OnnxPow,
    OnnxNeg,
    OnnxConcat,
    OnnxArrayFeatureExtractor,
    OnnxTranspose,
)
from ..algebra.custom_ops import OnnxSolve

try:
    from ..algebra.onnx_ops import OnnxConstantOfShape
except ImportError:
    OnnxConstantOfShape = None
try:
    from ..algebra.onnx_ops import OnnxErf
except ImportError:
    OnnxErf = None
try:
    from ..algebra.onnx_ops import OnnxEinsum
except ImportError:
    OnnxEinsum = None
from ._gp_kernels import convert_kernel_diag, convert_kernel, _zero_vector_of_size


def convert_gaussian_process_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    The method *predict* from class *GaussianProcessRegressor*
    may cache some results if it is called with parameter
    ``return_std=True`` or ``return_cov=True``. This converter
    needs to be called with theses options to enable
    the second results.
    See example :ref:`l-gpr-example` to see how to
    use this converter which does not behave exactly
    as the others.
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    if opv is None:
        raise RuntimeError("container.target_opset must not be None")
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32

    options = container.get_options(
        op, dict(return_cov=False, return_std=False, optim=None)
    )
    if hasattr(op, "kernel_") and op.kernel_ is not None:
        kernel = op.kernel_
    elif op.kernel is None:
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
    else:
        kernel = op.kernel

    if not hasattr(op, "X_train_") or op.X_train_ is None:
        out0 = _zero_vector_of_size(
            X, keepdims=1, output_names=out[:1], dtype=dtype, op_version=opv
        )

        outputs = [out0]
        if options["return_cov"]:
            outputs.append(
                convert_kernel(
                    kernel, X, output_names=out[1:], dtype=dtype, op_version=opv
                )
            )
        if options["return_std"]:
            outputs.append(
                OnnxSqrt(
                    convert_kernel_diag(kernel, X, dtype=dtype, op_version=opv),
                    output_names=out[1:],
                    op_version=opv,
                )
            )
    else:
        # Code scikit-learn
        # K_trans = self.kernel_(X, self.X_train_)
        # y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        # y_mean = self._y_train_mean + y_mean * self._y_train_std

        k_trans = convert_kernel(
            kernel,
            X,
            x_train=op.X_train_.astype(dtype),
            dtype=dtype,
            optim=options.get("optim", None),
            op_version=opv,
        )
        k_trans.set_onnx_name_prefix("kgpd")
        y_mean_b = OnnxMatMul(k_trans, op.alpha_.astype(dtype), op_version=opv)

        mean_y = op._y_train_mean.astype(dtype)
        if len(mean_y.shape) == 1:
            mean_y = mean_y.reshape(mean_y.shape + (1,))

        if not hasattr(op, "_y_train_std") or op._y_train_std == 1:
            if isinstance(y_mean_b, (np.float32, np.float64)):
                y_mean_b = np.array([y_mean_b])
            if isinstance(mean_y, (np.float32, np.float64)):
                mean_y = np.array([mean_y])
            y_mean = OnnxAdd(y_mean_b, mean_y, op_version=opv)
        else:
            # A bug was fixed in 0.23 and it changed
            # the predictions when return_std is True.
            # See https://github.com/scikit-learn/scikit-learn/pull/15782.
            # y_mean = self._y_train_std * y_mean + self._y_train_mean
            var_y = op._y_train_std.astype(dtype)
            if len(var_y.shape) == 1:
                var_y = var_y.reshape(var_y.shape + (1,))
            if isinstance(var_y, (np.float32, np.float64)):
                var_y = np.array([var_y])
            if isinstance(mean_y, (np.float32, np.float64)):
                mean_y = np.array([mean_y])
            y_mean = OnnxAdd(
                OnnxMul(y_mean_b, var_y, op_version=opv), mean_y, op_version=opv
            )

        y_mean.set_onnx_name_prefix("gpr")
        y_mean_reshaped = OnnxReshapeApi13(
            y_mean,
            np.array([-1, 1], dtype=np.int64),
            op_version=opv,
            output_names=out[:1],
        )
        outputs = [y_mean_reshaped]

        if options["return_cov"]:
            raise NotImplementedError()
        if options["return_std"]:
            if hasattr(op, "_K_inv") and op._K_inv is not None:
                # scikit-learn < 0.24.2
                _K_inv = op._K_inv
            else:
                # scikit-learn >= 0.24.2
                L_inv = solve_triangular(op.L_.T, np.eye(op.L_.shape[0]))
                _K_inv = L_inv.dot(L_inv.T)

            # y_var = self.kernel_.diag(X)
            y_var = convert_kernel_diag(
                kernel, X, dtype=dtype, optim=options.get("optim", None), op_version=opv
            )

            # y_var -= np.einsum("ij,ij->i",
            #       np.dot(K_trans, self._K_inv), K_trans)
            k_dot = OnnxMatMul(k_trans, _K_inv.astype(dtype), op_version=opv)
            ys_var = OnnxSub(
                y_var,
                OnnxReduceSumApi11(
                    OnnxMul(k_dot, k_trans, op_version=opv),
                    axes=[1],
                    keepdims=0,
                    op_version=opv,
                ),
                op_version=opv,
            )

            # y_var_negative = y_var < 0
            # if np.any(y_var_negative):
            #     y_var[y_var_negative] = 0.0
            ys0_var = OnnxMax(ys_var, np.array([0], dtype=dtype), op_version=opv)

            if hasattr(op, "_y_train_std") and op._y_train_std != 1:
                # y_var = y_var * self._y_train_std**2
                ys0_var = OnnxMul(ys0_var, var_y**2, op_version=opv)

            # var = np.sqrt(ys0_var)
            var = OnnxSqrt(ys0_var, output_names=out[1:], op_version=opv)
            var.set_onnx_name_prefix("gprv")
            outputs.append(var)

    for o in outputs:
        o.add_to(scope, container)


def convert_gaussian_process_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    The method *predict* from class *GaussianProcessClassifier*
    may cache some results if it is called with parameter
    ``return_std=True`` or ``return_cov=True``. This converter
    needs to be called with theses options to enable
    the second results.
    See example :ref:`l-gpr-example` to see how to
    use this converter which does not behave exactly
    as the others.
    """
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    op_est = operator.raw_operator.base_estimator_
    opv = container.target_opset
    if opv is None:
        raise RuntimeError("container.target_opset must not be None")
    if OnnxEinsum is None or OnnxErf is None:
        raise RuntimeError(
            "target opset must be >= 12 for operator 'einsum' and 'erf'."
        )
    if LAMBDAS is None:
        raise RuntimeError("Only scikit-learn>=0.22 is supported.")
    outputs = []

    options = container.get_options(op, dict(optim=None))
    if hasattr(op, "kernel_") and op.kernel_ is not None:
        kernel = op.kernel_
    elif op.kernel is None:
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
    else:
        kernel = op.kernel

    if not hasattr(op_est, "X_train_"):
        raise NotImplementedError("Only binary classification is iplemented.")
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    K_starT = convert_kernel(
        kernel,
        X,
        x_train=op_est.X_train_.astype(dtype),
        dtype=dtype,
        optim=options.get("optim", None),
        op_version=opv,
    )
    K_star = OnnxTranspose(K_starT, op_version=opv)
    K_star.set_onnx_name_prefix("kstar")

    # common
    # f_star = K_star.T.dot(self.y_train_ - self.pi_)
    f_star_right = (op_est.y_train_ - op_est.pi_).astype(dtype).reshape((-1, 1))
    f_star = OnnxMatMul(K_starT, f_star_right, op_version=opv)
    f_star.set_onnx_name_prefix("f_star")

    best = OnnxCast(
        OnnxNot(
            OnnxLess(f_star, np.array([0], dtype=dtype), op_version=opv), op_version=opv
        ),
        to=onnx_proto.TensorProto.INT64,
        op_version=opv,
    )
    classes = OnnxArrayFeatureExtractor(op.classes_.astype(np.int64), best)
    labels = OnnxTranspose(classes, op_version=opv, output_names=out[:1])
    labels.set_onnx_name_prefix("labels")
    outputs.append(labels)

    # predict_proba
    # a x = b, x = a^-1 b
    # v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
    v = OnnxSolve(
        op_est.L_.astype(dtype),
        OnnxMul(op_est.W_sr_[:, np.newaxis].astype(dtype), K_star, op_version=opv),
        op_version=opv,
    )
    v.set_onnx_name_prefix("solve")

    # var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
    var_f_star_kernel = convert_kernel_diag(
        kernel, X, dtype=dtype, optim=options.get("optim", None), op_version=opv
    )
    var_f_star_kernel.set_onnx_name_prefix("diag")
    var_f_star = OnnxSub(
        var_f_star_kernel,
        OnnxEinsum(v, v, equation="ij,ij->j", op_version=opv),
        op_version=opv,
    )
    var_f_star.set_onnx_name_prefix("var_f_star")

    # alpha = 1 / (2 * var_f_star)
    alpha = OnnxReciprocal(
        OnnxMul(var_f_star, np.array([2], dtype=dtype), op_version=opv), op_version=opv
    )
    alpha.set_onnx_name_prefix("alpha")

    # gamma = LAMBDAS * f_star
    gamma = OnnxMul(
        LAMBDAS.astype(dtype),
        OnnxReshapeApi13(f_star, np.array([1, -1], dtype=np.int64), op_version=opv),
        op_version=opv,
    )
    gamma.set_onnx_name_prefix("gamma")

    # integrals = np.sqrt(np.pi / alpha) *
    #               erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2))) /
    #               (2 * np.sqrt(var_f_star * 2 * np.pi))
    integrals_1 = OnnxSqrt(
        OnnxDiv(np.array([np.pi], dtype=dtype), alpha, op_version=opv), op_version=opv
    )
    integrals_1.set_onnx_name_prefix("int1")

    integrals_2_1 = OnnxAdd(
        alpha,
        OnnxPow(LAMBDAS.astype(dtype), np.array([2], dtype=dtype), op_version=opv),
        op_version=opv,
    )
    integrals_2_1.set_onnx_name_prefix("int21")

    integrals_2_2 = OnnxSqrt(
        OnnxDiv(alpha, integrals_2_1, op_version=opv), op_version=opv
    )
    integrals_2_2.set_onnx_name_prefix("int22")

    integrals_div = OnnxMul(
        np.array([2], dtype=dtype),
        OnnxSqrt(
            OnnxMul(
                OnnxMul(var_f_star, np.array([2], dtype=dtype), op_version=opv),
                np.array([np.pi], dtype=dtype),
                op_version=opv,
            ),
            op_version=opv,
        ),
        op_version=opv,
    )
    integrals_div.set_onnx_name_prefix("intdiv")

    integrals = OnnxMul(
        integrals_1,
        OnnxDiv(
            OnnxErf(OnnxMul(gamma, integrals_2_2, op_version=opv), op_version=opv),
            integrals_div,
            op_version=opv,
        ),
        op_version=opv,
    )
    integrals.set_onnx_name_prefix("integrals")

    # pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()
    coef_sum = (0.5 * COEFS.sum()).astype(dtype)
    if not isinstance(coef_sum, np.ndarray):
        coef_sum = np.array([coef_sum])
    pi_star = OnnxAdd(
        OnnxReduceSumApi11(
            OnnxMul(COEFS.astype(dtype), integrals, op_version=opv),
            op_version=opv,
            axes=[0],
        ),
        coef_sum,
        op_version=opv,
    )
    pi_star.set_onnx_name_prefix("pi_star")

    pi_star = OnnxReshapeApi13(
        pi_star, np.array([-1, 1], dtype=np.int64), op_version=opv
    )
    pi_star.set_onnx_name_prefix("pi_star2")
    final = OnnxConcat(
        OnnxAdd(
            OnnxNeg(pi_star, op_version=opv), np.array([1], dtype=dtype), op_version=opv
        ),
        pi_star,
        op_version=opv,
        axis=1,
        output_names=out[1:2],
    )
    outputs.append(final)

    for o in outputs:
        o.add_to(scope, container)


if OnnxConstantOfShape is not None:
    register_converter(
        "SklearnGaussianProcessRegressor",
        convert_gaussian_process_regressor,
        options={
            "return_cov": [False, True],
            "return_std": [False, True],
            "optim": [None, "cdist"],
        },
    )

if OnnxEinsum is not None and OnnxErf is not None:
    register_converter(
        "SklearnGaussianProcessClassifier",
        convert_gaussian_process_classifier,
        options={
            "optim": [None, "cdist"],
            "nocl": [False, True],
            "output_class_labels": [False, True],
            "zipmap": [False, True],
        },
    )
