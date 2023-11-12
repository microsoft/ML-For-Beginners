# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._apply_operation import (
    apply_cast,
    apply_add,
    apply_sqrt,
    apply_div,
    apply_sub,
    apply_reshape,
)
from ..common.data_types import (
    BooleanTensorType,
    Int64TensorType,
    DoubleTensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxCast,
    OnnxExp,
    OnnxIdentity,
    OnnxMatMul,
    OnnxReshape,
    OnnxSigmoid,
)


def convert_sklearn_linear_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    use_linear_op = container.is_allowed({"LinearRegressor"})

    if not use_linear_op or type(operator.inputs[0].type) in (DoubleTensorType,):
        proto_dtype = guess_proto_type(operator.inputs[0].type)
        coef = scope.get_unique_variable_name("coef")
        if len(op.coef_.shape) == 1:
            model_coef = op.coef_.reshape((-1, 1))
        else:
            model_coef = op.coef_.T
        container.add_initializer(
            coef, proto_dtype, model_coef.shape, model_coef.ravel().tolist()
        )
        intercept = scope.get_unique_variable_name("intercept")
        value_intercept = op.intercept_.reshape((-1,))
        container.add_initializer(
            intercept,
            proto_dtype,
            value_intercept.shape,
            value_intercept.ravel().tolist(),
        )
        multiplied = scope.get_unique_variable_name("multiplied")
        container.add_node(
            "MatMul",
            [operator.inputs[0].full_name, coef],
            multiplied,
            name=scope.get_unique_operator_name("MatMul"),
        )
        resh = scope.get_unique_variable_name("resh")
        apply_add(scope, [multiplied, intercept], resh, container)
        last_dim = 1 if len(model_coef.shape) == 1 else model_coef.shape[-1]
        apply_reshape(
            scope,
            resh,
            operator.outputs[0].full_name,
            container,
            desired_shape=(-1, last_dim),
        )
        return

    op_type = "LinearRegressor"
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype not in (np.float32, np.float64):
        dtype = np.float32
    attrs = {"name": scope.get_unique_operator_name(op_type)}
    attrs["coefficients"] = op.coef_.astype(dtype).ravel()
    attrs["intercepts"] = np.array([op.intercept_], dtype=dtype).ravel()
    if len(op.coef_.shape) == 2:
        attrs["targets"] = op.coef_.shape[0]

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope,
            operator.input_full_names,
            cast_input_name,
            container,
            to=(
                onnx_proto.TensorProto.DOUBLE
                if dtype == np.float64
                else onnx_proto.TensorProto.FLOAT
            ),
        )
        input_name = cast_input_name
    container.add_node(
        op_type,
        input_name,
        operator.outputs[0].full_name,
        op_domain="ai.onnx.ml",
        **attrs,
    )


def convert_sklearn_bayesian_ridge(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    convert_sklearn_linear_regressor(scope, operator, container)

    op = operator.raw_operator
    options = container.get_options(op, dict(return_std=False))
    return_std = options["return_std"]
    if not return_std:
        return

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if hasattr(op, "normalize") and op.normalize:
        # if self.normalize:
        #     X = (X - self.X_offset_) / self.X_scale_
        offset = scope.get_unique_variable_name("offset")
        container.add_initializer(
            offset, proto_dtype, op.X_offset_.shape, op.X_offset_.ravel().tolist()
        )
        scale = scope.get_unique_variable_name("scale")
        container.add_initializer(
            scale, proto_dtype, op.X_scale_.shape, op.X_scale_.ravel().tolist()
        )
        centered = scope.get_unique_variable_name("centered")
        apply_sub(scope, [operator.inputs[0].full_name, offset], centered, container)
        scaled = scope.get_unique_variable_name("scaled")
        apply_div(scope, [centered, scale], scaled, container)
        input_name = scaled
    else:
        input_name = operator.inputs[0].full_name

    # sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
    sigma = scope.get_unique_variable_name("sigma")
    container.add_initializer(
        sigma, proto_dtype, op.sigma_.shape, op.sigma_.ravel().tolist()
    )
    sigmaed0 = scope.get_unique_variable_name("sigma0")
    container.add_node(
        "MatMul",
        [input_name, sigma],
        sigmaed0,
        name=scope.get_unique_operator_name("MatMul"),
    )
    sigmaed = scope.get_unique_variable_name("sigma")
    if container.target_opset < 13:
        container.add_node(
            "ReduceSum",
            sigmaed0,
            sigmaed,
            axes=[1],
            name=scope.get_unique_operator_name("ReduceSum"),
        )
    else:
        axis_name = scope.get_unique_variable_name("axis")
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [1])
        container.add_node(
            "ReduceSum",
            [sigmaed0, axis_name],
            sigmaed,
            name=scope.get_unique_operator_name("ReduceSum"),
        )

    # y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
    # return y_mean, y_std
    std0 = scope.get_unique_variable_name("std0")
    alphainv = scope.get_unique_variable_name("alphainv")
    container.add_initializer(alphainv, proto_dtype, [1], [1 / op.alpha_])
    apply_add(scope, [sigmaed, alphainv], std0, container)
    apply_sqrt(scope, std0, operator.outputs[1].full_name, container)


def convert_sklearn_poisson_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32

    if isinstance(X.type, Int64TensorType):
        input_var = OnnxCast(X, to=np.float32, op_version=opv)
    else:
        input_var = X

    intercept = (
        op.intercept_.astype(dtype)
        if len(op.intercept_.shape) > 0
        else np.array([op.intercept_], dtype=dtype)
    )
    eta = OnnxAdd(
        OnnxMatMul(input_var, op.coef_.astype(dtype), op_version=opv),
        intercept,
        op_version=opv,
    )

    if hasattr(op, "_link_instance"):
        # scikit-learn < 1.1
        from sklearn.linear_model._glm.link import IdentityLink, LogLink, LogitLink

        if isinstance(op._link_instance, IdentityLink):
            Y = OnnxIdentity(eta, op_version=opv)
        elif isinstance(op._link_instance, LogLink):
            Y = OnnxExp(eta, op_version=opv)
        elif isinstance(op._link_instance, LogitLink):
            Y = OnnxSigmoid(eta, op_version=opv)
        else:
            raise RuntimeError(
                "Unexpected type %r for _link_instance "
                "in operator type %r." % (type(op._link_instance), type(op))
            )
    else:
        # scikit-learn >= 1.1
        from sklearn._loss.loss import (
            AbsoluteError,
            HalfBinomialLoss,
            HalfGammaLoss,
            HalfPoissonLoss,
            HalfSquaredError,
            HalfTweedieLoss,
            HalfTweedieLossIdentity,
            PinballLoss,
        )

        loss = op._get_loss()
        if isinstance(
            loss,
            (AbsoluteError, HalfSquaredError, HalfTweedieLossIdentity, PinballLoss),
        ):
            Y = OnnxIdentity(eta, op_version=opv)
        elif isinstance(loss, (HalfPoissonLoss, HalfGammaLoss, HalfTweedieLoss)):
            Y = OnnxExp(eta, op_version=opv)
        elif isinstance(loss, HalfBinomialLoss):
            Y = OnnxSigmoid(eta, op_version=opv)
        else:
            raise RuntimeError(
                f"Unexpected type of link for {loss!r} loss " "in operator type {op!r}."
            )

    last_dim = 1 if len(op.coef_.shape) == 1 else op.coef_.shape[-1]
    final = OnnxReshape(
        Y,
        np.array([-1, last_dim], dtype=np.int64),
        op_version=opv,
        output_names=out[:1],
    )
    final.add_to(scope, container)


register_converter("SklearnLinearRegressor", convert_sklearn_linear_regressor)
register_converter("SklearnLinearSVR", convert_sklearn_linear_regressor)
register_converter(
    "SklearnBayesianRidge",
    convert_sklearn_bayesian_ridge,
    options={"return_std": [True, False]},
)
register_converter("SklearnPoissonRegressor", convert_sklearn_poisson_regressor)
register_converter("SklearnTweedieRegressor", convert_sklearn_poisson_regressor)
