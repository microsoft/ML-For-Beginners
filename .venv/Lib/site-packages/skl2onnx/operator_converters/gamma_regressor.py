# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common.data_types import Int64TensorType, guess_numpy_type
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxCast,
    OnnxExp,
    OnnxIdentity,
    OnnxMatMul,
    OnnxReshape,
    OnnxSigmoid,
)


def convert_sklearn_gamma_regressor(
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


register_converter("SklearnGammaRegressor", convert_sklearn_gamma_regressor)
