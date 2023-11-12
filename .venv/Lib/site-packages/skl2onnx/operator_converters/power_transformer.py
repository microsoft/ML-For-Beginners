# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx import TensorProto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxSub,
    OnnxPow,
    OnnxDiv,
    OnnxMul,
    OnnxCast,
    OnnxNot,
    OnnxLess,
    OnnxLog,
    OnnxNeg,
    OnnxImputer,
    OnnxIdentity,
    OnnxScaler,
)


def convert_powertransformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """Converter for PowerTransformer"""
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    lambdas = op.lambdas_.astype(dtype)

    # tensors of units and zeros
    ones_ = OnnxDiv(op_in, op_in, op_version=opv)
    zeros_ = OnnxSub(op_in, op_in, op_version=opv)

    # logical masks for input
    less_than_zero = OnnxLess(op_in, zeros_, op_version=opv)
    less_mask = OnnxCast(
        less_than_zero, to=getattr(TensorProto, "FLOAT"), op_version=opv
    )

    greater_than_zero = OnnxNot(less_than_zero, op_version=opv)
    greater_mask = OnnxCast(
        greater_than_zero, to=getattr(TensorProto, "FLOAT"), op_version=opv
    )

    # logical masks for lambdas
    lambda_zero_mask = np.float32(lambdas == 0)
    lambda_nonzero_mask = np.float32(lambdas != 0)
    lambda_two_mask = np.float32(lambdas == 2)
    lambda_nontwo_mask = np.float32(lambdas != 2)

    if "yeo-johnson" in op.method:
        y0 = OnnxAdd(op_in, ones_, op_version=opv)  # For positive input
        y1 = OnnxSub(ones_, op_in, op_version=opv)  # For negative input

        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(y0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_, op_version=opv)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxImputer(
            y_gr0_l_ne0,
            imputed_value_floats=[0.0],
            replaced_value_float=np.inf,
            op_version=opv,
        )
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask, op_version=opv)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(y0, op_version=opv)
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask, op_version=opv)

        # positive input, an arbitrary lambda
        y_gr0 = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0, op_version=opv)
        y_gr0 = OnnxImputer(
            y_gr0,
            imputed_value_floats=[0.0],
            replaced_value_float=np.NAN,
            op_version=opv,
        )
        y_gr0 = OnnxMul(y_gr0, greater_mask, op_version=opv)

        # negative input, lambda != 2
        y_le0_l_ne2 = OnnxPow(y1, 2 - lambdas, op_version=opv)
        y_le0_l_ne2 = OnnxSub(ones_, y_le0_l_ne2, op_version=opv)
        y_le0_l_ne2 = OnnxDiv(y_le0_l_ne2, (2 - lambdas).astype(dtype), op_version=opv)
        y_le0_l_ne2 = OnnxImputer(
            y_le0_l_ne2,
            imputed_value_floats=[0.0],
            replaced_value_float=np.inf,
            op_version=opv,
        )
        y_le0_l_ne2 = OnnxMul(y_le0_l_ne2, lambda_nontwo_mask, op_version=opv)

        # negative input, lambda == 2
        y_le0_l_eq2 = OnnxNeg(OnnxLog(y1, op_version=opv), op_version=opv)
        y_le0_l_eq2 = OnnxMul(y_le0_l_eq2, lambda_two_mask, op_version=opv)

        # negative input, an arbitrary lambda
        y_le0 = OnnxAdd(y_le0_l_ne2, y_le0_l_eq2, op_version=opv)
        y_le0 = OnnxImputer(
            y_le0,
            imputed_value_floats=[0.0],
            replaced_value_float=np.NAN,
            op_version=opv,
        )
        y_le0 = OnnxMul(y_le0, less_mask, op_version=opv)

        # Arbitrary input and lambda
        y = OnnxAdd(y_gr0, y_le0, op_version=opv)

    elif "box-cox" in op.method:
        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(op_in, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_, op_version=opv)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxImputer(
            y_gr0_l_ne0,
            imputed_value_floats=[0.0],
            replaced_value_float=np.inf,
            op_version=opv,
        )
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask, op_version=opv)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(op_in, op_version=opv)
        y_gr0_l_eq0 = OnnxImputer(
            y_gr0_l_eq0,
            imputed_value_floats=[0.0],
            replaced_value_float=np.NAN,
            op_version=opv,
        )
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask, op_version=opv)

        # positive input, arbitrary lambda
        y = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0, op_version=opv)

        # negative input
        # PowerTransformer(method='box-cox').fit(negative_data)
        # raises ValueError.
        # Therefore we cannot use convert_sklearn() for that model
    else:
        raise NotImplementedError("Method {} is not supported".format(op.method))

    y.set_onnx_name_prefix("pref")

    if op.standardize:
        use_scaler_op = container.is_allowed({"Scaler"})
        if not use_scaler_op or dtype != np.float32:
            sub = OnnxSub(y, op._scaler.mean_.astype(dtype), op_version=opv)
            final = OnnxDiv(
                sub,
                op._scaler.scale_.astype(dtype),
                op_version=opv,
                output_names=[op_out],
            )
        else:
            final = OnnxScaler(
                y,
                offset=op._scaler.mean_.astype(dtype),
                scale=(1.0 / op._scaler.scale_).astype(dtype),
                op_version=opv,
                output_names=[op_out],
            )
    else:
        final = OnnxIdentity(y, op_version=opv, output_names=[op_out])

    final.add_to(scope, container)


register_converter("SklearnPowerTransformer", convert_powertransformer)
