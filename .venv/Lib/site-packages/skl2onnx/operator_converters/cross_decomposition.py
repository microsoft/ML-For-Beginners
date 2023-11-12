# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import Int64TensorType, guess_numpy_type, guess_proto_type
from ..algebra.onnx_ops import OnnxAdd, OnnxCast, OnnxDiv, OnnxMatMul, OnnxSub


def convert_pls_regression(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    X = operator.inputs[0]
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(X.type)
    if dtype != np.float64:
        dtype = np.float32
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if isinstance(X.type, Int64TensorType):
        X = OnnxCast(X, to=proto_dtype, op_version=opv)

    coefs = op.x_mean_ if hasattr(op, "x_mean_") else op._x_mean
    std = op.x_std_ if hasattr(op, "x_std_") else op._x_std
    ym = op.y_mean_ if hasattr(op, "x_mean_") else op._y_mean

    norm_x = OnnxDiv(
        OnnxSub(X, coefs.astype(dtype), op_version=opv),
        std.astype(dtype),
        op_version=opv,
    )
    if hasattr(op, "set_predict_request"):
        # new in 1.3
        coefs = op.coef_.T.astype(dtype)
    else:
        coefs = op.coef_.astype(dtype)
    dot = OnnxMatMul(norm_x, coefs, op_version=opv)
    pred = OnnxAdd(dot, ym.astype(dtype), op_version=opv, output_names=operator.outputs)
    pred.add_to(scope, container)


register_converter("SklearnPLSRegression", convert_pls_regression)
