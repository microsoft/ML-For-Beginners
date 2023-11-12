# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..proto import onnx_proto
from ..common.data_types import guess_numpy_type, guess_proto_type
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import (
    apply_add,
    apply_log,
    apply_mul,
    apply_identity,
    apply_normalizer,
)


def convert_sklearn_tfidf_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # TODO: use sparse containers when available
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    float_type = dtype
    # onnx_proto.TensorProto.FLOAT
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    data = operator.input_full_names
    output_name = scope.get_unique_variable_name("tfidftr_output")

    if op.sublinear_tf:
        # code scikit-learn
        # np.log(X.data, X.data) --> does not apply on null coefficient
        # X.data += 1
        # ONNX does not support sparse tensors before opset < 11
        # approximated by X.data += 1 --> np.log(X.data, X.data)
        if operator.target_opset < 11:
            plus1 = scope.get_unique_variable_name("plus1")
            C = operator.inputs[0].type.shape[1]
            ones = scope.get_unique_variable_name("ones")
            cst = np.ones((C,), dtype=float_type)
            container.add_initializer(ones, proto_dtype, [C], cst.flatten())
            apply_add(scope, data + [ones], plus1, container, broadcast=1)
            plus1logged = scope.get_unique_variable_name("plus1logged")
            apply_log(scope, plus1, plus1logged, container)
            data = [plus1logged]
        else:
            # sparse containers have not yet been implemented.
            raise RuntimeError(
                "ONNX does not support sparse tensors before opset < 11, "
                "sublinear_tf must be False."
            )

    if op.use_idf:
        cst = op.idf_.astype(float_type)
        if len(cst.shape) > 1:
            cst = np.diag(cst)
        cst = cst.ravel().flatten()
        shape = [len(cst)]
        idfcst = scope.get_unique_variable_name("idfcst")
        container.add_initializer(idfcst, proto_dtype, shape, cst)
        apply_mul(scope, data + [idfcst], output_name, container, broadcast=1)
    else:
        output_name = data[0]

    if op.norm is not None:
        norm_name = scope.get_unique_variable_name("tfidftr_norm")
        apply_normalizer(
            scope,
            output_name,
            norm_name,
            container,
            norm=op.norm.upper(),
            use_float=float_type == np.float32,
        )
        output_name = norm_name

    options = container.get_options(op, dict(nan=False))
    replace_by_nan = options.get("nan", False)
    if replace_by_nan:
        # This part replaces all null values by nan.
        cst_nan_name = scope.get_unique_variable_name("nan_name")
        container.add_initializer(cst_nan_name, proto_dtype, [1], [np.nan])
        cst_zero_name = scope.get_unique_variable_name("zero_name")
        container.add_initializer(cst_zero_name, proto_dtype, [1], [0])

        mask_name = scope.get_unique_variable_name("mask_name")
        container.add_node(
            "Equal",
            [output_name, cst_zero_name],
            mask_name,
            name=scope.get_unique_operator_name("Equal"),
        )

        where_name = scope.get_unique_variable_name("where_name")
        container.add_node(
            "Where",
            [mask_name, cst_nan_name, output_name],
            where_name,
            name=scope.get_unique_operator_name("Where"),
        )
        output_name = where_name

    apply_identity(scope, output_name, operator.output_full_names, container)


register_converter(
    "SklearnTfidfTransformer",
    convert_sklearn_tfidf_transformer,
    options={"nan": [True, False]},
)
