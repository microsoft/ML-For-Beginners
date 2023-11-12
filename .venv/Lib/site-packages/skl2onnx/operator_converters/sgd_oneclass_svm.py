# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_cast, apply_sub
from ..common.data_types import BooleanTensorType, Int64TensorType, guess_proto_type
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def convert_sklearn_sgd_oneclass_svm(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    input_name = operator.inputs[0].full_name
    output_names = operator.output_full_names
    model = operator.raw_operator
    coef = model.coef_.T
    offset = model.offset_

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if isinstance(operator.inputs[0].type, (BooleanTensorType, Int64TensorType)):
        cast_input_name = scope.get_unique_variable_name("cast_input")
        apply_cast(
            scope, operator.input_full_names, cast_input_name, container, to=proto_dtype
        )
        input_name = cast_input_name

    coef_name = scope.get_unique_variable_name("coef")
    container.add_initializer(coef_name, proto_dtype, coef.shape, coef.ravel())

    offset_name = scope.get_unique_variable_name("offset")
    container.add_initializer(offset_name, proto_dtype, offset.shape, offset)

    matmul_result_name = scope.get_unique_variable_name("matmul_result")
    container.add_node(
        "MatMul",
        [input_name, coef_name],
        matmul_result_name,
        name=scope.get_unique_operator_name("MatMul"),
    )

    apply_sub(
        scope,
        [matmul_result_name, offset_name],
        output_names[1],
        container,
        broadcast=0,
    )

    pred = scope.get_unique_variable_name("class_prediction")
    container.add_node("Sign", output_names[1], pred, op_version=9)
    apply_cast(scope, pred, output_names[0], container, to=onnx_proto.TensorProto.INT64)


register_converter("SklearnSGDOneClassSVM", convert_sklearn_sgd_oneclass_svm)
