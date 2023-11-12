# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_proto_type


def convert_sklearn_replace_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    input_name = operator.inputs[0].full_name
    output_name = operator.outputs[0].full_name

    proto_dtype = guess_proto_type(operator.inputs[0].type)

    cst_nan_name = scope.get_unique_variable_name("nan_name")
    container.add_initializer(cst_nan_name, proto_dtype, [1], [op.to_value])
    cst_zero_name = scope.get_unique_variable_name("zero_name")
    container.add_initializer(cst_zero_name, proto_dtype, [1], [op.from_value])

    mask_name = scope.get_unique_variable_name("mask_name")
    container.add_node(
        "Equal",
        [input_name, cst_zero_name],
        mask_name,
        name=scope.get_unique_operator_name("Equal"),
    )

    container.add_node(
        "Where",
        [mask_name, cst_nan_name, input_name],
        output_name,
        name=scope.get_unique_operator_name("Where"),
    )


register_converter("SklearnReplaceTransformer", convert_sklearn_replace_transformer)
