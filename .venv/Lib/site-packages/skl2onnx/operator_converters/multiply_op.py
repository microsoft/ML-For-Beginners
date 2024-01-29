# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_mul
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_proto_type


def convert_sklearn_multiply(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    for input, output in zip(operator.inputs, operator.outputs):
        operand_name = scope.get_unique_variable_name("operand")

        container.add_initializer(
            operand_name, guess_proto_type(input.type), [], [operator.operand]
        )

        apply_mul(
            scope,
            [input.full_name, operand_name],
            output.full_name,
            container,
        )


register_converter("SklearnMultiply", convert_sklearn_multiply)
