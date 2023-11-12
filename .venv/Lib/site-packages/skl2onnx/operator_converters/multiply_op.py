# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_mul
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def convert_sklearn_multiply(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    operand_name = scope.get_unique_variable_name("operand")

    container.add_initializer(
        operand_name, onnx_proto.TensorProto.FLOAT, [], [operator.operand]
    )

    apply_mul(
        scope,
        [operator.inputs[0].full_name, operand_name],
        operator.outputs[0].full_name,
        container,
    )


register_converter("SklearnMultiply", convert_sklearn_multiply)
