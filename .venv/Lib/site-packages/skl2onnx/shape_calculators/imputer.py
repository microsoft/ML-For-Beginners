# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import (
    FloatTensorType,
    Int64TensorType,
    DoubleTensorType,
    StringTensorType,
)
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_imputer_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C_1], ..., [N, C_n] ---> [N, C_1 + ... + C_n]

    It's possible to receive multiple inputs so we need to concatenate
    them along C-axis. The produced tensor's shape is used as the
    output shape.
    """
    check_input_and_output_numbers(
        operator, input_count_range=[1, None], output_count_range=1
    )
    check_input_and_output_types(
        operator,
        good_input_types=[
            FloatTensorType,
            Int64TensorType,
            DoubleTensorType,
            StringTensorType,
        ],
    )
    output = operator.outputs[0]
    for variable in operator.inputs:
        if not isinstance(variable.type, type(output.type)):  # noqa
            raise RuntimeError(
                "Inputs and outputs should have the same type "
                "%r != %r." % (type(variable.type), type(output.type))
            )

    N = operator.inputs[0].get_first_dimension()
    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] is not None:
            C += variable.type.shape[1]
        else:
            C = None
            break

    output.type.shape = [N, C]


register_shape_calculator("SklearnImputer", calculate_sklearn_imputer_output_shapes)
register_shape_calculator(
    "SklearnSimpleImputer", calculate_sklearn_imputer_output_shapes
)
register_shape_calculator("SklearnBinarizer", calculate_sklearn_imputer_output_shapes)
