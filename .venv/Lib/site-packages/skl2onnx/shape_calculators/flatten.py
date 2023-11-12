# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatType, Int64Type, StringType, TensorType
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_flatten(operator):
    check_input_and_output_numbers(operator, output_count_range=1, input_count_range=1)
    i = operator.inputs[0]
    N = i.get_first_dimension()
    if isinstance(i.type, TensorType):
        if i.type.shape[1] is None:
            C = None
        else:
            C = i.type.shape[1]
    elif isinstance(i.type, (Int64Type, FloatType, StringType)):
        C = 1
    else:
        C = None
    if C is None:
        operator.outputs[0].type.shape = [N, C]
    else:
        operator.outputs[0].type.shape = [N * C]


register_shape_calculator("SklearnFlatten", calculate_sklearn_flatten)
