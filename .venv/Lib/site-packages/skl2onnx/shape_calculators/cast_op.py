# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers
from ..common.data_types import _guess_numpy_type
from ..common.shape_calculator import calculate_linear_regressor_output_shapes


def calculate_sklearn_cast(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)


def calculate_sklearn_cast_transformer(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    op = operator.raw_operator
    otype = _guess_numpy_type(op.dtype, operator.inputs[0].type.shape)
    operator.outputs[0].type = otype


register_shape_calculator("SklearnCast", calculate_sklearn_cast)
register_shape_calculator("SklearnCastTransformer", calculate_sklearn_cast_transformer)
register_shape_calculator(
    "SklearnCastRegressor", calculate_linear_regressor_output_shapes
)
