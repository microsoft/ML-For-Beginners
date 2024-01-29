# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers
from ..common.shape_calculator import _infer_linear_regressor_output_types


def voting_regressor_shape_calculator(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    return _infer_linear_regressor_output_types(operator)


register_shape_calculator("SklearnVotingRegressor", voting_regressor_shape_calculator)
