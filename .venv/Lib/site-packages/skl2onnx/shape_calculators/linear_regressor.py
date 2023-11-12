# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers, check_input_and_output_types
from ..common.shape_calculator import calculate_linear_regressor_output_shapes
from ..common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)


def calculate_bayesian_ridge_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a
    batch. If the input batch size is N, the output shape may be
    [N, 1].
    """
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator,
        good_input_types=[
            BooleanTensorType,
            DoubleTensorType,
            FloatTensorType,
            Int64TensorType,
        ],
    )

    inp0 = operator.inputs[0].type
    if isinstance(inp0, (FloatTensorType, DoubleTensorType)):
        cls_type = inp0.__class__
    else:
        cls_type = FloatTensorType

    N = operator.inputs[0].get_first_dimension()
    if (
        hasattr(operator.raw_operator, "coef_")
        and len(operator.raw_operator.coef_.shape) > 1
    ):
        operator.outputs[0].type = cls_type([N, operator.raw_operator.coef_.shape[1]])
    else:
        operator.outputs[0].type = cls_type([N, 1])

    if len(operator.inputs) == 2:
        # option return_std is True
        operator.outputs[1].type = cls_type([N, 1])


register_shape_calculator(
    "SklearnAdaBoostRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnBaggingRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnBayesianRidge", calculate_bayesian_ridge_output_shapes
)
register_shape_calculator(
    "SklearnLinearRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator("SklearnLinearSVR", calculate_linear_regressor_output_shapes)
register_shape_calculator(
    "SklearnMLPRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnPoissonRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnRANSACRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnStackingRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnTweedieRegressor", calculate_linear_regressor_output_shapes
)
register_shape_calculator(
    "SklearnGammaRegressor", calculate_linear_regressor_output_shapes
)
