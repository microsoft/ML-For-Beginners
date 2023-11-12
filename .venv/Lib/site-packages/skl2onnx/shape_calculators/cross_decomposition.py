# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from ..common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_pls_regression_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Input must be a [N, C]-tensor")

    op = operator.raw_operator
    cls_type = operator.inputs[0].type.__class__
    if cls_type != DoubleTensorType:
        cls_type = FloatTensorType
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = cls_type([N, op.coef_.shape[1]])


register_shape_calculator(
    "SklearnPLSRegression", calculate_pls_regression_output_shapes
)
