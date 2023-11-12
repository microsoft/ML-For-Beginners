# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from ..common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_gaussian_mixture_output_shapes(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[2, 3]
    )
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Input must be a [N, C]-tensor")

    op = operator.raw_operator
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([N, 1])
    operator.outputs[1].type.shape = [N, op.n_components]
    if len(operator.outputs) > 2:
        operator.outputs[2].type.shape = [N, 1]


register_shape_calculator(
    "SklearnGaussianMixture", calculate_gaussian_mixture_output_shapes
)
register_shape_calculator(
    "SklearnBayesianGaussianMixture", calculate_gaussian_mixture_output_shapes
)
