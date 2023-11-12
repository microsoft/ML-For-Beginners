# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, DoubleTensorType
from ..common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_kernel_pca_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType],
    )
    N = operator.inputs[0].get_first_dimension()
    op = operator.raw_operator
    lbd = op.eigenvalues_ if hasattr(op, "eigenvalues_") else op.lambdas_
    C = lbd.shape[0]
    operator.outputs[0].type.shape = [N, C]


def calculate_sklearn_kernel_centerer_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType],
    )
    N = operator.inputs[0].get_first_dimension()
    C = operator.raw_operator.K_fit_rows_.shape[0]
    operator.outputs[0].type.shape = [N, C]


register_shape_calculator(
    "SklearnKernelCenterer", calculate_sklearn_kernel_centerer_output_shapes
)
register_shape_calculator(
    "SklearnKernelPCA", calculate_sklearn_kernel_pca_output_shapes
)
