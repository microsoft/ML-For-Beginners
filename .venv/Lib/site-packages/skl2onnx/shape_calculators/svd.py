# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_truncated_svd_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, K]

    Transform feature dimension from C to K
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType],
    )

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Only 2-D tensor(s) can be input(s).")

    cls_type = operator.inputs[0].type.__class__
    if cls_type != DoubleTensorType:
        cls_type = FloatTensorType
    N = operator.inputs[0].get_first_dimension()
    K = (
        operator.raw_operator.n_components
        if operator.type == "SklearnTruncatedSVD"
        else operator.raw_operator.n_components_
    )

    operator.outputs[0].type = cls_type([N, K])


register_shape_calculator(
    "SklearnIncrementalPCA", calculate_sklearn_truncated_svd_output_shapes
)
register_shape_calculator("SklearnPCA", calculate_sklearn_truncated_svd_output_shapes)
register_shape_calculator(
    "SklearnTruncatedSVD", calculate_sklearn_truncated_svd_output_shapes
)
