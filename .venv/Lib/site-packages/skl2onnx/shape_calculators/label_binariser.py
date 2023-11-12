# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType, StringTensorType
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_label_binariser_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[Int64TensorType, StringTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([N, len(operator.raw_operator.classes_)])


register_shape_calculator(
    "SklearnLabelBinarizer", calculate_sklearn_label_binariser_output_shapes
)
