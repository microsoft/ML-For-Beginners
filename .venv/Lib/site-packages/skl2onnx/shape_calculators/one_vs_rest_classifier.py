# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType
from ..common.shape_calculator import calculate_linear_classifier_output_shapes


def calculate_constant_predictor_output_shapes(operator):
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = Int64TensorType([N])
    operator.outputs[1].type.shape = [N, 2]


register_shape_calculator(
    "Sklearn_ConstantPredictor", calculate_constant_predictor_output_shapes
)

register_shape_calculator(
    "SklearnOneVsRestClassifier", calculate_linear_classifier_output_shapes
)
