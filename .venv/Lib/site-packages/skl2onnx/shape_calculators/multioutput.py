# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers
from ..common.data_types import SequenceType

_stack = []


def multioutput_regressor_shape_calculator(operator):
    """Shape calculator for MultiOutputRegressor"""
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    i = operator.inputs[0]
    o = operator.outputs[0]
    N = i.get_first_dimension()
    C = len(operator.raw_operator.estimators_)
    o.type = o.type.__class__([N, C])


def multioutput_classifier_shape_calculator(operator):
    """Shape calculator for MultiOutputClassifier"""
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=2)
    if not isinstance(operator.outputs[1].type, SequenceType):
        raise RuntimeError(
            "Probabilites should be a sequence not %r." "" % operator.outputs[1].type
        )
    i = operator.inputs[0]
    outputs = operator.outputs
    N = i.get_first_dimension()
    C = len(operator.raw_operator.estimators_)
    outputs[0].type.shape = [N, C]


register_shape_calculator(
    "SklearnMultiOutputRegressor", multioutput_regressor_shape_calculator
)
register_shape_calculator(
    "SklearnMultiOutputClassifier", multioutput_classifier_shape_calculator
)
