# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_dict_vectorizer_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. Map ---> [1, C]

    C is the total number of allowed keys in the input dictionary.
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    C = len(operator.raw_operator.feature_names_)
    operator.outputs[0].type.shape = [None, C]


register_shape_calculator(
    "SklearnDictVectorizer", calculate_sklearn_dict_vectorizer_output_shapes
)
