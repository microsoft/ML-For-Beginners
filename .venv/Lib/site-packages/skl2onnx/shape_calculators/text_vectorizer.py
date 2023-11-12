# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_text_vectorizer_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. Map ---> [1, C]

    C is the total number of allowed keys in the input dictionary.
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    C = max(operator.raw_operator.vocabulary_.values()) + 1
    operator.outputs[0].type.shape = [None, C]


register_shape_calculator(
    "SklearnCountVectorizer", calculate_sklearn_text_vectorizer_output_shapes
)
register_shape_calculator(
    "SklearnTfidfVectorizer", calculate_sklearn_text_vectorizer_output_shapes
)
