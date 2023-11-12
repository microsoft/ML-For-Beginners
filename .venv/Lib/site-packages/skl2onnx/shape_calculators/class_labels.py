# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_class_labels(operator):
    check_input_and_output_numbers(operator, output_count_range=1)


register_shape_calculator("SklearnClassLabels", calculate_sklearn_class_labels)
