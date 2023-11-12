# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator


def calculate_sklearn_sequence_at(operator):
    pass


def calculate_sklearn_sequence_construct(operator):
    pass


register_shape_calculator("SklearnSequenceAt", calculate_sklearn_sequence_at)
register_shape_calculator(
    "SklearnSequenceConstruct", calculate_sklearn_sequence_construct
)
