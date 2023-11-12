# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_shape_calculator


def pipeline_shape_calculator(operator):
    pass


def feature_union_shape_calculator(operator):
    pass


def column_transformer_shape_calculator(operator):
    pass


register_shape_calculator("SklearnPipeline", pipeline_shape_calculator)
register_shape_calculator("SklearnFeatureUnion", feature_union_shape_calculator)
register_shape_calculator(
    "SklearnColumnTransformer", column_transformer_shape_calculator
)
