# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.shape_calculator import calculate_linear_classifier_output_shapes


register_shape_calculator(
    "SklearnLinearClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator("SklearnLinearSVC", calculate_linear_classifier_output_shapes)
register_shape_calculator(
    "SklearnAdaBoostClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnBaggingClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnBernoulliNB", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnCategoricalNB", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnComplementNB", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnGaussianNB", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnMultinomialNB", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnCalibratedClassifierCV", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnMLPClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnSGDClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnStackingClassifier", calculate_linear_classifier_output_shapes
)
