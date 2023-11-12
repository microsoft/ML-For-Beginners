# SPDX-License-Identifier: Apache-2.0


import copy
import numpy as np
from ..common._registration import register_shape_calculator
from ..common.shape_calculator import calculate_linear_classifier_output_shapes
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_neighbours_transformer(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    n_samples_fit = operator.raw_operator.n_samples_fit_
    output_type = (
        DoubleTensorType
        if isinstance(operator.inputs[0].type, DoubleTensorType)
        else FloatTensorType
    )
    operator.outputs[0].type = output_type([N, n_samples_fit])


def calculate_sklearn_nearest_neighbours(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    neighbours = operator.raw_operator.n_neighbors
    operator.outputs[0].type = Int64TensorType([N, neighbours])
    operator.outputs[1].type.shape = [N, neighbours]


def calculate_sklearn_nearest_neighbours_regressor(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    if (
        hasattr(operator.raw_operator, "_y")
        and len(np.squeeze(operator.raw_operator._y).shape) == 1
    ):
        C = 1
    else:
        C = operator.raw_operator._y.shape[-1]
    operator.outputs[0].type.shape = [N, C]


def calculate_sklearn_nca(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType, DoubleTensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    output_type = (
        DoubleTensorType
        if isinstance(operator.inputs[0].type, DoubleTensorType)
        else FloatTensorType
    )
    n_components = operator.raw_operator.components_.shape[0]
    operator.outputs[0].type = output_type([N, n_components])


def calculate_sklearn_knn_imputer(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = operator.inputs[0].type.shape


register_shape_calculator(
    "SklearnKNeighborsRegressor", calculate_sklearn_nearest_neighbours_regressor
)
register_shape_calculator(
    "SklearnRadiusNeighborsRegressor", calculate_sklearn_nearest_neighbours_regressor
)
register_shape_calculator(
    "SklearnKNeighborsClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator(
    "SklearnRadiusNeighborsClassifier", calculate_linear_classifier_output_shapes
)
register_shape_calculator("SklearnKNNImputer", calculate_sklearn_knn_imputer)
register_shape_calculator(
    "SklearnKNeighborsTransformer", calculate_sklearn_neighbours_transformer
)
register_shape_calculator(
    "SklearnNearestNeighbors", calculate_sklearn_nearest_neighbours
)
register_shape_calculator(
    "SklearnNeighborhoodComponentsAnalysis", calculate_sklearn_nca
)
