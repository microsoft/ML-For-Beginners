# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common.data_types import (
    StringTensorType,
    Int64TensorType,
    FloatTensorType,
    DoubleTensorType,
)
from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers
from ..common.utils import check_input_and_output_types


def calculate_sklearn_feature_hasher(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[StringTensorType, Int64TensorType]
    )

    N = operator.inputs[0].get_first_dimension()
    model = operator.raw_operator
    shape = [N, model.n_features]
    if model.dtype == np.float32:
        operator.outputs[0].type = FloatTensorType(shape=shape)
    elif model.dtype == np.float64:
        operator.outputs[0].type = DoubleTensorType(shape=shape)
    elif model.dtype in (np.int32, np.uint32, np.int64):
        operator.outputs[0].type = Int64TensorType(shape=shape)
    else:
        raise RuntimeError(
            f"Converter is not implemented for FeatureHasher.dtype={model.dtype}."
        )


register_shape_calculator("SklearnFeatureHasher", calculate_sklearn_feature_hasher)
