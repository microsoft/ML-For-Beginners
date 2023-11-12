# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common._registration import register_shape_calculator
from ..common.data_types import Int64TensorType, FloatTensorType


def calculate_sklearn_ordinal_encoder_output_shapes(operator):
    ordinal_op = operator.raw_operator
    op_features = sum(list(map(lambda x: x.type.shape[1], operator.inputs)))
    if np.issubdtype(ordinal_op.dtype, np.floating):
        operator.outputs[0].type = FloatTensorType(
            [operator.inputs[0].get_first_dimension(), op_features]
        )
    else:
        operator.outputs[0].type = Int64TensorType(
            [operator.inputs[0].get_first_dimension(), op_features]
        )


register_shape_calculator(
    "SklearnOrdinalEncoder", calculate_sklearn_ordinal_encoder_output_shapes
)
