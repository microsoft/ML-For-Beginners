# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType


def calculate_sklearn_random_trees_embedding_output_shapes(operator):
    op = operator.raw_operator.one_hot_encoder_
    categories_len = 0
    for index, categories in enumerate(op.categories_):
        if hasattr(op, "drop_idx_") and op.drop_idx_ is not None:
            categories = categories[np.arange(len(categories)) != op.drop_idx_[index]]
        categories_len += len(categories)
    instances = operator.inputs[0].get_first_dimension()
    if np.issubdtype(op.dtype, np.signedinteger):
        operator.outputs[0].type = Int64TensorType([instances, categories_len])
    else:
        operator.outputs[0].type = FloatTensorType([instances, categories_len])


register_shape_calculator(
    "SklearnRandomTreesEmbedding",
    calculate_sklearn_random_trees_embedding_output_shapes,
)
