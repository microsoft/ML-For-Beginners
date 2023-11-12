# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType, DoubleTensorType
from ..common.utils import check_input_and_output_types


def calculate_sklearn_kmeans_output_shapes(operator):
    check_input_and_output_types(
        operator,
        good_input_types=[Int64TensorType, FloatTensorType, DoubleTensorType],
        good_output_types=[Int64TensorType, FloatTensorType, DoubleTensorType],
    )
    if len(operator.inputs) != 1:
        raise RuntimeError("Only one input vector is allowed for KMeans.")
    if len(operator.outputs) != 2:
        raise RuntimeError("Two outputs are expected for KMeans.")

    variable = operator.inputs[0]
    N = variable.get_first_dimension()
    op = operator.raw_operator
    operator.outputs[0].type.shape = [N]
    operator.outputs[1].type.shape = [N, op.n_clusters]


register_shape_calculator("SklearnKMeans", calculate_sklearn_kmeans_output_shapes)
register_shape_calculator(
    "SklearnMiniBatchKMeans", calculate_sklearn_kmeans_output_shapes
)
