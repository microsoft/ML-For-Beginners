# SPDX-License-Identifier: Apache-2.0


from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_array_feature_extractor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Extracts a subset of columns. This is used by *ColumnTransformer*.
    """
    column_indices_name = scope.get_unique_variable_name("column_indices")

    for i, ind in enumerate(operator.column_indices):
        if not isinstance(ind, int):
            raise RuntimeError(
                (
                    "Column {0}:'{1}' indices must be specified "
                    "as integers. This error may happen when "
                    "column names are used to define a "
                    "ColumnTransformer. Column name in input data "
                    "do not necessarily match input variables "
                    "defined for the ONNX model."
                ).format(i, ind)
            )
    container.add_initializer(
        column_indices_name,
        onnx_proto.TensorProto.INT64,
        [len(operator.column_indices)],
        operator.column_indices,
    )

    container.add_node(
        "ArrayFeatureExtractor",
        [operator.inputs[0].full_name, column_indices_name],
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        op_domain="ai.onnx.ml",
    )


register_converter(
    "SklearnArrayFeatureExtractor", convert_sklearn_array_feature_extractor
)
