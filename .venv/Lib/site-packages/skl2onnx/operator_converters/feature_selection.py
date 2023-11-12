# SPDX-License-Identifier: Apache-2.0


from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_feature_selection(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    # Get indices of the features selected
    index = op.get_support(indices=True)
    if len(index) == 0:
        raise RuntimeError(
            "Model '{}' did not select any feature. "
            "This model cannot be converted into ONNX."
            "".format(op.__class__.__name__)
        )
    output_name = operator.outputs[0].full_name
    if index.any():
        column_indices_name = scope.get_unique_variable_name("column_indices")

        container.add_initializer(
            column_indices_name, onnx_proto.TensorProto.INT64, [len(index)], index
        )

        container.add_node(
            "ArrayFeatureExtractor",
            [operator.inputs[0].full_name, column_indices_name],
            output_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )
    else:
        container.add_node(
            "ConstantOfShape", operator.inputs[0].full_name, output_name, op_version=9
        )


register_converter("SklearnGenericUnivariateSelect", convert_sklearn_feature_selection)
register_converter("SklearnRFE", convert_sklearn_feature_selection)
register_converter("SklearnRFECV", convert_sklearn_feature_selection)
register_converter("SklearnSelectFdr", convert_sklearn_feature_selection)
register_converter("SklearnSelectFpr", convert_sklearn_feature_selection)
register_converter("SklearnSelectFromModel", convert_sklearn_feature_selection)
register_converter("SklearnSelectFwe", convert_sklearn_feature_selection)
register_converter("SklearnSelectKBest", convert_sklearn_feature_selection)
register_converter("SklearnSelectPercentile", convert_sklearn_feature_selection)
register_converter("SklearnVarianceThreshold", convert_sklearn_feature_selection)
