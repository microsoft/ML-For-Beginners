# SPDX-License-Identifier: Apache-2.0

from sklearn.base import is_classifier
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_cast
from ..common.data_types import guess_proto_type
from .._parse import _parse_sklearn


def convert_pipeline(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    model = operator.raw_operator
    inputs = operator.inputs
    for step in model.steps:
        step_model = step[1]
        if is_classifier(step_model):
            scope.add_options(id(step_model), options={"zipmap": False})
            container.add_options(id(step_model), options={"zipmap": False})
        outputs = _parse_sklearn(scope, step_model, inputs, custom_parsers=None)
        inputs = outputs
    if len(outputs) != len(operator.outputs):
        raise RuntimeError(
            "Mismatch between pipeline output %d and "
            "last step outputs %d." % (len(outputs), len(operator.outputs))
        )
    for fr, to in zip(outputs, operator.outputs):
        if isinstance(to.type, type(fr.type)):
            container.add_node(
                "Identity",
                fr.full_name,
                to.full_name,
                name=scope.get_unique_operator_name("Id" + operator.onnx_name),
            )
        else:
            # If Pipeline output types are different with last stage output type
            apply_cast(
                scope,
                fr.full_name,
                to.full_name,
                container,
                operator_name=scope.get_unique_operator_name(
                    "Cast" + operator.onnx_name
                ),
                to=guess_proto_type(to.type),
            )


def convert_feature_union(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    raise NotImplementedError(
        "This converter not needed so far. It is usually handled " "during parsing."
    )


def convert_column_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    raise NotImplementedError(
        "This converter not needed so far. It is usually handled " "during parsing."
    )


register_converter("SklearnPipeline", convert_pipeline)
register_converter("SklearnFeatureUnion", convert_feature_union)
register_converter("SklearnColumnTransformer", convert_column_transformer)
