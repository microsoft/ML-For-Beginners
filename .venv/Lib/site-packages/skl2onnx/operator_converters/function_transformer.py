# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_converter
from ..common._apply_operation import apply_concat, apply_identity
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_function_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    if op.func is not None:
        raise RuntimeError(
            "FunctionTransformer is not supported unless the "
            "transform function is None (= identity). "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    if len(operator.inputs) == 1:
        apply_identity(
            scope,
            operator.inputs[0].full_name,
            operator.outputs[0].full_name,
            container,
        )
    else:
        apply_concat(
            scope,
            [i.full_name for i in operator.inputs],
            operator.outputs[0].full_name,
            container,
            axis=1,
        )


register_converter("SklearnFunctionTransformer", convert_sklearn_function_transformer)
