# SPDX-License-Identifier: Apache-2.0


import copy
from ..common._registration import register_shape_calculator


def calculate_sklearn_function_transformer_output_shapes(operator):
    """
    This operator is used only to merge columns in a pipeline.
    Only identity function is supported.
    """
    if operator.raw_operator.func is not None:
        raise RuntimeError(
            "FunctionTransformer is not supported unless the "
            "transform function is None (= identity). "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    N = operator.inputs[0].get_first_dimension()
    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] is not None:
            C += variable.type.shape[1]
        else:
            C = None
            break

    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, C]


register_shape_calculator(
    "SklearnFunctionTransformer", calculate_sklearn_function_transformer_output_shapes
)
