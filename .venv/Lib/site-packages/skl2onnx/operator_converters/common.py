# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_cast
from ..common.data_types import (
    Int64TensorType,
    FloatTensorType,
    DoubleTensorType,
    StringTensorType,
    guess_proto_type,
)


def concatenate_variables(scope, variables, container, main_type=None):
    """
    This function allocate operators to from a float tensor by concatenating
    all input variables. Notice that if all integer inputs would be converted
    to floats before concatenation.
    """
    if main_type is None:
        main_type = variables[0].type.__class__

    # Check if it's possible to concatenate those inputs.
    type_set = set(type(variable.type) for variable in variables)
    number_type_set = {
        FloatTensorType,
        Int64TensorType,
        DoubleTensorType,
        StringTensorType,
    }
    if any(itype not in number_type_set for itype in type_set):
        raise RuntimeError(
            "Numerical tensor(s) and string tensor(s) " "cannot be concatenated."
        )
    # input variables' names we want to concatenate
    input_names = []
    # dimensions of the variables that is going to be concatenated
    input_dims = []

    # Collect input variable names and do cast if needed
    for variable in variables:
        if not isinstance(variable.type, main_type):
            proto_type = guess_proto_type(main_type())
            new_name = scope.get_unique_variable_name("cast")
            apply_cast(scope, variable.full_name, new_name, container, to=proto_type)
            input_names.append(new_name)
        else:
            input_names.append(variable.full_name)
        # We assume input variables' shape are [1, C_1], ..., [1, C_n],
        # if there are n inputs.
        input_dims.append(variable.type.shape[1])

    if len(input_names) == 1:
        # No need to concatenate tensors if there is only one input
        return input_names[0]

    # To combine all inputs, we need a FeatureVectorizer
    op_type = "FeatureVectorizer"
    attrs = {
        "name": scope.get_unique_operator_name(op_type),
        "inputdimensions": input_dims,
    }
    # Create a variable name to capture feature vectorizer's output
    # Set up our FeatureVectorizer
    concatenated_name = scope.get_unique_variable_name("concatenated")
    container.add_node(
        op_type, input_names, concatenated_name, op_domain="ai.onnx.ml", **attrs
    )
    if main_type == FloatTensorType:
        return concatenated_name
    # Cast output as FeatureVectorizer always produces float32.
    concatenated_name_cast = scope.get_unique_variable_name("concatenated_cast")
    container.add_node(
        "CastLike", [concatenated_name, input_names[0]], concatenated_name_cast
    )

    return concatenated_name_cast
