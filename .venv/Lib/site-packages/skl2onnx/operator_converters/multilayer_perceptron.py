# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common.data_types import guess_proto_type
from ..common._apply_operation import (
    apply_add,
    apply_cast,
    apply_concat,
    apply_identity,
    apply_reshape,
    apply_sub,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def _forward_pass(scope, container, model, activations, proto_dtype):
    """
    Perform a forward pass on the network by computing the values of
    the neurons in the hidden layers and the output layer.
    """
    activations_map = {
        "identity": "Identity",
        "tanh": "Tanh",
        "logistic": "Sigmoid",
        "relu": "Relu",
        "softmax": "Softmax",
    }

    out_activation_result_name = scope.get_unique_variable_name(
        "out_activations_result"
    )

    # Iterate over the hidden layers
    for i in range(model.n_layers_ - 1):
        coefficient_name = scope.get_unique_variable_name("coefficient")
        intercepts_name = scope.get_unique_variable_name("intercepts")
        mul_result_name = scope.get_unique_variable_name("mul_result")
        add_result_name = scope.get_unique_variable_name("add_result")

        container.add_initializer(
            coefficient_name,
            proto_dtype,
            model.coefs_[i].shape,
            model.coefs_[i].ravel(),
        )
        container.add_initializer(
            intercepts_name,
            proto_dtype,
            [1, len(model.intercepts_[i])],
            model.intercepts_[i],
        )

        container.add_node(
            "MatMul",
            [activations[i], coefficient_name],
            mul_result_name,
            name=scope.get_unique_operator_name("MatMul"),
        )
        apply_add(
            scope,
            [mul_result_name, intercepts_name],
            add_result_name,
            container,
            broadcast=1,
        )

        # For the hidden layers
        if (i + 1) != (model.n_layers_ - 1):
            activations_result_name = scope.get_unique_variable_name("next_activations")

            container.add_node(
                activations_map[model.activation],
                add_result_name,
                activations_result_name,
                name=scope.get_unique_operator_name(activations_map[model.activation]),
            )
            activations.append(activations_result_name)

    # For the last layer
    container.add_node(
        activations_map[model.out_activation_],
        add_result_name,
        out_activation_result_name,
        name=scope.get_unique_operator_name(activations_map[model.activation]),
    )
    activations.append(out_activation_result_name)

    return activations


def _predict(scope, input_name, container, model, proto_dtype):
    """
    This function initialises the input layer, calls _forward_pass()
    and returns the final layer.
    """
    cast_input_name = scope.get_unique_variable_name("cast_input")

    apply_cast(scope, input_name, cast_input_name, container, to=proto_dtype)

    # forward propagate
    activations = _forward_pass(scope, container, model, [cast_input_name], proto_dtype)
    return activations[-1]


def convert_sklearn_mlp_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for MLPClassifier.
    This function calls _predict() which returns the probability scores
    of the positive class in case of binary labels and class
    probabilities in case of multi-class. It then calculates probability
    scores for the negative class in case of binary labels. It
    calculates the class labels and sets the output.
    """
    mlp_op = operator.raw_operator
    classes = mlp_op.classes_
    class_type = onnx_proto.TensorProto.STRING

    argmax_output_name = scope.get_unique_variable_name("argmax_output")
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    y_pred = _predict(
        scope, operator.inputs[0].full_name, container, mlp_op, proto_dtype
    )

    if (
        np.issubdtype(mlp_op.classes_.dtype, np.floating)
        or mlp_op.classes_.dtype == np.bool_
    ):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(mlp_op.classes_.dtype, np.integer):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])

    if len(classes) == 2:
        unity_name = scope.get_unique_variable_name("unity")
        negative_class_proba_name = scope.get_unique_variable_name(
            "negative_class_proba"
        )
        container.add_initializer(unity_name, proto_dtype, [], [1])
        apply_sub(
            scope,
            [unity_name, y_pred],
            negative_class_proba_name,
            container,
            broadcast=1,
        )
        apply_concat(
            scope,
            [negative_class_proba_name, y_pred],
            operator.outputs[1].full_name,
            container,
            axis=1,
        )
    else:
        apply_identity(scope, y_pred, operator.outputs[1].full_name, container)

    if mlp_op._label_binarizer.y_type_ == "multilabel-indicator":
        binariser_output_name = scope.get_unique_variable_name("binariser_output")

        container.add_node(
            "Binarizer",
            y_pred,
            binariser_output_name,
            threshold=0.5,
            op_domain="ai.onnx.ml",
        )
        apply_cast(
            scope,
            binariser_output_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:
        classes_name = scope.get_unique_variable_name("classes")
        container.add_initializer(classes_name, class_type, classes.shape, classes)

        container.add_node(
            "ArgMax",
            operator.outputs[1].full_name,
            argmax_output_name,
            axis=1,
            name=scope.get_unique_operator_name("ArgMax"),
        )
        container.add_node(
            "ArrayFeatureExtractor",
            [classes_name, argmax_output_name],
            array_feature_extractor_result_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )

        if class_type == onnx_proto.TensorProto.INT32:
            reshaped_result_name = scope.get_unique_variable_name("reshaped_result")

            apply_reshape(
                scope,
                array_feature_extractor_result_name,
                reshaped_result_name,
                container,
                desired_shape=(-1,),
            )
            apply_cast(
                scope,
                reshaped_result_name,
                operator.outputs[0].full_name,
                container,
                to=onnx_proto.TensorProto.INT64,
            )
        else:
            apply_reshape(
                scope,
                array_feature_extractor_result_name,
                operator.outputs[0].full_name,
                container,
                desired_shape=(-1,),
            )


def convert_sklearn_mlp_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for MLPRegressor.
    This function calls _predict() which returns the scores.
    """
    mlp_op = operator.raw_operator

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    y_pred = _predict(
        scope, operator.inputs[0].full_name, container, mlp_op, proto_dtype=proto_dtype
    )
    apply_reshape(
        scope, y_pred, operator.output_full_names, container, desired_shape=(-1, 1)
    )


register_converter(
    "SklearnMLPClassifier",
    convert_sklearn_mlp_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter("SklearnMLPRegressor", convert_sklearn_mlp_regressor)
