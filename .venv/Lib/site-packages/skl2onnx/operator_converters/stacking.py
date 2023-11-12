# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_concat, apply_reshape
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_proto_type, Int64TensorType
from .._supported_operators import sklearn_operator_name_map


def _fetch_scores(
    scope, container, model, inputs, raw_scores=False, is_regressor=False
):
    op_type = sklearn_operator_name_map[type(model)]
    this_operator = scope.declare_local_operator(op_type, model)
    if container.has_options(model, "raw_scores"):
        container.add_options(id(model), {"raw_scores": raw_scores})
    this_operator.inputs.append(inputs)
    if is_regressor:
        output_proba = scope.declare_local_variable("variable", inputs.type.__class__())
        this_operator.outputs.append(output_proba)
    else:
        label_name = scope.declare_local_variable("label", Int64TensorType())
        this_operator.outputs.append(label_name)
        output_proba = scope.declare_local_variable(
            "probability_tensor", inputs.type.__class__()
        )
        this_operator.outputs.append(output_proba)

    proto_type = guess_proto_type(inputs.type)
    new_name = scope.get_unique_variable_name(output_proba.full_name + "_castio")
    apply_cast(scope, output_proba.full_name, new_name, container, to=proto_type)
    return new_name


def _add_passthrough_connection(operator, predictions):
    if operator.raw_operator.passthrough:
        predictions.append(operator.inputs[0].onnx_name)


def _transform_regressor(scope, operator, container, model):
    merged_prob_tensor = scope.get_unique_variable_name("merged_probability_tensor")

    predictions = [
        _fetch_scores(scope, container, est, operator.inputs[0], is_regressor=True)
        for est in model.estimators_
    ]

    _add_passthrough_connection(operator, predictions)

    apply_concat(scope, predictions, merged_prob_tensor, container, axis=1)
    return merged_prob_tensor


def _transform(scope, operator, container, model):
    merged_prob_tensor = scope.get_unique_variable_name("merged_probability_tensor")

    predictions = [
        _fetch_scores(
            scope,
            container,
            est,
            operator.inputs[0],
            raw_scores=meth == "decision_function",
        )
        for est, meth in zip(model.estimators_, model.stack_method_)
        if est != "drop"
    ]

    op = operator.raw_operator
    select_lact_column = len(op.classes_) == 2 and all(
        op.stack_method_[est_idx] == "predict_proba"
        for est_idx in range(0, len(op.estimators_))
    )
    if select_lact_column:
        column_index_name = scope.get_unique_variable_name("column_index")
        container.add_initializer(
            column_index_name, onnx_proto.TensorProto.INT64, [], [1]
        )
        new_predictions = []
        for ipred, pred in enumerate(predictions):
            prob1 = scope.get_unique_variable_name("stack_prob%d" % ipred)
            container.add_node(
                "ArrayFeatureExtractor",
                [pred, column_index_name],
                prob1,
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                op_domain="ai.onnx.ml",
            )
            new_predictions.append(prob1)
        predictions = new_predictions

    _add_passthrough_connection(operator, predictions)

    apply_concat(scope, predictions, merged_prob_tensor, container, axis=1)
    return merged_prob_tensor


def convert_sklearn_stacking_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for StackingClassifier. It invokes converters for each
    estimator, concatenating their results before calling converter
    for the final estimator on the concatenated score.
    """
    stacking_op = operator.raw_operator
    classes = stacking_op.classes_
    options = container.get_options(stacking_op, dict(raw_scores=False))
    use_raw_scores = options["raw_scores"]
    class_type = onnx_proto.TensorProto.STRING
    if (
        np.issubdtype(stacking_op.classes_.dtype, np.floating)
        or stacking_op.classes_.dtype == np.bool_
    ):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(stacking_op.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])

    classes_name = scope.get_unique_variable_name("classes")
    argmax_output_name = scope.get_unique_variable_name("argmax_output")
    reshaped_result_name = scope.get_unique_variable_name("reshaped_result")
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )

    container.add_initializer(classes_name, class_type, classes.shape, classes)

    merged_proba_tensor = _transform(scope, operator, container, stacking_op)
    merge_proba = scope.declare_local_variable(
        "merged_stacked_proba", operator.inputs[0].type.__class__()
    )
    container.add_node("Identity", [merged_proba_tensor], [merge_proba.onnx_name])
    prob = _fetch_scores(
        scope,
        container,
        stacking_op.final_estimator_,
        merge_proba,
        raw_scores=use_raw_scores,
    )
    container.add_node(
        "Identity",
        prob,
        operator.outputs[1].onnx_name,
        name=scope.get_unique_operator_name("OpProb"),
    )
    container.add_node(
        "ArgMax",
        prob,
        argmax_output_name,
        name=scope.get_unique_operator_name("ArgMax"),
        axis=1,
    )
    container.add_node(
        "ArrayFeatureExtractor",
        [classes_name, argmax_output_name],
        array_feature_extractor_result_name,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
    )

    if class_type == onnx_proto.TensorProto.INT32:
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


def convert_sklearn_stacking_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for StackingRegressor. It invokes converters for each
    estimator, concatenating their results before calling converter
    for the final estimator on the concatenated score.
    """
    stacking_op = operator.raw_operator

    merged_proba_tensor = _transform_regressor(scope, operator, container, stacking_op)
    merge_proba = scope.declare_local_variable(
        "merged_stacked_proba", operator.inputs[0].type.__class__()
    )
    container.add_node("Identity", [merged_proba_tensor], [merge_proba.onnx_name])
    prob = _fetch_scores(
        scope, container, stacking_op.final_estimator_, merge_proba, is_regressor=True
    )
    container.add_node(
        "Identity",
        prob,
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("Identity"),
    )


register_converter(
    "SklearnStackingClassifier",
    convert_sklearn_stacking_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "raw_scores": [True, False],
    },
)
register_converter("SklearnStackingRegressor", convert_sklearn_stacking_regressor)
