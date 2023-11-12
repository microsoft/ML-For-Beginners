# SPDX-License-Identifier: Apache-2.0


import numpy as np
from .._supported_operators import sklearn_operator_name_map
from ..common.data_types import Int64TensorType
from ..common._apply_operation import apply_cast, apply_concat, apply_div, apply_reshape
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def _calculate_proba(scope, operator, container, model):
    """
    This function calculates class probability scores for
    BaggingClassifier.
    """
    final_proba_name = operator.outputs[1].full_name
    proba_list = []
    options = container.get_options(model, dict(raw_scores=False))
    use_raw_scores = options["raw_scores"]
    has_proba = hasattr(model.estimators_[0], "predict_proba") or (
        use_raw_scores and hasattr(model.estimators_[0], "decision_function")
    )
    for index, estimator in enumerate(model.estimators_):
        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type, estimator)
        if container.has_options(estimator, "raw_scores"):
            container.add_options(id(estimator), {"raw_scores": use_raw_scores})
            scope.add_options(id(estimator), {"raw_scores": use_raw_scores})

        label_name = scope.declare_local_variable("label_%d" % index, Int64TensorType())
        proba_name = scope.declare_local_variable(
            "proba_%d" % index, operator.inputs[0].type.__class__()
        )

        features = model.estimators_features_[index]
        n_features = (
            model.n_features_in_
            if hasattr(model, "n_features_in_")
            else model.n_features_
        )
        if len(features) == n_features and list(features) == list(range(n_features)):
            this_operator.inputs = operator.inputs
        else:
            # subset of features
            feat_name = scope.declare_local_variable(
                "fsel_%d" % index, operator.inputs[0].type.__class__()
            )
            index_name = scope.get_unique_variable_name("index_name_%d" % index)
            container.add_initializer(
                index_name,
                onnx_proto.TensorProto.INT64,
                (len(features),),
                list(features),
            )
            container.add_node(
                "Gather",
                [operator.inputs[0].full_name, index_name],
                [feat_name.full_name],
                name=scope.get_unique_operator_name("GatherBG"),
                axis=1,
            )
            this_operator.inputs.append(feat_name)

        this_operator.outputs.append(label_name)
        this_operator.outputs.append(proba_name)
        proba_output_name = proba_name.onnx_name if has_proba else label_name.onnx_name
        reshape_dim_val = len(model.classes_) if has_proba else 1
        reshaped_proba_name = scope.get_unique_variable_name("reshaped_proba")
        apply_reshape(
            scope,
            proba_output_name,
            reshaped_proba_name,
            container,
            desired_shape=(1, -1, reshape_dim_val),
        )
        proba_list.append(reshaped_proba_name)
    merged_proba_name = scope.get_unique_variable_name("merged_proba")
    apply_concat(scope, proba_list, merged_proba_name, container, axis=0)
    if has_proba:
        if container.target_opset >= 18:
            axis_name = scope.get_unique_variable_name("axis")
            container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [0])
            container.add_node(
                "ReduceMean",
                [merged_proba_name, axis_name],
                final_proba_name,
                name=scope.get_unique_operator_name("ReduceMean"),
                keepdims=0,
            )
        else:
            container.add_node(
                "ReduceMean",
                merged_proba_name,
                final_proba_name,
                name=scope.get_unique_operator_name("ReduceMean"),
                axes=[0],
                keepdims=0,
            )
    else:
        n_estimators_name = scope.get_unique_variable_name("n_estimators")
        class_labels_name = scope.get_unique_variable_name("class_labels")
        equal_result_name = scope.get_unique_variable_name("equal_result")
        cast_output_name = scope.get_unique_variable_name("cast_output")
        reduced_proba_name = scope.get_unique_variable_name("reduced_proba")

        container.add_initializer(
            n_estimators_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [len(model.estimators_)],
        )
        container.add_initializer(
            class_labels_name,
            onnx_proto.TensorProto.INT64,
            [1, 1, len(model.estimators_[0].classes_)],
            model.estimators_[0].classes_,
        )

        container.add_node(
            "Equal",
            [class_labels_name, merged_proba_name],
            equal_result_name,
            name=scope.get_unique_operator_name("Equal"),
        )
        apply_cast(
            scope,
            equal_result_name,
            cast_output_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        if container.target_opset < 13:
            container.add_node(
                "ReduceSum",
                cast_output_name,
                reduced_proba_name,
                name=scope.get_unique_operator_name("ReduceSum"),
                axes=[0],
                keepdims=0,
            )
        else:
            axis_name = scope.get_unique_variable_name("axis")
            container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [1])
            container.add_node(
                "ReduceSum",
                [cast_output_name, axis_name],
                reduced_proba_name,
                keepdims=0,
                name=scope.get_unique_operator_name("ReduceSum"),
            )
        apply_div(
            scope,
            [reduced_proba_name, n_estimators_name],
            final_proba_name,
            container,
            broadcast=1,
        )
    return final_proba_name


def convert_sklearn_bagging_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for BaggingClassifier.
    """
    if scope.get_options(operator.raw_operator, dict(nocl=False))["nocl"]:
        raise RuntimeError(
            "Option 'nocl' is not implemented for operator '{}'.".format(
                operator.raw_operator.__class__.__name__
            )
        )

    bagging_op = operator.raw_operator
    classes = bagging_op.classes_
    output_shape = (-1,)
    classes_name = scope.get_unique_variable_name("classes")
    argmax_output_name = scope.get_unique_variable_name("argmax_output")
    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )
    class_type = onnx_proto.TensorProto.STRING

    if (
        np.issubdtype(bagging_op.classes_.dtype, np.floating)
        or bagging_op.classes_.dtype == np.bool_
    ):
        class_type = onnx_proto.TensorProto.INT32
        classes = classes.astype(np.int32)
    elif np.issubdtype(bagging_op.classes_.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])

    container.add_initializer(classes_name, class_type, classes.shape, classes)

    proba_name = _calculate_proba(scope, operator, container, bagging_op)
    container.add_node(
        "ArgMax",
        proba_name,
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
        cast_result_name = scope.get_unique_variable_name("cast_result")
        reshaped_result_name = scope.get_unique_variable_name("reshaped_result")
        apply_cast(
            scope,
            array_feature_extractor_result_name,
            cast_result_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
        apply_reshape(
            scope,
            cast_result_name,
            reshaped_result_name,
            container,
            desired_shape=output_shape,
        )
        apply_cast(
            scope,
            reshaped_result_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:  # string labels
        apply_reshape(
            scope,
            array_feature_extractor_result_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=output_shape,
        )


def convert_sklearn_bagging_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for BaggingRegressor.
    """
    bagging_op = operator.raw_operator
    proba_list = []
    for index, estimator in enumerate(bagging_op.estimators_):
        op_type = sklearn_operator_name_map[type(estimator)]
        this_operator = scope.declare_local_operator(op_type, estimator)

        features = bagging_op.estimators_features_[index]
        n_features = (
            bagging_op.n_features_in_
            if hasattr(bagging_op, "n_features_in_")
            else bagging_op.n_features_
        )
        if len(features) == n_features and list(features) == list(range(n_features)):
            this_operator.inputs = operator.inputs
        else:
            # subset of features
            feat_name = scope.declare_local_variable(
                "fsel_%d" % index, operator.inputs[0].type.__class__()
            )
            index_name = scope.get_unique_variable_name("index_name")
            container.add_initializer(
                index_name,
                onnx_proto.TensorProto.INT64,
                (len(features),),
                list(features),
            )
            container.add_node(
                "Gather",
                [operator.inputs[0].full_name, index_name],
                [feat_name.full_name],
                name=scope.get_unique_operator_name("GatherBG"),
                axis=1,
            )
            this_operator.inputs.append(feat_name)

        label_name = scope.declare_local_variable(
            "variable_%d" % index, this_operator.inputs[0].type.__class__()
        )
        this_operator.outputs.append(label_name)
        reshaped_proba_name = scope.get_unique_variable_name("reshaped_proba")
        apply_reshape(
            scope,
            label_name.onnx_name,
            reshaped_proba_name,
            container,
            desired_shape=(1, -1, 1),
        )
        proba_list.append(reshaped_proba_name)

    merged_proba_name = scope.get_unique_variable_name("merged_proba")
    apply_concat(scope, proba_list, merged_proba_name, container, axis=0)
    if container.target_opset >= 18:
        axis_name = scope.get_unique_variable_name("axis")
        container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [0])
        container.add_node(
            "ReduceMean",
            [merged_proba_name, axis_name],
            operator.outputs[0].full_name,
            name=scope.get_unique_operator_name("ReduceMean"),
            keepdims=0,
        )
    else:
        container.add_node(
            "ReduceMean",
            merged_proba_name,
            operator.outputs[0].full_name,
            name=scope.get_unique_operator_name("ReduceMean"),
            axes=[0],
            keepdims=0,
        )


register_converter(
    "SklearnBaggingClassifier",
    convert_sklearn_bagging_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "raw_scores": [True, False],
    },
)
register_converter("SklearnBaggingRegressor", convert_sklearn_bagging_regressor)
