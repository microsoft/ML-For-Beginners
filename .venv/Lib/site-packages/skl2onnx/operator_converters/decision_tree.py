# SPDX-License-Identifier: Apache-2.0


import numbers
import numpy as np
from onnx.numpy_helper import from_array
from ..common._apply_operation import (
    apply_cast,
    apply_concat,
    apply_div,
    apply_mul,
    apply_reshape,
    apply_reducesum,
    apply_transpose,
)
from ..common._registration import register_converter
from ..common.data_types import (
    BooleanTensorType,
    Int64TensorType,
    guess_numpy_type,
    guess_proto_type,
)
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto


def populate_tree_attributes(model, name, dtype):
    """Construct attrs dictionary to be used in predict()
    while adding a node with TreeEnsembleClassifier ONNX op.
    """
    attrs = {}
    attrs["name"] = name
    attrs["post_transform"] = "NONE"
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["class_treeids"] = []
    attrs["class_nodeids"] = []
    attrs["class_ids"] = []
    attrs["class_weights"] = []
    attrs["classlabels_int64s"] = list(range(model.tree_.node_count))

    for i in range(model.tree_.node_count):
        node_id = i
        if model.tree_.children_left[i] > i and model.tree_.children_right[i] > i:
            feat = model.tree_.feature[i]
            thresh = model.tree_.threshold[i]
            left = model.tree_.children_left[i]
            right = model.tree_.children_right[i]
            mode = "BRANCH_LEQ"
        else:
            feat, thresh, left, right = 0, 0.0, 0, 0
            mode = "LEAF"
        attrs["nodes_nodeids"].append(node_id)
        attrs["nodes_treeids"].append(0)
        attrs["nodes_featureids"].append(feat)
        attrs["nodes_modes"].append(mode)
        attrs["nodes_truenodeids"].append(left)
        attrs["nodes_falsenodeids"].append(right)
        attrs["nodes_missing_value_tracks_true"].append(False)
        attrs["nodes_hitrates"].append(1.0)
        attrs["nodes_values"].append(thresh)
        if mode == "LEAF":
            attrs["class_ids"].append(node_id)
            attrs["class_weights"].append(1.0)
            attrs["class_treeids"].append(0)
            attrs["class_nodeids"].append(node_id)
    if dtype is not None:
        for k in attrs:
            if k in ("node_values", "class_weights", "target_weights"):
                attrs[k] = np.array(attrs[k], dtype=dtype)
    return attrs


def predict(
    model, scope, operator, container, op_type, op_domain, op_version, is_ensemble=False
):
    """Predict target and calculate probability scores."""
    indices_name = scope.get_unique_variable_name("indices")
    dummy_proba_name = scope.get_unique_variable_name("dummy_proba")
    values_name = scope.get_unique_variable_name("values")
    out_values_name = scope.get_unique_variable_name("out_indices")
    transposed_result_name = scope.get_unique_variable_name("transposed_result")
    proba_output_name = scope.get_unique_variable_name("proba_output")
    cast_result_name = scope.get_unique_variable_name("cast_result")
    reshaped_indices_name = scope.get_unique_variable_name("reshaped_indices")
    sum_output_name = scope.get_unique_variable_name("sum_proba")
    value = model.tree_.value.transpose(1, 2, 0)

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32

    container.add_initializer(values_name, proto_dtype, value.shape, value.ravel())

    input_name = operator.input_full_names
    if isinstance(operator.inputs[0].type, BooleanTensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(scope, input_name, cast_input_name, container, to=proto_dtype)
        input_name = cast_input_name

    if model.tree_.node_count > 1:
        attrs = populate_tree_attributes(
            model, scope.get_unique_operator_name(op_type), dtype
        )
        container.add_node(
            op_type,
            input_name,
            [indices_name, dummy_proba_name],
            op_domain=op_domain,
            op_version=op_version,
            **attrs,
        )
    else:
        zero_name = scope.get_unique_variable_name("zero")
        zero_matrix_name = scope.get_unique_variable_name("zero_matrix")
        reduced_zero_matrix_name = scope.get_unique_variable_name("reduced_zero_matrix")

        container.add_initializer(zero_name, proto_dtype, [], [0])
        apply_mul(
            scope, [input_name[0], zero_name], zero_matrix_name, container, broadcast=1
        )
        if container.target_opset < 13:
            container.add_node(
                "ReduceSum",
                zero_matrix_name,
                reduced_zero_matrix_name,
                axes=[1],
                name=scope.get_unique_operator_name("ReduceSum"),
            )
        else:
            axis_name = scope.get_unique_variable_name("axis")
            container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [1])
            container.add_node(
                "ReduceSum",
                [zero_matrix_name, axis_name],
                reduced_zero_matrix_name,
                name=scope.get_unique_operator_name("ReduceSum"),
            )
        apply_cast(
            scope,
            reduced_zero_matrix_name,
            indices_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    apply_reshape(
        scope, indices_name, reshaped_indices_name, container, desired_shape=[1, -1]
    )
    container.add_node(
        "ArrayFeatureExtractor",
        [values_name, reshaped_indices_name],
        out_values_name,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
    )
    apply_transpose(
        scope, out_values_name, proba_output_name, container, perm=(0, 2, 1)
    )

    if is_ensemble:
        proba_result_name = scope.get_unique_variable_name("proba_result")
        apply_reducesum(
            scope, proba_output_name, sum_output_name, container, keepdims=1, axes=[2]
        )
        apply_div(
            scope, [proba_output_name, sum_output_name], proba_result_name, container
        )
        return proba_result_name
    else:
        apply_cast(
            scope,
            proba_output_name,
            cast_result_name,
            container,
            to=onnx_proto.TensorProto.BOOL,
        )
        apply_cast(
            scope,
            cast_result_name,
            operator.outputs[1].full_name,
            container,
            to=proto_dtype,
        )
        apply_transpose(
            scope, out_values_name, transposed_result_name, container, perm=(2, 1, 0)
        )
        return transposed_result_name


def _append_decision_output(
    input_name,
    attrs,
    fct_label,
    n_out,
    scope,
    operator,
    container,
    op_type="TreeEnsembleClassifier",
    op_domain="ai.onnx.ml",
    op_version=1,
    cast_encode=False,
    regression=False,
    dtype=np.float32,
    overwrite_tree=None,
):
    attrs = attrs.copy()
    attrs["name"] = scope.get_unique_operator_name(op_type)
    attrs["n_targets"] = 1
    attrs["post_transform"] = "NONE"
    if regression:
        attrs["target_weights"] = np.array(
            [float(_) for _ in attrs["target_nodeids"]], dtype=dtype
        )
    else:
        attrs["target_ids"] = [0 for _ in attrs["class_ids"]]
        attrs["target_weights"] = [float(_) for _ in attrs["class_nodeids"]]
        attrs["target_nodeids"] = attrs["class_nodeids"]
        attrs["target_treeids"] = attrs["class_treeids"]

    rem = [k for k in attrs if k.startswith("class")]
    for k in rem:
        del attrs[k]
    dpath = scope.get_unique_variable_name("dpath")
    container.add_node(
        op_type.replace("Classifier", "Regressor"),
        input_name,
        dpath,
        op_domain=op_domain,
        op_version=op_version,
        **attrs,
    )

    if n_out is None:
        final_name = scope.get_unique_variable_name("dpatho")
    else:
        final_name = operator.outputs[n_out].full_name

    if cast_encode:
        apply_cast(
            scope,
            dpath,
            final_name,
            container,
            to=onnx_proto.TensorProto.INT64,
            operator_name=scope.get_unique_operator_name("TreePathType"),
        )
    else:
        op = operator.raw_operator
        labels = fct_label(overwrite_tree if overwrite_tree is not None else op.tree_)
        ordered = list(sorted(labels.items()))
        keys = [float(_[0]) for _ in ordered]
        values = [_[1] for _ in ordered]
        name = scope.get_unique_variable_name("spath")
        container.add_node(
            "LabelEncoder",
            dpath,
            name,
            op_domain=op_domain,
            op_version=2,
            default_string="0",
            keys_floats=keys,
            values_strings=values,
            name=scope.get_unique_operator_name("TreePath"),
        )
        apply_reshape(
            scope,
            name,
            final_name,
            container,
            desired_shape=(-1, 1),
            operator_name=scope.get_unique_operator_name("TreePathShape"),
        )

    return final_name


def convert_sklearn_decision_tree_classifier(
    scope,
    operator,
    container,
    op_type="TreeEnsembleClassifier",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    try:
        dtype = guess_numpy_type(operator.inputs[0].type)
    except NotImplementedError as e:
        raise RuntimeError("Unknown variable {}.".format(operator.inputs[0])) from e
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator
    options = scope.get_options(op, dict(decision_path=False, decision_leaf=False))
    if np.asarray(op.classes_).size == 1:
        # The model was trained with one label.
        # There is no need to build a tree.
        if op.n_outputs_ != 1:
            raise RuntimeError(
                f"One training class and multiple outputs is not "
                f"supported yet for class {op.__class__.__name__!r}."
            )
        if options["decision_path"] or options["decision_leaf"]:
            raise RuntimeError(
                f"One training class, option 'decision_path' "
                f"or 'decision_leaf' are not supported for "
                f"class {op.__class__.__name__!r}."
            )

        zero = scope.get_unique_variable_name("zero")
        one = scope.get_unique_variable_name("one")
        new_shape = scope.get_unique_variable_name("new_shape")
        container.add_initializer(zero, onnx_proto.TensorProto.INT64, [1], [0])
        container.add_initializer(one, onnx_proto.TensorProto.INT64, [1], [1])
        container.add_initializer(new_shape, onnx_proto.TensorProto.INT64, [2], [-1, 1])
        shape = scope.get_unique_variable_name("shape")
        container.add_node("Shape", [operator.inputs[0].full_name], [shape])
        shape_sliced = scope.get_unique_variable_name("shape_sliced")
        container.add_node("Slice", [shape, zero, one, zero], [shape_sliced])

        # labels
        container.add_node(
            "ConstantOfShape",
            [shape_sliced],
            [operator.outputs[0].full_name],
            value=from_array(np.array([op.classes_[0]], dtype=np.int64)),
        )

        # probabilities
        probas = scope.get_unique_variable_name("probas")
        container.add_node(
            "ConstantOfShape",
            [shape_sliced],
            [probas],
            value=from_array(np.array([1], dtype=dtype)),
        )
        container.add_node(
            "Reshape", [probas, new_shape], [operator.outputs[1].full_name]
        )
        return

    if op.n_outputs_ == 1:
        attrs = get_default_tree_classifier_attribute_pairs()
        attrs["name"] = scope.get_unique_operator_name(op_type)
        classes = get_label_classes(scope, op)

        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)
        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attrs["classlabels_int64s"] = class_labels
        elif all(isinstance(i, str) for i in classes):
            class_labels = [str(i) for i in classes]
            attrs["classlabels_strings"] = class_labels
        else:
            raise ValueError("Labels must be all integers or all strings.")

        add_tree_to_attribute_pairs(
            attrs, True, op.tree_, 0, 1.0, 0, True, True, dtype=dtype
        )
        input_name = operator.input_full_names
        if isinstance(operator.inputs[0].type, BooleanTensorType):
            cast_input_name = scope.get_unique_variable_name("cast_input")

            apply_cast(
                scope,
                input_name,
                cast_input_name,
                container,
                to=onnx_proto.TensorProto.FLOAT,
            )
            input_name = cast_input_name

        if dtype is not None:
            for k in attrs:
                if k in (
                    "nodes_values",
                    "class_weights",
                    "target_weights",
                    "nodes_hitrates",
                    "base_values",
                ):
                    attrs[k] = np.array(attrs[k], dtype=dtype)

        container.add_node(
            op_type,
            input_name,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain=op_domain,
            op_version=op_version,
            **attrs,
        )

        n_out = 2
        if options["decision_path"]:
            # decision_path
            _append_decision_output(
                input_name,
                attrs,
                _build_labels_path,
                n_out,
                scope,
                operator,
                container,
                op_type=op_type,
                op_domain=op_domain,
                op_version=op_version,
                dtype=dtype,
            )
            n_out += 1
        if options["decision_leaf"]:
            # decision_path
            _append_decision_output(
                input_name,
                attrs,
                _build_labels_leaf,
                n_out,
                scope,
                operator,
                container,
                op_type=op_type,
                op_domain=op_domain,
                op_version=op_version,
                cast_encode=True,
                dtype=dtype,
            )
            n_out += 1
    else:
        transposed_result_name = predict(
            op, scope, operator, container, op_type, op_domain, op_version
        )
        predictions = []
        for k in range(op.n_outputs_):
            preds_name = scope.get_unique_variable_name("preds")
            reshaped_preds_name = scope.get_unique_variable_name("reshaped_preds")
            k_name = scope.get_unique_variable_name("k_column")
            out_k_name = scope.get_unique_variable_name("out_k_column")
            argmax_output_name = scope.get_unique_variable_name("argmax_output")
            classes_name = scope.get_unique_variable_name("classes")
            reshaped_result_name = scope.get_unique_variable_name("reshaped_result")

            container.add_initializer(k_name, onnx_proto.TensorProto.INT64, [], [k])
            container.add_initializer(
                classes_name,
                onnx_proto.TensorProto.INT64,
                op.classes_[k].shape,
                [int(i) for i in op.classes_[k]],
            )

            container.add_node(
                "ArrayFeatureExtractor",
                [transposed_result_name, k_name],
                out_k_name,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            )
            container.add_node(
                "ArgMax",
                out_k_name,
                argmax_output_name,
                name=scope.get_unique_operator_name("ArgMax"),
                axis=1,
            )
            apply_reshape(
                scope,
                argmax_output_name,
                reshaped_result_name,
                container,
                desired_shape=(1, -1),
            )
            container.add_node(
                "ArrayFeatureExtractor",
                [classes_name, reshaped_result_name],
                preds_name,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            )
            apply_reshape(
                scope, preds_name, reshaped_preds_name, container, desired_shape=(-1, 1)
            )
            predictions.append(reshaped_preds_name)
        apply_concat(
            scope, predictions, operator.outputs[0].full_name, container, axis=1
        )

        if options["decision_path"]:
            raise RuntimeError(
                "Option decision_path for multi-outputs " "is not implemented yet."
            )
        if options["decision_leaf"]:
            raise RuntimeError(
                "Option decision_leaf for multi-outputs " "is not implemented yet."
            )


def convert_sklearn_decision_tree_regressor(
    scope,
    operator,
    container,
    op_type="TreeEnsembleRegressor",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs["name"] = scope.get_unique_operator_name(op_type)
    attrs["n_targets"] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(
        attrs, False, op.tree_, 0, 1.0, 0, False, True, dtype=dtype
    )

    if dtype is not None:
        for k in attrs:
            if k in (
                "nodes_values",
                "class_weights",
                "target_weights",
                "nodes_hitrates",
                "base_values",
            ):
                attrs[k] = np.array(attrs[k], dtype=dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope,
            operator.input_full_names,
            cast_input_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        input_name = [cast_input_name]

    container.add_node(
        op_type,
        input_name,
        operator.outputs[0].full_name,
        op_domain=op_domain,
        op_version=op_version,
        **attrs,
    )

    options = scope.get_options(op, dict(decision_path=False, decision_leaf=False))

    # decision_path
    n_out = 1
    if options["decision_path"]:
        # decision_path
        _append_decision_output(
            input_name,
            attrs,
            _build_labels_path,
            n_out,
            scope,
            operator,
            container,
            op_type=op_type,
            op_domain=op_domain,
            op_version=op_version,
            regression=True,
        )
        n_out += 1
    if options["decision_leaf"]:
        # decision_path
        _append_decision_output(
            input_name,
            attrs,
            _build_labels_leaf,
            n_out,
            scope,
            operator,
            container,
            op_type=op_type,
            op_domain=op_domain,
            op_version=op_version,
            regression=True,
            cast_encode=True,
        )
        n_out += 1


def _recursive_build_labels(tree, index, current):
    current[index] = True
    if tree.children_left[index] == -1:
        yield (index, current.copy())
    else:
        for it in _recursive_build_labels(tree, tree.children_left[index], current):
            yield it
        for it in _recursive_build_labels(tree, tree.children_right[index], current):
            yield it
    current[index] = False


def _build_labels_path(tree):
    paths = {}
    current = {}

    for leave_index, path in _recursive_build_labels(tree, 0, current):
        spath = ["0" for _ in range(tree.node_count)]
        for nodeid, b in path.items():
            if b:
                spath[nodeid] = "1"
        paths[leave_index] = "".join(spath)
    return paths


def _build_labels_leaf(tree):
    paths = {}
    current = {}

    for leave_index, path in _recursive_build_labels(tree, 0, current):
        paths[leave_index] = leave_index
    return paths


register_converter(
    "SklearnDecisionTreeClassifier",
    convert_sklearn_decision_tree_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "decision_path": [True, False],
        "decision_leaf": [True, False],
    },
)
register_converter(
    "SklearnDecisionTreeRegressor",
    convert_sklearn_decision_tree_regressor,
    options={"decision_path": [True, False], "decision_leaf": [True, False]},
)
register_converter(
    "SklearnExtraTreeClassifier",
    convert_sklearn_decision_tree_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "decision_path": [True, False],
        "decision_leaf": [True, False],
    },
)
register_converter(
    "SklearnExtraTreeRegressor",
    convert_sklearn_decision_tree_regressor,
    options={"decision_path": [True, False], "decision_leaf": [True, False]},
)
