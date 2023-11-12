# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import is_regressor
from sklearn.svm import LinearSVC
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_concat,
    apply_identity,
    apply_mul,
    apply_reshape,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_normalization
from ..common._apply_operation import (
    apply_slice,
    apply_sub,
    apply_cast,
    apply_abs,
    apply_add,
    apply_div,
)
from ..common.utils_classifier import _finalize_converter_classes
from ..common.data_types import guess_proto_type, Int64TensorType
from ..algebra.onnx_ops import OnnxReshape, OnnxShape, OnnxSlice, OnnxTile
from .._supported_operators import sklearn_operator_name_map


def _iteration_one_versus(
    scope,
    container,
    inputs,
    i,
    estimator,
    cl_type,
    proto_dtype,
    use_raw_scores=True,
    prob_shape=None,
):
    op_type = sklearn_operator_name_map[type(estimator)]

    this_operator = scope.declare_local_operator(op_type, raw_model=estimator)
    this_operator.inputs = inputs

    if is_regressor(estimator):
        score_name = scope.declare_local_variable("score_%d" % i, cl_type())
        this_operator.outputs.append(score_name)

        if hasattr(estimator, "coef_") and len(estimator.coef_.shape) == 2:
            raise RuntimeError(
                "OneVsRestClassifier or OneVsOneClassifier accepts "
                "regressor with only one target."
            )
        p1 = score_name.onnx_name
    else:
        if container.has_options(estimator, "raw_scores"):
            container.add_options(id(estimator), {"raw_scores": use_raw_scores})
            scope.add_options(id(estimator), {"raw_scores": use_raw_scores})
        label_name = scope.declare_local_variable("label_%d" % i, Int64TensorType())
        prob_name = scope.declare_local_variable("proba_%d" % i, cl_type())
        this_operator.outputs.append(label_name)
        this_operator.outputs.append(prob_name)

        # gets the probability for the class 1
        p1 = scope.get_unique_variable_name("probY_%d" % i)
        if isinstance(estimator, LinearSVC):
            apply_identity(scope, prob_name.onnx_name, p1, container)
        else:
            apply_slice(
                scope,
                prob_name.onnx_name,
                p1,
                container,
                starts=[1],
                ends=[2],
                axes=[1],
            )
    return None, None, p1


def convert_one_vs_rest_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *OneVsRestClassifier* into *ONNX* format.
    """
    if scope.get_options(operator.raw_operator, dict(nocl=False))["nocl"]:
        raise RuntimeError(
            "Option 'nocl' is not implemented for operator '{}'.".format(
                operator.raw_operator.__class__.__name__
            )
        )
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    options = container.get_options(op, dict(raw_scores=False))
    use_raw_scores = options["raw_scores"]
    probs_names = []
    cl_type = operator.inputs[0].type.__class__
    prob_shape = None
    for i, estimator in enumerate(op.estimators_):
        prob_shape, _, p1 = _iteration_one_versus(
            scope,
            container,
            operator.inputs,
            i,
            estimator,
            cl_type,
            proto_dtype,
            use_raw_scores,
            prob_shape=prob_shape,
        )
        probs_names.append(p1)

    if op.multilabel_:
        # concatenates outputs
        conc_name = operator.outputs[1].full_name
        apply_concat(scope, probs_names, conc_name, container, axis=1)

        # builds the labels (matrix with integer)
        # scikit-learn may use probabilities or raw score
        # but ONNX converters only uses probabilities.
        # https://github.com/scikit-learn/scikit-learn/sklearn/
        # multiclass.py#L290
        # Raw score would mean: scores = conc_name.
        thresh_name = scope.get_unique_variable_name("thresh")
        container.add_initializer(
            thresh_name, proto_dtype, [1, len(op.classes_)], [0.5] * len(op.classes_)
        )
        scores = scope.get_unique_variable_name("threshed")
        apply_sub(scope, [conc_name, thresh_name], scores, container)

        # sign
        signed_input = scope.get_unique_variable_name("signed")
        container.add_node(
            "Sign",
            [scores],
            [signed_input],
            name=scope.get_unique_operator_name("Sign"),
        )
        # clip
        signed_input_cast = scope.get_unique_variable_name("signed_int64")
        apply_cast(
            scope,
            signed_input,
            signed_input_cast,
            container,
            to=onnx_proto.TensorProto.INT64,
        )

        label_name = scope.get_unique_variable_name("label")
        if container.target_opset <= 11:
            abs_name = scope.get_unique_variable_name("abs")
            add_name = scope.get_unique_variable_name("add")
            cst_2 = scope.get_unique_variable_name("cst2")
            container.add_initializer(cst_2, onnx_proto.TensorProto.INT64, [1], [2])
            apply_abs(scope, [signed_input_cast], [abs_name], container)
            apply_add(scope, [signed_input_cast, abs_name], [add_name], container)
            apply_div(scope, [add_name, cst_2], [label_name], container)
        else:
            zero_cst = scope.get_unique_variable_name("zero")
            container.add_initializer(zero_cst, onnx_proto.TensorProto.INT64, [], [0])
            container.add_node(
                "Clip",
                [signed_input_cast, zero_cst],
                [label_name],
                name=scope.get_unique_operator_name("Clip"),
            )
        apply_reshape(
            scope,
            [label_name],
            [operator.outputs[0].full_name],
            container,
            desired_shape=(-1, op.n_classes_),
        )
    else:
        # concatenates outputs
        conc_name = scope.get_unique_variable_name("concatenated")
        apply_concat(scope, probs_names, conc_name, container, axis=1)
        if len(op.estimators_) == 1:
            zeroth_col_name = scope.get_unique_variable_name("zeroth_col")
            merged_prob_name = scope.get_unique_variable_name("merged_prob")
            unit_float_tensor_name = scope.get_unique_variable_name("unit_float_tensor")
            if use_raw_scores:
                container.add_initializer(
                    unit_float_tensor_name, proto_dtype, [], [-1.0]
                )
                apply_mul(
                    scope,
                    [unit_float_tensor_name, conc_name],
                    zeroth_col_name,
                    container,
                    broadcast=1,
                )
            else:
                container.add_initializer(
                    unit_float_tensor_name, proto_dtype, [], [1.0]
                )
                apply_sub(
                    scope,
                    [unit_float_tensor_name, conc_name],
                    zeroth_col_name,
                    container,
                    broadcast=1,
                )
            apply_concat(
                scope, [zeroth_col_name, conc_name], merged_prob_name, container, axis=1
            )
            conc_name = merged_prob_name

        if use_raw_scores:
            apply_identity(scope, conc_name, operator.outputs[1].full_name, container)
        else:
            # normalizes the outputs
            apply_normalization(
                scope, conc_name, operator.outputs[1].full_name, container, axis=1, p=1
            )

        # extracts the labels
        label_name = scope.get_unique_variable_name("label_name")
        container.add_node(
            "ArgMax",
            conc_name,
            label_name,
            name=scope.get_unique_operator_name("ArgMax"),
            axis=1,
        )

        _finalize_converter_classes(
            scope,
            label_name,
            operator.outputs[0].full_name,
            container,
            op.classes_,
            proto_dtype,
        )


def convert_constant_predictor_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *_ConstantPredictor* into *ONNX* format.
    """
    op_version = container.target_opset
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    dtype = {
        onnx_proto.TensorProto.DOUBLE: np.float64,
        onnx_proto.TensorProto.FLOAT: np.float32,
    }
    shape = OnnxShape(operator.inputs[0].full_name, op_version=op_version)
    first = OnnxSlice(
        shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        op_version=op_version,
    )
    y = op.y_.astype(dtype[proto_dtype]).ravel()
    labels = OnnxTile(
        y.astype(np.int64),
        first,
        op_version=op_version,
        output_names=[operator.outputs[0].full_name],
    )

    cst = np.hstack([(1 - y).astype(y.dtype), y])
    proba_flat = OnnxTile(cst, first, op_version=op_version)
    proba_reshape = OnnxReshape(
        proba_flat,
        np.array([-1, 2], dtype=np.int64),
        output_names=[operator.outputs[1].full_name],
        op_version=op_version,
    )

    labels.add_to(scope, container)
    proba_reshape.add_to(scope, container)


register_converter(
    "SklearnOneVsRestClassifier",
    convert_one_vs_rest_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "raw_scores": [True, False],
    },
)

register_converter(
    "Sklearn_ConstantPredictor",
    convert_constant_predictor_classifier,
    options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
)
