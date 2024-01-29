# SPDX-License-Identifier: Apache-2.0


from onnx.helper import make_tensor
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_mul
from ..common.utils_classifier import _finalize_converter_classes
from ..common.data_types import guess_proto_type, Int64TensorType
from .._supported_operators import sklearn_operator_name_map
from ..proto import onnx_proto


def convert_voting_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *VotingClassifier* into *ONNX* format.

    *predict_proba* is not defined by *scikit-learn* when ``voting='hard'``.
    The converted model still defines a probability vector equal to the
    highest probability obtained for each class over all estimators.

    *scikit-learn* enables both modes, transformer and predictor
    for the voting classifier. *ONNX* does not make this
    distinction and always creates two outputs, labels
    and probabilities.
    """
    if scope.get_options(operator.raw_operator, dict(nocl=False))["nocl"]:
        raise RuntimeError(
            "Option 'nocl' is not implemented for operator '{}'.".format(
                operator.raw_operator.__class__.__name__
            )
        )
    proto_dtype = guess_proto_type(operator.outputs[1].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    op = operator.raw_operator
    n_classes = len(op.classes_)

    classes_ind_name = scope.get_unique_variable_name("classes_ind")
    container.add_initializer(
        classes_ind_name,
        onnx_proto.TensorProto.INT64,
        (1, n_classes),
        list(range(n_classes)),
    )

    probs_names = []
    one_name = None
    for i, estimator in enumerate(op.estimators_):
        if estimator is None:
            continue

        op_type = sklearn_operator_name_map[type(estimator)]

        this_operator = scope.declare_local_operator(op_type, estimator)
        this_operator.inputs = operator.inputs

        label_name = scope.declare_local_variable("label_%d" % i, Int64TensorType())
        prob_name = scope.declare_local_variable(
            "voting_proba_%d" % i, operator.outputs[1].type.__class__()
        )
        this_operator.outputs.append(label_name)
        this_operator.outputs.append(prob_name)

        if op.voting == "hard":
            if one_name is None:
                shape_name = scope.get_unique_variable_name("shape")
                container.add_node(
                    "Shape",
                    prob_name.onnx_name,
                    shape_name,
                    name=scope.get_unique_operator_name("Shape"),
                )
                zero_name = scope.get_unique_variable_name("zero")
                container.add_node(
                    "ConstantOfShape",
                    shape_name,
                    zero_name,
                    name=scope.get_unique_operator_name("CoSA"),
                    value=make_tensor("value", proto_dtype, (1,), [0.0]),
                    op_version=9,
                )
                one_name = scope.get_unique_variable_name("one")
                container.add_node(
                    "ConstantOfShape",
                    shape_name,
                    one_name,
                    name=scope.get_unique_operator_name("CoSB"),
                    value=make_tensor("value", proto_dtype, (1,), [1.0]),
                    op_version=9,
                )

            argmax_output_name = scope.get_unique_variable_name("argmax_output")
            container.add_node(
                "ArgMax",
                prob_name.onnx_name,
                argmax_output_name,
                name=scope.get_unique_operator_name("ArgMax"),
                axis=1,
            )

            equal_name = scope.get_unique_variable_name("equal")
            container.add_node(
                "Equal",
                [argmax_output_name, classes_ind_name],
                equal_name,
                name=scope.get_unique_operator_name("Equal"),
            )

            max_proba_name = scope.get_unique_variable_name("probsmax")
            container.add_node(
                "Where",
                [equal_name, one_name, zero_name],
                max_proba_name,
                name=scope.get_unique_operator_name("Where"),
            )
            prob_name = max_proba_name
        else:
            prob_name = prob_name.onnx_name

        if op.weights is not None:
            val = op.weights[i] / op.weights.sum()
        else:
            val = 1.0 / len(op.estimators_)

        weights_name = scope.get_unique_variable_name("w%d" % i)
        container.add_initializer(weights_name, proto_dtype, [1], [val])
        wprob_name = scope.get_unique_variable_name("wprob_name")
        apply_mul(scope, [prob_name, weights_name], wprob_name, container, broadcast=1)
        probs_names.append(wprob_name)

    if op.flatten_transform in (False, None):
        container.add_node(
            "Sum",
            probs_names,
            operator.outputs[1].full_name,
            name=scope.get_unique_operator_name("Sum"),
        )
    else:
        raise NotImplementedError(
            "flatten_transform==True is not implemented yet. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )

    # labels
    label_name = scope.get_unique_variable_name("label_name")
    container.add_node(
        "ArgMax",
        operator.outputs[1].full_name,
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


register_converter(
    "SklearnVotingClassifier",
    convert_voting_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
