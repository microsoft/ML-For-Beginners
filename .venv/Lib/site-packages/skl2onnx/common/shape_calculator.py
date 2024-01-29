# SPDX-License-Identifier: Apache-2.0

"""
Functions to calculate output shapes of linear classifiers
and regressors.
"""
import numbers
import numpy as np
from .data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from .utils import check_input_and_output_numbers, check_input_and_output_types
from .utils_classifier import get_label_classes


def calculate_linear_classifier_output_shapes(operator):
    """
    This operator maps an input feature vector into a scalar label if
    the number of outputs is one. If two outputs appear in this
    operator's output list, we should further generate a tensor storing
    all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N, 1], A sequence of map

    """
    _calculate_linear_classifier_output_shapes(operator)


def _calculate_linear_classifier_output_shapes(
    operator, decision_path=False, decision_leaf=False, enable_type_checking=True
):
    n_out = 0
    if decision_path:
        n_out += 1
    if decision_leaf:
        n_out += 1
    out_range = [2, 2 + n_out]
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=out_range
    )
    if enable_type_checking:
        check_input_and_output_types(
            operator,
            good_input_types=[
                BooleanTensorType,
                DoubleTensorType,
                FloatTensorType,
                Int64TensorType,
            ],
        )

    _infer_linear_classifier_output_types(operator)


def _infer_linear_classifier_output_types(operator):
    N = operator.inputs[0].get_first_dimension()
    op = operator.raw_operator
    class_labels = get_label_classes(operator.scope_inst, op)

    number_of_classes = len(class_labels)
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        shape = (
            [N, len(op.classes_)]
            if (
                getattr(op, "multilabel_", False)
                or (
                    isinstance(op.classes_, list)
                    and isinstance(op.classes_[0], np.ndarray)
                )
            )
            else [N]
        )
        operator.outputs[0].set_type(StringTensorType(shape=shape))
        if number_of_classes > 2 or operator.type != "SklearnLinearSVC":
            shape = (
                [len(op.classes_), N, max([len(x) for x in op.classes_])]
                if isinstance(op.classes_, list)
                and isinstance(op.classes_[0], np.ndarray)
                else [N, number_of_classes]
            )
            operator.outputs[1].type.shape = shape
        else:
            # For binary LinearSVC, we produce probability of
            # the positive class
            operator.outputs[1].type.shape = [N, 1]
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
        shape = (
            [N, len(op.classes_)]
            if (
                getattr(op, "multilabel_", False)
                or (
                    isinstance(op.classes_, list)
                    and isinstance(op.classes_[0], np.ndarray)
                )
            )
            else [N]
        )
        operator.outputs[0].set_type(Int64TensorType(shape=shape))
        if number_of_classes > 2 or operator.type != "SklearnLinearSVC":
            shape = (
                [len(op.classes_), N, max([len(x) for x in op.classes_])]
                if isinstance(op.classes_, list)
                and isinstance(op.classes_[0], np.ndarray)
                else [N, number_of_classes]
            )
            operator.outputs[1].type.shape = shape
        else:
            # For binary LinearSVC, we produce probability of
            # the positive class
            operator.outputs[1].type.shape = [N, 1]
    else:
        raise ValueError("Label types must be all integers or all strings.")

    # decision_path, decision_leaf
    for n in range(2, len(operator.outputs)):
        operator.outputs[n].type.shape = [N, 1]


def calculate_linear_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a
    batch. If the input batch size is N, the output shape may be
    [N, 1].
    """
    _calculate_linear_regressor_output_shapes(operator)


def _calculate_linear_regressor_output_shapes(operator, enable_type_checking=True):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    if enable_type_checking:
        check_input_and_output_types(
            operator,
            good_input_types=[
                BooleanTensorType,
                DoubleTensorType,
                FloatTensorType,
                Int64TensorType,
            ],
        )

    _infer_linear_regressor_output_types(operator)


def _infer_linear_regressor_output_types(operator):
    inp0 = operator.inputs[0].type
    if isinstance(inp0, (FloatTensorType, DoubleTensorType)):
        cls_type = inp0.__class__
    else:
        cls_type = FloatTensorType

    N = operator.inputs[0].get_first_dimension()
    if (
        hasattr(operator.raw_operator, "coef_")
        and len(operator.raw_operator.coef_.shape) > 1
    ):
        operator.outputs[0].set_type(
            cls_type([N, operator.raw_operator.coef_.shape[0]])
        )
    else:
        operator.outputs[0].set_type(cls_type([N, 1]))

    # decision_path, decision_leaf
    for n in range(1, len(operator.outputs)):
        operator.outputs[n].type.shape = [N, 1]
