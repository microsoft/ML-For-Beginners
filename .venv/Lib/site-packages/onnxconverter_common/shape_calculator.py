# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
Common functions to convert any learner based on trees.
"""

import numbers
import numpy as np
from .data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType
from .utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_linear_classifier_output_shapes(operator):
    '''
    This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
    appear in this operator's output list, we should further generate a map storing all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N, 1], A sequence of map

    Note that the second case is not allowed as long as ZipMap only produces dictionary.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(StringTensorType([1]), FloatTensorType([1]))
            else:
                operator.outputs[1].type = SequenceType(DictionaryType(StringTensorType([]), FloatTensorType([])), N)
        else:
            # For binary LinearSVC, we produce probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[N, 1])
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.target_opset < 7:
                operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
            else:
                operator.outputs[1].type = SequenceType(DictionaryType(Int64TensorType([]), FloatTensorType([])), N)
        else:
            # For binary LinearSVC, we produce probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[N, 1])
    else:
        raise ValueError('Unsupported or mixed label types')


def calculate_linear_regressor_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a batch. If the input batch size is N, the output
    shape may be [N, 1].
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = operator.inputs[0].type.shape[0]
    op = operator.raw_operator
    if hasattr(op, 'n_outputs_'):
        nout = op.n_outputs_
    else:
        nout = 1
    operator.outputs[0].type = FloatTensorType([N, nout])
