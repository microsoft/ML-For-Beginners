# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import OnnxConcat, OnnxReshapeApi13, OnnxIdentity

try:
    from ..algebra.onnx_ops import OnnxSequenceConstruct
except ImportError:
    # Available since opset 11
    OnnxSequenceConstruct = None
from ..algebra.onnx_operator import OnnxSubEstimator


def convert_multi_output_regressor_converter(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *MultiOutputRegressor* into *ONNX* format.
    """
    op_version = container.target_opset
    op = operator.raw_operator
    inp = operator.inputs[0]
    y_list = [
        OnnxReshapeApi13(
            OnnxSubEstimator(sub, inp, op_version=op_version),
            np.array([-1, 1], dtype=np.int64),
            op_version=op_version,
        )
        for sub in op.estimators_
    ]

    output = OnnxConcat(
        *y_list, axis=1, op_version=op_version, output_names=[operator.outputs[0]]
    )
    output.add_to(scope=scope, container=container)


def convert_multi_output_classifier_converter(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *MultiOutputClassifier* into *ONNX* format.
    """
    if OnnxSequenceConstruct is None:
        raise RuntimeError("This converter requires opset>=11.")
    op_version = container.target_opset
    op_version = container.target_opset
    op = operator.raw_operator
    inp = operator.inputs[0]
    options = scope.get_options(op)
    if options.get("nocl", True):
        options = options.copy()
    else:
        options = {}

    options.update({"zipmap": False})
    y_list = [
        OnnxSubEstimator(sub, inp, op_version=op_version, options=options)
        for sub in op.estimators_
    ]

    # labels
    label_list = [
        OnnxReshapeApi13(y[0], np.array([-1, 1], dtype=np.int64), op_version=op_version)
        for y in y_list
    ]

    # probabilities
    proba_list = [OnnxIdentity(y[1], op_version=op_version) for y in y_list]
    label = OnnxConcat(
        *label_list, axis=1, op_version=op_version, output_names=[operator.outputs[0]]
    )
    label.add_to(scope=scope, container=container)

    proba = OnnxSequenceConstruct(
        *proba_list, op_version=op_version, output_names=[operator.outputs[1]]
    )
    proba.add_to(scope=scope, container=container)


register_converter(
    "SklearnMultiOutputRegressor", convert_multi_output_regressor_converter
)
register_converter(
    "SklearnMultiOutputClassifier",
    convert_multi_output_classifier_converter,
    options={
        "nocl": [False, True],
        "output_class_labels": [False, True],
        "zipmap": [False, True],
    },
)
