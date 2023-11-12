# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_operator import OnnxSubEstimator
from ..algebra.onnx_ops import OnnxIdentity, OnnxConcat, OnnxReshape


def convert_sklearn_random_tree_embedding(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator
    opv = container.target_opset

    if op.sparse_output:
        raise RuntimeError(
            "The converter cannot convert the model with sparse outputs."
        )

    outputs = []
    for est in op.estimators_:
        leave = OnnxSubEstimator(
            est, X, op_version=opv, options={"decision_leaf": True}
        )
        outputs.append(
            OnnxReshape(leave[1], np.array([-1, 1], dtype=np.int64), op_version=opv)
        )
    merged = OnnxConcat(*outputs, axis=1, op_version=opv)
    ohe = OnnxSubEstimator(op.one_hot_encoder_, merged, op_version=opv)
    y = OnnxIdentity(ohe, op_version=opv, output_names=out)
    y.add_to(scope, container)


register_converter("SklearnRandomTreesEmbedding", convert_sklearn_random_tree_embedding)
