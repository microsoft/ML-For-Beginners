# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common.data_types import guess_numpy_type
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import OnnxMatMul


def convert_random_projection(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """Converter for PowerTransformer"""
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    op = operator.raw_operator
    opv = container.target_opset
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32

    y = OnnxMatMul(
        op_in, op.components_.T.astype(dtype), op_version=opv, output_names=[op_out]
    )
    y.add_to(scope, container)


register_converter("SklearnGaussianRandomProjection", convert_random_projection)
