# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType


def powertransformer_shape_calculator(operator):
    """Shape calculator for PowerTransformer"""
    inputs = operator.inputs[0]
    output = operator.outputs[0]
    n, c = inputs.type.shape
    output.type = FloatTensorType([n, c])


register_shape_calculator("SklearnPowerTransformer", powertransformer_shape_calculator)
