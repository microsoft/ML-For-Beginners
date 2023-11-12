# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator


def calculate_sklearn_ovr_decision_function(operator):
    N = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = operator.inputs[0].type.__class__(
        [N, len(operator.raw_operator.classes_)]
    )


register_shape_calculator(
    "SklearnOVRDecisionFunction", calculate_sklearn_ovr_decision_function
)
