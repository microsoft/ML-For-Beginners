# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_flatten(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    name = scope.get_unique_operator_name("Flatten")
    container.add_node(
        "Flatten",
        operator.inputs[0].full_name,
        operator.outputs[0].full_name,
        name=name,
        axis=1,
    )


register_converter("SklearnFlatten", convert_sklearn_flatten)
