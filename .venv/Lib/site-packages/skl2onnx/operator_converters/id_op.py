# SPDX-License-Identifier: Apache-2.0


from ..common._apply_operation import apply_identity
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_identity(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    apply_identity(
        scope,
        operator.inputs[0].full_name,
        operator.outputs[0].full_name,
        container,
        operator_name=scope.get_unique_operator_name("CIdentity"),
    )


register_converter("SklearnIdentity", convert_sklearn_identity)
