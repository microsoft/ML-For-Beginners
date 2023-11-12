# SPDX-License-Identifier: Apache-2.0


from .._supported_operators import sklearn_operator_name_map
from ..common._apply_operation import apply_identity
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_ransac_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for RANSACRegressor.
    """
    ransac_op = operator.raw_operator
    op_type = sklearn_operator_name_map[type(ransac_op.estimator_)]
    this_operator = scope.declare_local_operator(op_type, ransac_op.estimator_)
    this_operator.inputs = operator.inputs
    label_name = scope.declare_local_variable(
        "label", operator.inputs[0].type.__class__()
    )
    this_operator.outputs.append(label_name)
    apply_identity(
        scope, label_name.full_name, operator.outputs[0].full_name, container
    )


register_converter("SklearnRANSACRegressor", convert_sklearn_ransac_regressor)
