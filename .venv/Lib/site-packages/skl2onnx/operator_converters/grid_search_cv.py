# SPDX-License-Identifier: Apache-2.0

from ..common._apply_operation import apply_identity
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_grid_search_cv(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converter for scikit-learn's GridSearchCV.
    """
    opts = scope.get_options(operator.raw_operator)
    grid_search_op = operator.raw_operator
    best_estimator = grid_search_op.best_estimator_
    op_type = sklearn_operator_name_map[type(best_estimator)]
    grid_search_operator = scope.declare_local_operator(op_type, best_estimator)
    container.add_options(id(best_estimator), opts)
    scope.add_options(id(best_estimator), opts)
    grid_search_operator.inputs = operator.inputs

    for i, o in enumerate(operator.outputs):
        v = scope.declare_local_variable(o.onnx_name, type=o.type)
        grid_search_operator.outputs.append(v)
        apply_identity(scope, v.full_name, o.full_name, container)


register_converter(
    "SklearnGridSearchCV", convert_sklearn_grid_search_cv, options="passthrough"
)
