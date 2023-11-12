# SPDX-License-Identifier: Apache-2.0

import logging
from ..common._registration import register_shape_calculator, get_shape_calculator
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_grid_search_cv(operator):
    grid_search_op = operator.raw_operator
    best_estimator = grid_search_op.best_estimator_
    name = sklearn_operator_name_map.get(type(best_estimator), None)
    if name is None:
        logger = logging.getLogger("skl2onnx")
        logger.warn(
            "[convert_sklearn_grid_search_cv] failed to find alias "
            "to model type %r.",
            type(best_estimator),
        )
        return
    op = operator.new_raw_operator(best_estimator, name)
    shape_calc = get_shape_calculator(name)
    shape_calc(op)
    operator.outputs = op.outputs


register_shape_calculator("SklearnGridSearchCV", convert_sklearn_grid_search_cv)
