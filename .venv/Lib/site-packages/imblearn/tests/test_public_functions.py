"""This is a copy of sklearn/tests/test_public_functions.py. It can be
removed when we support scikit-learn >= 1.2.
"""
from importlib import import_module
from inspect import signature

import pytest

from imblearn.utils._param_validation import (
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)

PARAM_VALIDATION_FUNCTION_LIST = [
    "imblearn.datasets.fetch_datasets",
    "imblearn.datasets.make_imbalance",
    "imblearn.metrics.classification_report_imbalanced",
    "imblearn.metrics.geometric_mean_score",
    "imblearn.metrics.macro_averaged_mean_absolute_error",
    "imblearn.metrics.make_index_balanced_accuracy",
    "imblearn.metrics.sensitivity_specificity_support",
    "imblearn.metrics.sensitivity_score",
    "imblearn.metrics.specificity_score",
    "imblearn.pipeline.make_pipeline",
]


@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    module_name, func_name = func_module.rsplit(".", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    func_sig = signature(func)
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    # Generate valid values for the required parameters
    # The parameters `*args` and `**kwargs` are ignored since we cannot generate
    # constraints.
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # check that there is a constraint for each parameter
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        assert set(validation_params) == set(func_params), err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    for param_name in func_params:
        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue

        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        # First, check that the error is raised if param doesn't match any valid type.
        with pytest.raises(ValueError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            with pytest.raises(ValueError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
