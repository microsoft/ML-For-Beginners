# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _optimize


__all__ = [  # noqa: F822
    'Brent',
    'FD_METHODS',
    'Inf',
    'LineSearchWarning',
    'MapWrapper',
    'MemoizeJac',
    'OptimizeResult',
    'OptimizeWarning',
    'ScalarFunction',
    'approx_derivative',
    'approx_fhess_p',
    'approx_fprime',
    'argmin',
    'asarray',
    'asfarray',
    'atleast_1d',
    'bracket',
    'brent',
    'brute',
    'check_grad',
    'check_random_state',
    'eye',
    'fmin',
    'fmin_bfgs',
    'fmin_cg',
    'fmin_ncg',
    'fmin_powell',
    'fminbound',
    'golden',
    'is_array_scalar',
    'line_search',
    'line_search_wolfe1',
    'line_search_wolfe2',
    'main',
    'rosen',
    'rosen_der',
    'rosen_hess',
    'rosen_hess_prod',
    'shape',
    'show_options',
    'sqrt',
    'squeeze',
    'sys',
    'vecnorm',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.optimize is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.optimize` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_optimize, name)
