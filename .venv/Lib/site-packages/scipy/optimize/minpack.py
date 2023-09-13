# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _minpack_py


__all__ = [  # noqa: F822
    'LEASTSQ_FAILURE',
    'LEASTSQ_SUCCESS',
    'LinAlgError',
    'OptimizeResult',
    'OptimizeWarning',
    'asarray',
    'atleast_1d',
    'check_gradient',
    'cholesky',
    'curve_fit',
    'dot',
    'dtype',
    'error',
    'eye',
    'finfo',
    'fixed_point',
    'fsolve',
    'greater',
    'inexact',
    'inf',
    'inv',
    'issubdtype',
    'least_squares',
    'leastsq',
    'prepare_bounds',
    'prod',
    'shape',
    'solve_triangular',
    'svd',
    'take',
    'transpose',
    'triu',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.minpack is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.minpack` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_minpack_py, name)
