# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _slsqp_py


__all__ = [  # noqa: F822
    'OptimizeResult',
    'append',
    'approx_derivative',
    'approx_jacobian',
    'array',
    'asfarray',
    'atleast_1d',
    'concatenate',
    'exp',
    'finfo',
    'fmin_slsqp',
    'inf',
    'isfinite',
    'linalg',
    'old_bound_to_new',
    'slsqp',
    'sqrt',
    'vstack',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.slsqp is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.slsqp` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_slsqp_py, name)
