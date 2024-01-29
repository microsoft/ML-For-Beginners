# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'append',
    'approx_derivative',
    'approx_jacobian',
    'array',
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
    return _sub_module_deprecation(sub_package="optimize", module="slsqp",
                                   private_modules=["_slsqp_py"], all=__all__,
                                   attribute=name)
