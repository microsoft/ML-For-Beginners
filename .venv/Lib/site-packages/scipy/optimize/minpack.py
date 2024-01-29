# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


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
    return _sub_module_deprecation(sub_package="optimize", module="minpack",
                                   private_modules=["_minpack_py"], all=__all__,
                                   attribute=name)
