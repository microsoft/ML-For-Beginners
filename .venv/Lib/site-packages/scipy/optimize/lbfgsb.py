# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'LbfgsInvHessProduct',
    'LinearOperator',
    'MemoizeJac',
    'OptimizeResult',
    'array',
    'asarray',
    'float64',
    'fmin_l_bfgs_b',
    'old_bound_to_new',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="lbfgsb",
                                   private_modules=["_lbfgsb_py"], all=__all__,
                                   attribute=name)
