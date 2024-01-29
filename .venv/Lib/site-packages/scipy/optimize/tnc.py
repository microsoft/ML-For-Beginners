# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'CONSTANT',
    'FCONVERGED',
    'INFEASIBLE',
    'LOCALMINIMUM',
    'LSFAIL',
    'MAXFUN',
    'MSGS',
    'MSG_ALL',
    'MSG_EXIT',
    'MSG_INFO',
    'MSG_ITER',
    'MSG_NONE',
    'MSG_VERS',
    'MemoizeJac',
    'NOPROGRESS',
    'OptimizeResult',
    'RCSTRINGS',
    'USERABORT',
    'XCONVERGED',
    'array',
    'fmin_tnc',
    'inf',
    'moduleTNC',
    'old_bound_to_new',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="tnc",
                                   private_modules=["_tnc"], all=__all__,
                                   attribute=name)
