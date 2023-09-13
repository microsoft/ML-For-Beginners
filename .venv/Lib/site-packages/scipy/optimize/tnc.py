# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _tnc


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
    'asfarray',
    'fmin_tnc',
    'inf',
    'moduleTNC',
    'old_bound_to_new',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.tnc is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.tnc` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_tnc, name)
