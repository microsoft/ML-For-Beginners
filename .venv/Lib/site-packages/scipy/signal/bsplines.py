# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

import warnings
from . import _bsplines

__all__ = [  # noqa: F822
    'spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic',
    'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval',
    'logical_and', 'zeros_like', 'piecewise', 'array', 'arctan2',
    'tan', 'arange', 'floor', 'exp', 'greater', 'less', 'add',
    'less_equal', 'greater_equal', 'cspline2d', 'sepfir2d', 'comb',
    'float_factorial'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.bsplines is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.bsplines` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_bsplines, name)
