# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

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
    return _sub_module_deprecation(sub_package="signal", module="bsplines",
                                   private_modules=["_bsplines"], all=__all__,
                                   attribute=name)
