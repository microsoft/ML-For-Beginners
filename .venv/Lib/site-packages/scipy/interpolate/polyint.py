# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BarycentricInterpolator',
    'KroghInterpolator',
    'approximate_taylor_polynomial',
    'barycentric_interpolate',
    'factorial',
    'float_factorial',
    'krogh_interpolate',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="polyint",
                                   private_modules=["_polyint"], all=__all__,
                                   attribute=name)
