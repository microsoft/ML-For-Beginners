# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BPoly',
    'BSpline',
    'NdPPoly',
    'PPoly',
    'RectBivariateSpline',
    'RegularGridInterpolator',
    'array',
    'asarray',
    'atleast_1d',
    'atleast_2d',
    'comb',
    'dfitpack',
    'interp1d',
    'interp2d',
    'interpn',
    'intp',
    'itertools',
    'lagrange',
    'make_interp_spline',
    'poly1d',
    'prod',
    'ravel',
    'searchsorted',
    'spec',
    'transpose',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="interpolate",
                                   private_modules=["_interpolate", "fitpack2", "_rgi"],
                                   all=__all__, attribute=name)
