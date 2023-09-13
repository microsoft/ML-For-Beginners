# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

import warnings
from . import _interpolate


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
    if name not in __all__:
        raise AttributeError(
            "scipy.interpolate.interpolate is deprecated and has no attribute "
            f"{name}. Try looking in scipy.interpolate instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.interpolate` namespace, "
                  "the `scipy.interpolate.interpolate` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_interpolate, name)
