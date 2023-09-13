# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

import warnings
from . import _fitpack2


__all__ = [  # noqa: F822
    'BivariateSpline',
    'InterpolatedUnivariateSpline',
    'LSQBivariateSpline',
    'LSQSphereBivariateSpline',
    'LSQUnivariateSpline',
    'RectBivariateSpline',
    'RectSphereBivariateSpline',
    'SmoothBivariateSpline',
    'SmoothSphereBivariateSpline',
    'SphereBivariateSpline',
    'UnivariateSpline',
    'array',
    'concatenate',
    'dfitpack',
    'dfitpack_int',
    'diff',
    'ones',
    'ravel',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.interpolate.fitpack2 is deprecated and has no attribute "
            f"{name}. Try looking in scipy.interpolate instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.interpolate` namespace, "
                  "the `scipy.interpolate.fitpack2` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_fitpack2, name)
