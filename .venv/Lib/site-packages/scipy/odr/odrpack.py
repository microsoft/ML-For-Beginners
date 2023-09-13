# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.odr` namespace for importing the functions
# included below.

import warnings
from . import _odrpack

__all__ = [  # noqa: F822
    'odr', 'OdrWarning', 'OdrError', 'OdrStop',
    'Data', 'RealData', 'Model', 'Output', 'ODR',
    'odr_error', 'odr_stop'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.odr.odrpack is deprecated and has no attribute "
            f"{name}. Try looking in scipy.odr instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.odr` namespace, "
                  "the `scipy.odr.odrpack` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_odrpack, name)
