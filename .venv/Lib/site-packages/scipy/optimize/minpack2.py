# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

import warnings
from . import _minpack2


__all__ = [  # noqa: F822
    'dcsrch',
    'dcstep',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.optimize.minpack2 is deprecated and has no attribute "
            f"{name}. Try looking in scipy.optimize instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.optimize` namespace, "
                  "the `scipy.optimize.minpack2` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_minpack2, name)
