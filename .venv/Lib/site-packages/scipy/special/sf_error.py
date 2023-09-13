# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

import warnings
from . import _sf_error

__all__ = [  # noqa: F822
    'SpecialFunctionWarning',
    'SpecialFunctionError'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.sf_error is deprecated and has no attribute "
            f"{name}. Try looking in scipy.special instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.special` namespace, "
                  "the `scipy.special.sf_error` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_sf_error, name)
