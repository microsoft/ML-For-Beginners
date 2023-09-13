# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.odr` namespace for importing the functions
# included below.

import warnings
from . import _models

__all__ = [  # noqa: F822
    'Model', 'exponential', 'multilinear', 'unilinear',
    'quadratic', 'polynomial'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.odr.models is deprecated and has no attribute "
            f"{name}. Try looking in scipy.odr instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.odr` namespace, "
                  "the `scipy.odr.models` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_models, name)
