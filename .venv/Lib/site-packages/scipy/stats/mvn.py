# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

import warnings
from . import _mvn  # type: ignore


__all__ = [  # noqa: F822
    'mvnun',
    'mvnun_weighted',
    'mvndst',
    'dkblck'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.stats.mvn is deprecated and has no attribute "
            f"{name}. Try looking in scipy.stats instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.stats` namespace, "
                  "the `scipy.stats.mvn` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_mvn, name)
