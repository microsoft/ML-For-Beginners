# This file is not meant for public use and will be removed in SciPy v2.0.0.

import warnings
from . import _add_newdocs

__all__ = ['get', 'add_newdoc', 'Dict', 'docdict']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.special.add_newdocs is deprecated and has no attribute "
            f"{name}.")

    warnings.warn("The `scipy.special.add_newdocs` namespace is deprecated."
                  " and will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_add_newdocs, name)
