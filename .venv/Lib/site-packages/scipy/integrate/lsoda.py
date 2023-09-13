# This file is not meant for public use and will be removed in SciPy v2.0.0.


import warnings
from . import _lsoda  # type: ignore


__all__ = ['lsoda']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.integrate.lsoda is deprecated and has no attribute "
            f"{name}.")

    warnings.warn("The `scipy.integrate.lsoda` namespace is deprecated "
                  "and will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_lsoda, name)
