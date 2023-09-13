# This file is not meant for public use and will be removed in SciPy v2.0.0.


import warnings
from . import _dop  # type: ignore


__all__ = [  # noqa: F822
    'dopri5',
    'dop853'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.integrate.dop is deprecated and has no attribute "
            f"{name}")

    warnings.warn("The `scipy.integrate.dop` namespace is deprecated "
                  "and will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_dop, name)
