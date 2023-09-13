# This file is not meant for public use and will be removed in SciPy v2.0.0.


import warnings
from . import _vode  # type: ignore


__all__ = [  # noqa: F822
    'dvode',
    'zvode'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.integrate.vode is deprecated and has no attribute "
            f"{name}.")

    warnings.warn("The `scipy.integrate.vode` namespace is deprecated "
                  "and will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_vode, name)
