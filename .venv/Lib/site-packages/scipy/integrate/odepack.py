# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.integrate` namespace for importing the functions
# included below.

import warnings
from . import _odepack_py

__all__ = ['odeint', 'ODEintWarning']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.integrate.odepack is deprecated and has no attribute "
            f"{name}. Try looking in scipy.integrate instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.integrate` namespace, "
                  "the `scipy.integrate.odepack` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_odepack_py, name)
