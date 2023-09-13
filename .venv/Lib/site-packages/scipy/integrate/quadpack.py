# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.integrate` namespace for importing the functions
# included below.

import warnings
from . import _quadpack_py

__all__ = [  # noqa: F822
    "quad",
    "dblquad",
    "tplquad",
    "nquad",
    "IntegrationWarning",
    "error",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.integrate.quadpack is deprecated and has no attribute "
            f"{name}. Try looking in scipy.integrate instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.integrate` namespace, "
                  "the `scipy.integrate.quadpack` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_quadpack_py, name)
