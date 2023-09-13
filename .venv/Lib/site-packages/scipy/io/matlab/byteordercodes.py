# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

import warnings
from . import _byteordercodes


__all__ = [  # noqa: F822
    'aliases', 'native_code', 'swapped_code',
    'sys_is_le', 'to_numpy_code'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.io.matlab.byteordercodes is deprecated and has no attribute "
            f"{name}. Try looking in scipy.io.matlab instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.io.matlab` namespace, "
                  "the `scipy.io.matlab.byteordercodes` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_byteordercodes, name)
