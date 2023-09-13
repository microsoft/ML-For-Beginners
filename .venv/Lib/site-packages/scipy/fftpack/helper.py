# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

import warnings
from . import _helper

__all__ = [  # noqa: F822
    'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.fftpack.helper is deprecated and has no attribute "
            f"{name}. Try looking in scipy.fftpack instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.fftpack` namespace, "
                  "the `scipy.fftpack.helper` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_helper, name)
