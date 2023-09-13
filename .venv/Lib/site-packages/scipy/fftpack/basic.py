# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

import warnings
from . import _basic

__all__ = [  # noqa: F822
    'fft','ifft','fftn','ifftn','rfft','irfft',
    'fft2','ifft2'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.fftpack.basic is deprecated and has no attribute "
            f"{name}. Try looking in scipy.fftpack instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.fftpack` namespace, "
                  "the `scipy.fftpack.basic` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_basic, name)
