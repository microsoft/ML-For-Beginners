# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.


import warnings
from . import _fir_filter_design


__all__ = [  # noqa: F822
    'kaiser_beta', 'kaiser_atten', 'kaiserord',
    'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase',
    'ceil', 'log', 'irfft', 'fft', 'ifft', 'sinc', 'toeplitz',
    'hankel', 'solve', 'LinAlgError', 'LinAlgWarning', 'lstsq'

]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.fir_filter_design is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.fir_filter_design` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_fir_filter_design, name)
