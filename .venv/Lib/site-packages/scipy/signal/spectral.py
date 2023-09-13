# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.


import warnings
from . import _spectral_py


__all__ = [  # noqa: F822
    'periodogram', 'welch', 'lombscargle', 'csd', 'coherence',
    'spectrogram', 'stft', 'istft', 'check_COLA', 'check_NOLA',
    'sp_fft', 'get_window', 'const_ext', 'even_ext',
    'odd_ext', 'zero_ext'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.spectral is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.spectral` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_spectral_py, name)
