# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal.windows` namespace for importing the functions
# included below.

import warnings
from . import _windows

__all__ = [  # noqa: F822
    'boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
    'blackmanharris', 'flattop', 'bartlett', 'barthann',
    'hamming', 'kaiser', 'gaussian', 'general_cosine',
    'general_gaussian', 'general_hamming', 'chebwin', 'cosine',
    'hann', 'exponential', 'tukey', 'taylor', 'dpss', 'get_window',
    'linalg', 'sp_fft', 'k', 'v', 'key'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.windows.windows is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal.windows instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal.windows` namespace, "
                  "the `scipy.signal.windows.windows` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_windows, name)
