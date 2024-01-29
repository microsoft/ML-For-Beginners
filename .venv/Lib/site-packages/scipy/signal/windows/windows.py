# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal.windows` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

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
    return _sub_module_deprecation(sub_package="signal.windows", module="windows",
                                   private_modules=["_windows"], all=__all__,
                                   attribute=name)
