# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'kaiser_beta', 'kaiser_atten', 'kaiserord',
    'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase',
    'ceil', 'log', 'irfft', 'fft', 'ifft', 'sinc', 'toeplitz',
    'hankel', 'solve', 'LinAlgError', 'LinAlgWarning', 'lstsq'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="fir_filter_design",
                                   private_modules=["_fir_filter_design"], all=__all__,
                                   attribute=name)
