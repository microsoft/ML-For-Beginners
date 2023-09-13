# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

import warnings
from . import _filter_design

__all__ = [  # noqa: F822
    'findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
    'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
    'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
    'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
    'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
    'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
    'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
    'sosfreqz', 'iirnotch', 'iirpeak', 'bilinear_zpk',
    'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
    'gammatone', 'iircomb',
    'atleast_1d', 'poly', 'polyval', 'roots', 'resize', 'absolute',
    'logspace', 'tan', 'log10', 'arctan', 'arcsinh', 'exp', 'arccosh',
    'ceil', 'conjugate', 'append', 'prod', 'full', 'array', 'mintypecode',
    'npp_polyval', 'polyvalfromroots', 'optimize', 'sp_fft', 'comb',
    'float_factorial', 'abs', 'maxflat', 'yulewalk',
    'EPSILON', 'filter_dict', 'band_dict', 'bessel_norms'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.filter_design is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.filter_design` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_filter_design, name)
