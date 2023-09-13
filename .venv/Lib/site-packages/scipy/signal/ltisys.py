# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

import warnings
from . import _ltisys

__all__ = [  # noqa: F822
    'lti', 'dlti', 'TransferFunction', 'ZerosPolesGain', 'StateSpace',
    'lsim', 'lsim2', 'impulse', 'impulse2', 'step', 'step2', 'bode',
    'freqresp', 'place_poles', 'dlsim', 'dstep', 'dimpulse',
    'dfreqresp', 'dbode', 's_qr', 'integrate', 'interpolate', 'linalg',
    'interp1d', 'tf2zpk', 'zpk2tf', 'normalize', 'freqs',
    'freqz', 'freqs_zpk', 'freqz_zpk', 'tf2ss', 'abcd_normalize',
    'ss2tf', 'zpk2ss', 'ss2zpk', 'cont2discrete', 'atleast_1d',
    'atleast_2d', 'squeeze', 'transpose', 'zeros_like', 'linspace',
    'nan_to_num', 'LinearTimeInvariant', 'TransferFunctionContinuous',
    'TransferFunctionDiscrete', 'ZerosPolesGainContinuous',
    'ZerosPolesGainDiscrete', 'StateSpaceContinuous',
    'StateSpaceDiscrete', 'Bunch'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.signal.ltisys is deprecated and has no attribute "
            f"{name}. Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.ltisys` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_ltisys, name)
