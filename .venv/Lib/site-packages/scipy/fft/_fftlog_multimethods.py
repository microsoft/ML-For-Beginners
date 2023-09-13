'''Multimethods for fast Hankel transforms.
'''

import numpy as np

from ._basic import _dispatch
from ._fftlog import fht as _fht
from ._fftlog import ifht as _ifht
from scipy._lib.uarray import Dispatchable


__all__ = ['fht', 'ifht']


@_dispatch
def fht(a, dln, mu, offset=0.0, bias=0.0):
    """fht multimethod."""
    return (Dispatchable(a, np.ndarray),)


@_dispatch
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    """ifht multimethod."""
    return (Dispatchable(A, np.ndarray),)


# copy over the docstrings
fht.__doc__ = _fht.__doc__
ifht.__doc__ = _ifht.__doc__
