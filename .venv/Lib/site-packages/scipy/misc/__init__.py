"""
==========================================
Miscellaneous routines (:mod:`scipy.misc`)
==========================================

.. currentmodule:: scipy.misc

.. deprecated:: 1.10.0

   This module is deprecated and will be completely
   removed in SciPy v2.0.0.

Various utilities that don't have another home.

.. autosummary::
   :toctree: generated/

   ascent - Get example image for processing
   central_diff_weights - Weights for an n-point central mth derivative
   derivative - Find the nth derivative of a function at a point
   face - Get example image for processing
   electrocardiogram - Load an example of a 1-D signal

"""


from ._common import *
from . import _common
import warnings

# Deprecated namespaces, to be removed in v2.0.0
from . import common, doccer

__all__ = _common.__all__

dataset_methods = ['ascent', 'face', 'electrocardiogram']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.misc is deprecated and has no attribute "
            f"{name}.")

    if name in dataset_methods:
        msg = ("The module `scipy.misc` is deprecated and will be "
               "completely removed in SciPy v2.0.0. "
               f"All dataset methods including {name}, must be imported "
               "directly from the new `scipy.datasets` module.")
    else:
        msg = (f"The method `{name}` from the `scipy.misc` namespace is"
               " deprecated, and will be removed in SciPy v1.12.0.")

    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)

    return getattr(name)


del _common

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
