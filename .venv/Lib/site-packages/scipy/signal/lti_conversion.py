# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
    'cont2discrete','eye', 'atleast_2d',
    'poly', 'prod', 'array', 'outer', 'linalg', 'tf2zpk', 'zpk2tf', 'normalize'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="lti_conversion",
                                   private_modules=["_lti_conversion"], all=__all__,
                                   attribute=name)
