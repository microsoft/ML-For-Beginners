# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'airyzo',
    'bernob',
    'cerzo',
    'clpmn',
    'clpn',
    'clqmn',
    'clqn',
    'cpbdn',
    'cyzo',
    'eulerb',
    'fcoef',
    'fcszo',
    'jdzo',
    'jyzo',
    'klvnzo',
    'lamn',
    'lamv',
    'lpmn',
    'lpn',
    'lqmn',
    'lqnb',
    'pbdv',
    'rctj',
    'rcty',
    'segv'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="specfun",
                                   private_modules=["_specfun"], all=__all__,
                                   attribute=name)
