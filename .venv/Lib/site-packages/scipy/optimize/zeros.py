# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'CONVERGED',
    'CONVERR',
    'INPROGRESS',
    'RootResults',
    'SIGNERR',
    'TOMS748Solver',
    'VALUEERR',
    'bisect',
    'brenth',
    'brentq',
    'flag_map',
    'namedtuple',
    'newton',
    'operator',
    'results_c',
    'ridder',
    'toms748',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="zeros",
                                   private_modules=["_zeros_py"], all=__all__,
                                   attribute=name)
