# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'LineSearchWarning',
    'line_search',
    'line_search_BFGS',
    'line_search_armijo',
    'line_search_wolfe1',
    'line_search_wolfe2',
    'minpack2',
    'scalar_search_armijo',
    'scalar_search_wolfe1',
    'scalar_search_wolfe2',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="linesearch",
                                   private_modules=["_linesearch"], all=__all__,
                                   attribute=name)
