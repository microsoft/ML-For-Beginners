# This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ['lsoda']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="lsoda",
                                   private_modules=["_lsoda"], all=__all__,
                                   attribute=name)
