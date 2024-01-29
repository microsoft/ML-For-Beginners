# This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ['get', 'add_newdoc', 'docdict']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="add_newdocs",
                                   private_modules=["_add_newdocs"], all=__all__,
                                   attribute=name)
