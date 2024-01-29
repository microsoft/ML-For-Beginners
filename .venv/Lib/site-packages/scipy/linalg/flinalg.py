# This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ['get_flinalg_funcs', 'has_column_major_storage']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="flinalg",
                                   private_modules=["_flinalg_py"],
                                   all=__all__, attribute=name)
