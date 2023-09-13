# This file is not meant for public use and will be removed in SciPy v2.0.0.

import warnings
from . import _flinalg_py

__all__ = ['get_flinalg_funcs', 'has_column_major_storage']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.linalg.flinalg is deprecated and has no attribute "
            f"{name}. Try looking in scipy.linalg instead.")

    warnings.warn("The `scipy.linalg.flinalg` namespace is deprecated and "
                  "will be removed in SciPy v2.0.0.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_flinalg_py, name)
