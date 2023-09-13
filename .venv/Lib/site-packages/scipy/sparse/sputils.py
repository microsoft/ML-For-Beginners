# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

import warnings
from . import _sputils


__all__ = [  # noqa: F822
    'asmatrix',
    'check_reshape_kwargs',
    'check_shape',
    'downcast_intp_index',
    'get_index_dtype',
    'get_sum_dtype',
    'getdata',
    'getdtype',
    'is_pydata_spmatrix',
    'isdense',
    'isintlike',
    'ismatrix',
    'isscalarlike',
    'issequence',
    'isshape',
    'matrix',
    'operator',
    'prod',
    'supported_dtypes',
    'sys',
    'to_native',
    'upcast',
    'upcast_char',
    'upcast_scalar',
    'validateaxis',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.sputils is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse` namespace, "
                  "the `scipy.sparse.sputils` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_sputils, name)
