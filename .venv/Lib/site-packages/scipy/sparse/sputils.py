# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


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
    return _sub_module_deprecation(sub_package="sparse", module="sputils",
                                   private_modules=["_sputils"], all=__all__,
                                   attribute=name)
