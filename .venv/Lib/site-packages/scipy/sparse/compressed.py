# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'IndexMixin',
    'SparseEfficiencyWarning',
    'check_shape',
    'csr_column_index1',
    'csr_column_index2',
    'csr_row_index',
    'csr_row_slice',
    'csr_sample_offsets',
    'csr_sample_values',
    'csr_todense',
    'downcast_intp_index',
    'get_csr_submatrix',
    'get_sum_dtype',
    'getdtype',
    'is_pydata_spmatrix',
    'isdense',
    'isintlike',
    'isscalarlike',
    'isshape',
    'operator',
    'to_native',
    'upcast',
    'upcast_char',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="compressed",
                                   private_modules=["_compressed"], all=__all__,
                                   attribute=name)
