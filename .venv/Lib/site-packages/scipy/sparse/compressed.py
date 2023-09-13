# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

import warnings
from . import _compressed


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
    'get_index_dtype',
    'get_sum_dtype',
    'getdtype',
    'is_pydata_spmatrix',
    'isdense',
    'isintlike',
    'isscalarlike',
    'isshape',
    'isspmatrix',
    'operator',
    'spmatrix',
    'to_native',
    'upcast',
    'upcast_char',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.sparse.compressed is deprecated and has no attribute "
            f"{name}. Try looking in scipy.sparse instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.sparse` namespace, "
                  "the `scipy.sparse.compressed` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_compressed, name)
