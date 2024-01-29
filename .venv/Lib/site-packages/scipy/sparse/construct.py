# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'block_diag',
    'bmat',
    'bsr_matrix',
    'check_random_state',
    'coo_matrix',
    'csc_matrix',
    'csr_hstack',
    'csr_matrix',
    'dia_matrix',
    'diags',
    'eye',
    'get_index_dtype',
    'hstack',
    'identity',
    'isscalarlike',
    'issparse',
    'kron',
    'kronsum',
    'numbers',
    'partial',
    'rand',
    'random',
    'rng_integers',
    'spdiags',
    'upcast',
    'vstack',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="construct",
                                   private_modules=["_construct"], all=__all__,
                                   attribute=name)
