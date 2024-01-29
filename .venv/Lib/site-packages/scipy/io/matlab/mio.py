# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.matlab` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'mat_reader_factory', 'loadmat', 'savemat', 'whosmat',
    'contextmanager', 'docfiller',
    'MatFile4Reader', 'MatFile4Writer', 'MatFile5Reader', 'MatFile5Writer'
]

def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.matlab", module="mio",
                                   private_modules=["_mio"], all=__all__,
                                   attribute=name)
