"""
MATLAB® file utilities (:mod:`scipy.io.matlab`)
===============================================

.. currentmodule:: scipy.io.matlab

This submodule is meant to provide lower-level file utilities related to reading
and writing MATLAB files.

.. autosummary::
   :toctree: generated/

   matfile_version - Get the MATLAB file version
   MatReadError - Exception indicating a read issue
   MatReadWarning - Warning class for read issues
   MatWriteError - Exception indicating a write issue
   mat_struct - Class used when ``struct_as_record=False``

.. autosummary::
   :toctree: generated/
   :template: autosummary/ndarray_subclass.rst
   :nosignatures:

   MatlabObject - Class for a MATLAB object
   MatlabOpaque - Class for a MATLAB opaque matrix
   MatlabFunction - Class for a MATLAB function object

The following utilities that live in the :mod:`scipy.io`
namespace also exist in this namespace:

.. autosummary::
   :toctree: generated/

   loadmat - Read a MATLAB style mat file (version 4 through 7.1)
   savemat - Write a MATLAB style mat file (version 4 through 7.1)
   whosmat - List contents of a MATLAB style mat file (version 4 through 7.1)

Notes
-----
MATLAB(R) is a registered trademark of The MathWorks, Inc., 3 Apple Hill
Drive, Natick, MA 01760-2098, USA.

"""
# Matlab file read and write utilities
from ._mio import loadmat, savemat, whosmat
from ._mio5 import MatlabFunction
from ._mio5_params import MatlabObject, MatlabOpaque, mat_struct
from ._miobase import (matfile_version, MatReadError, MatReadWarning,
                      MatWriteError)

# Deprecated namespaces, to be removed in v2.0.0
from .import (mio, mio5, mio5_params, mio4, byteordercodes,
            miobase, mio_utils, streams, mio5_utils)

__all__ = [
    'loadmat', 'savemat', 'whosmat', 'MatlabObject',
    'matfile_version', 'MatReadError', 'MatReadWarning',
    'MatWriteError', 'mat_struct', 'MatlabOpaque', 'MatlabFunction'
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
