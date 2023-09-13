"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg._eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""
from .arpack import *
from .lobpcg import *
from ._svds import svds

from . import arpack

__all__ = [
    'ArpackError', 'ArpackNoConvergence',
    'eigs', 'eigsh', 'lobpcg', 'svds'
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
