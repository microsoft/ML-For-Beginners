"""test sparse matrix construction functions"""

from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, csr_array, sparray

import numpy as np
from scipy.sparse import _extract


class TestExtract:
    def setup_method(self):
        self.cases = [
            csr_array([[1,2]]),
            csr_array([[1,0]]),
            csr_array([[0,0]]),
            csr_array([[1],[2]]),
            csr_array([[1],[0]]),
            csr_array([[0],[0]]),
            csr_array([[1,2],[3,4]]),
            csr_array([[0,1],[0,0]]),
            csr_array([[0,0],[1,0]]),
            csr_array([[0,0],[0,0]]),
            csr_array([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]),
            csr_array([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]).T,
        ]

    def test_find(self):
        for A in self.cases:
            I,J,V = _extract.find(A)
            B = csr_array((V,(I,J)), shape=A.shape)
            assert_equal(A.toarray(), B.toarray())

    def test_tril(self):
        for A in self.cases:
            B = A.toarray()
            for k in [-3,-2,-1,0,1,2,3]:
                assert_equal(_extract.tril(A,k=k).toarray(), np.tril(B,k=k))

    def test_triu(self):
        for A in self.cases:
            B = A.toarray()
            for k in [-3,-2,-1,0,1,2,3]:
                assert_equal(_extract.triu(A,k=k).toarray(), np.triu(B,k=k))

    def test_array_vs_matrix(self):
        for A in self.cases:
            assert isinstance(_extract.tril(A), sparray)
            assert isinstance(_extract.triu(A), sparray)
            M = csr_matrix(A)
            assert not isinstance(_extract.tril(M), sparray)
            assert not isinstance(_extract.triu(M), sparray)
