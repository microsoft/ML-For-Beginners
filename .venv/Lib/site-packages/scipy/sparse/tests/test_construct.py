"""test sparse matrix construction functions"""

import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
        assert_array_equal, assert_array_almost_equal_nulp)
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state

from scipy.sparse import (csr_matrix, coo_matrix,
                          csr_array, coo_array,
                          sparray, spmatrix,
                          _construct as construct)
from scipy.sparse._construct import rand as sprand

sparse_formats = ['csr','csc','coo','bsr','dia','lil','dok']

#TODO check whether format=XXX is respected


def _sprandn(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    # Helper function for testing.
    random_state = check_random_state(random_state)
    data_rvs = random_state.standard_normal
    return construct.random(m, n, density, format, dtype,
                            random_state, data_rvs)


def _sprandn_array(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    # Helper function for testing.
    random_state = check_random_state(random_state)
    data_sampler = random_state.standard_normal
    return construct.random_array((m, n), density=density, format=format, dtype=dtype,
                                  random_state=random_state, data_sampler=data_sampler)


class TestConstructUtils:
    def test_spdiags(self):
        diags1 = array([[1, 2, 3, 4, 5]])
        diags2 = array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9,10]])
        diags3 = array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9,10],
                         [11,12,13,14,15]])

        cases = []
        cases.append((diags1, 0, 1, 1, [[1]]))
        cases.append((diags1, [0], 1, 1, [[1]]))
        cases.append((diags1, [0], 2, 1, [[1],[0]]))
        cases.append((diags1, [0], 1, 2, [[1,0]]))
        cases.append((diags1, [1], 1, 2, [[0,2]]))
        cases.append((diags1,[-1], 1, 2, [[0,0]]))
        cases.append((diags1, [0], 2, 2, [[1,0],[0,2]]))
        cases.append((diags1,[-1], 2, 2, [[0,0],[1,0]]))
        cases.append((diags1, [3], 2, 2, [[0,0],[0,0]]))
        cases.append((diags1, [0], 3, 4, [[1,0,0,0],[0,2,0,0],[0,0,3,0]]))
        cases.append((diags1, [1], 3, 4, [[0,2,0,0],[0,0,3,0],[0,0,0,4]]))
        cases.append((diags1, [2], 3, 5, [[0,0,3,0,0],[0,0,0,4,0],[0,0,0,0,5]]))

        cases.append((diags2, [0,2], 3, 3, [[1,0,8],[0,2,0],[0,0,3]]))
        cases.append((diags2, [-1,0], 3, 4, [[6,0,0,0],[1,7,0,0],[0,2,8,0]]))
        cases.append((diags2, [2,-3], 6, 6, [[0,0,3,0,0,0],
                                              [0,0,0,4,0,0],
                                              [0,0,0,0,5,0],
                                              [6,0,0,0,0,0],
                                              [0,7,0,0,0,0],
                                              [0,0,8,0,0,0]]))

        cases.append((diags3, [-1,0,1], 6, 6, [[6,12, 0, 0, 0, 0],
                                                [1, 7,13, 0, 0, 0],
                                                [0, 2, 8,14, 0, 0],
                                                [0, 0, 3, 9,15, 0],
                                                [0, 0, 0, 4,10, 0],
                                                [0, 0, 0, 0, 5, 0]]))
        cases.append((diags3, [-4,2,-1], 6, 5, [[0, 0, 8, 0, 0],
                                                 [11, 0, 0, 9, 0],
                                                 [0,12, 0, 0,10],
                                                 [0, 0,13, 0, 0],
                                                 [1, 0, 0,14, 0],
                                                 [0, 2, 0, 0,15]]))
        cases.append((diags3, [-1, 1, 2], len(diags3[0]), len(diags3[0]),
                      [[0, 7, 13, 0, 0],
                       [1, 0, 8, 14, 0],
                       [0, 2, 0, 9, 15],
                       [0, 0, 3, 0, 10],
                       [0, 0, 0, 4, 0]]))

        for d, o, m, n, result in cases:
            if len(d[0]) == m and m == n:
                assert_equal(construct.spdiags(d, o).toarray(), result)
            assert_equal(construct.spdiags(d, o, m, n).toarray(), result)
            assert_equal(construct.spdiags(d, o, (m, n)).toarray(), result)

    def test_diags(self):
        a = array([1, 2, 3, 4, 5])
        b = array([6, 7, 8, 9, 10])
        c = array([11, 12, 13, 14, 15])

        cases = []
        cases.append((a[:1], 0, (1, 1), [[1]]))
        cases.append(([a[:1]], [0], (1, 1), [[1]]))
        cases.append(([a[:1]], [0], (2, 1), [[1],[0]]))
        cases.append(([a[:1]], [0], (1, 2), [[1,0]]))
        cases.append(([a[:1]], [1], (1, 2), [[0,1]]))
        cases.append(([a[:2]], [0], (2, 2), [[1,0],[0,2]]))
        cases.append(([a[:1]],[-1], (2, 2), [[0,0],[1,0]]))
        cases.append(([a[:3]], [0], (3, 4), [[1,0,0,0],[0,2,0,0],[0,0,3,0]]))
        cases.append(([a[:3]], [1], (3, 4), [[0,1,0,0],[0,0,2,0],[0,0,0,3]]))
        cases.append(([a[:1]], [-2], (3, 5), [[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0]]))
        cases.append(([a[:2]], [-1], (3, 5), [[0,0,0,0,0],[1,0,0,0,0],[0,2,0,0,0]]))
        cases.append(([a[:3]], [0], (3, 5), [[1,0,0,0,0],[0,2,0,0,0],[0,0,3,0,0]]))
        cases.append(([a[:3]], [1], (3, 5), [[0,1,0,0,0],[0,0,2,0,0],[0,0,0,3,0]]))
        cases.append(([a[:3]], [2], (3, 5), [[0,0,1,0,0],[0,0,0,2,0],[0,0,0,0,3]]))
        cases.append(([a[:2]], [3], (3, 5), [[0,0,0,1,0],[0,0,0,0,2],[0,0,0,0,0]]))
        cases.append(([a[:1]], [4], (3, 5), [[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,0]]))
        cases.append(([a[:1]], [-4], (5, 3), [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0]]))
        cases.append(([a[:2]], [-3], (5, 3), [[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,2,0]]))
        cases.append(([a[:3]], [-2], (5, 3), [[0,0,0],[0,0,0],[1,0,0],[0,2,0],[0,0,3]]))
        cases.append(([a[:3]], [-1], (5, 3), [[0,0,0],[1,0,0],[0,2,0],[0,0,3],[0,0,0]]))
        cases.append(([a[:3]], [0], (5, 3), [[1,0,0],[0,2,0],[0,0,3],[0,0,0],[0,0,0]]))
        cases.append(([a[:2]], [1], (5, 3), [[0,1,0],[0,0,2],[0,0,0],[0,0,0],[0,0,0]]))
        cases.append(([a[:1]], [2], (5, 3), [[0,0,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]))

        cases.append(([a[:3],b[:1]], [0,2], (3, 3), [[1,0,6],[0,2,0],[0,0,3]]))
        cases.append(([a[:2],b[:3]], [-1,0], (3, 4), [[6,0,0,0],[1,7,0,0],[0,2,8,0]]))
        cases.append(([a[:4],b[:3]], [2,-3], (6, 6), [[0,0,1,0,0,0],
                                                     [0,0,0,2,0,0],
                                                     [0,0,0,0,3,0],
                                                     [6,0,0,0,0,4],
                                                     [0,7,0,0,0,0],
                                                     [0,0,8,0,0,0]]))

        cases.append(([a[:4],b,c[:4]], [-1,0,1], (5, 5), [[6,11, 0, 0, 0],
                                                            [1, 7,12, 0, 0],
                                                            [0, 2, 8,13, 0],
                                                            [0, 0, 3, 9,14],
                                                            [0, 0, 0, 4,10]]))
        cases.append(([a[:2],b[:3],c], [-4,2,-1], (6, 5), [[0, 0, 6, 0, 0],
                                                          [11, 0, 0, 7, 0],
                                                          [0,12, 0, 0, 8],
                                                          [0, 0,13, 0, 0],
                                                          [1, 0, 0,14, 0],
                                                          [0, 2, 0, 0,15]]))

        # too long arrays are OK
        cases.append(([a], [0], (1, 1), [[1]]))
        cases.append(([a[:3],b], [0,2], (3, 3), [[1, 0, 6], [0, 2, 0], [0, 0, 3]]))
        cases.append((
            np.array([[1, 2, 3], [4, 5, 6]]),
            [0,-1],
            (3, 3),
            [[1, 0, 0], [4, 2, 0], [0, 5, 3]]
        ))

        # scalar case: broadcasting
        cases.append(([1,-2,1], [1,0,-1], (3, 3), [[-2, 1, 0],
                                                    [1, -2, 1],
                                                    [0, 1, -2]]))

        for d, o, shape, result in cases:
            err_msg = f"{d!r} {o!r} {shape!r} {result!r}"
            assert_equal(construct.diags(d, offsets=o, shape=shape).toarray(),
                         result, err_msg=err_msg)

            if (shape[0] == shape[1]
                and hasattr(d[0], '__len__')
                and len(d[0]) <= max(shape)):
                # should be able to find the shape automatically
                assert_equal(construct.diags(d, offsets=o).toarray(), result,
                             err_msg=err_msg)

    def test_diags_default(self):
        a = array([1, 2, 3, 4, 5])
        assert_equal(construct.diags(a).toarray(), np.diag(a))

    def test_diags_default_bad(self):
        a = array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        assert_raises(ValueError, construct.diags, a)

    def test_diags_bad(self):
        a = array([1, 2, 3, 4, 5])
        b = array([6, 7, 8, 9, 10])
        c = array([11, 12, 13, 14, 15])

        cases = []
        cases.append(([a[:0]], 0, (1, 1)))
        cases.append(([a[:4],b,c[:3]], [-1,0,1], (5, 5)))
        cases.append(([a[:2],c,b[:3]], [-4,2,-1], (6, 5)))
        cases.append(([a[:2],c,b[:3]], [-4,2,-1], None))
        cases.append(([], [-4,2,-1], None))
        cases.append(([1], [-5], (4, 4)))
        cases.append(([a], 0, None))

        for d, o, shape in cases:
            assert_raises(ValueError, construct.diags, d, offsets=o, shape=shape)

        assert_raises(TypeError, construct.diags, [[None]], offsets=[0])

    def test_diags_vs_diag(self):
        # Check that
        #
        #    diags([a, b, ...], [i, j, ...]) == diag(a, i) + diag(b, j) + ...
        #

        np.random.seed(1234)

        for n_diags in [1, 2, 3, 4, 5, 10]:
            n = 1 + n_diags//2 + np.random.randint(0, 10)

            offsets = np.arange(-n+1, n-1)
            np.random.shuffle(offsets)
            offsets = offsets[:n_diags]

            diagonals = [np.random.rand(n - abs(q)) for q in offsets]

            mat = construct.diags(diagonals, offsets=offsets)
            dense_mat = sum([np.diag(x, j) for x, j in zip(diagonals, offsets)])

            assert_array_almost_equal_nulp(mat.toarray(), dense_mat)

            if len(offsets) == 1:
                mat = construct.diags(diagonals[0], offsets=offsets[0])
                dense_mat = np.diag(diagonals[0], offsets[0])
                assert_array_almost_equal_nulp(mat.toarray(), dense_mat)

    def test_diags_dtype(self):
        x = construct.diags([2.2], offsets=[0], shape=(2, 2), dtype=int)
        assert_equal(x.dtype, int)
        assert_equal(x.toarray(), [[2, 0], [0, 2]])

    def test_diags_one_diagonal(self):
        d = list(range(5))
        for k in range(-5, 6):
            assert_equal(construct.diags(d, offsets=k).toarray(),
                         construct.diags([d], offsets=[k]).toarray())

    def test_diags_empty(self):
        x = construct.diags([])
        assert_equal(x.shape, (0, 0))

    @pytest.mark.parametrize("identity", [construct.identity, construct.eye_array])
    def test_identity(self, identity):
        assert_equal(identity(1).toarray(), [[1]])
        assert_equal(identity(2).toarray(), [[1,0],[0,1]])

        I = identity(3, dtype='int8', format='dia')
        assert_equal(I.dtype, np.dtype('int8'))
        assert_equal(I.format, 'dia')

        for fmt in sparse_formats:
            I = identity(3, format=fmt)
            assert_equal(I.format, fmt)
            assert_equal(I.toarray(), [[1,0,0],[0,1,0],[0,0,1]])

    @pytest.mark.parametrize("eye", [construct.eye, construct.eye_array])
    def test_eye(self, eye):
        assert_equal(eye(1,1).toarray(), [[1]])
        assert_equal(eye(2,3).toarray(), [[1,0,0],[0,1,0]])
        assert_equal(eye(3,2).toarray(), [[1,0],[0,1],[0,0]])
        assert_equal(eye(3,3).toarray(), [[1,0,0],[0,1,0],[0,0,1]])

        assert_equal(eye(3,3,dtype='int16').dtype, np.dtype('int16'))

        for m in [3, 5]:
            for n in [3, 5]:
                for k in range(-5,6):
                    # scipy.sparse.eye deviates from np.eye here. np.eye will
                    # create arrays of all 0's when the diagonal offset is
                    # greater than the size of the array. For sparse arrays
                    # this makes less sense, especially as it results in dia
                    # arrays with negative diagonals. Therefore sp.sparse.eye
                    # validates that diagonal offsets fall within the shape of
                    # the array. See gh-18555.
                    if (k > 0 and k > n) or (k < 0 and abs(k) > m):
                        with pytest.raises(
                            ValueError, match="Offset.*out of bounds"
                        ):
                            eye(m, n, k=k)

                    else:
                        assert_equal(
                            eye(m, n, k=k).toarray(),
                            np.eye(m, n, k=k)
                        )
                        if m == n:
                            assert_equal(
                                eye(m, k=k).toarray(),
                                np.eye(m, n, k=k)
                            )

    @pytest.mark.parametrize("eye", [construct.eye, construct.eye_array])
    def test_eye_one(self, eye):
        assert_equal(eye(1).toarray(), [[1]])
        assert_equal(eye(2).toarray(), [[1,0],[0,1]])

        I = eye(3, dtype='int8', format='dia')
        assert_equal(I.dtype, np.dtype('int8'))
        assert_equal(I.format, 'dia')

        for fmt in sparse_formats:
            I = eye(3, format=fmt)
            assert_equal(I.format, fmt)
            assert_equal(I.toarray(), [[1,0,0],[0,1,0],[0,0,1]])

    def test_eye_array_vs_matrix(self):
        assert isinstance(construct.eye_array(3), sparray)
        assert not isinstance(construct.eye(3), sparray)

    def test_kron(self):
        cases = []

        cases.append(array([[0]]))
        cases.append(array([[-1]]))
        cases.append(array([[4]]))
        cases.append(array([[10]]))
        cases.append(array([[0],[0]]))
        cases.append(array([[0,0]]))
        cases.append(array([[1,2],[3,4]]))
        cases.append(array([[0,2],[5,0]]))
        cases.append(array([[0,2,-6],[8,0,14]]))
        cases.append(array([[5,4],[0,0],[6,0]]))
        cases.append(array([[5,4,4],[1,0,0],[6,0,8]]))
        cases.append(array([[0,1,0,2,0,5,8]]))
        cases.append(array([[0.5,0.125,0,3.25],[0,2.5,0,0]]))

        # test all cases with some formats
        for a in cases:
            ca = csr_array(a)
            for b in cases:
                cb = csr_array(b)
                expected = np.kron(a, b)
                for fmt in sparse_formats[1:4]:
                    result = construct.kron(ca, cb, format=fmt)
                    assert_equal(result.format, fmt)
                    assert_array_equal(result.toarray(), expected)
                    assert isinstance(result, sparray)

        # test one case with all formats
        a = cases[-1]
        b = cases[-3]
        ca = csr_array(a)
        cb = csr_array(b)

        expected = np.kron(a, b)
        for fmt in sparse_formats:
            result = construct.kron(ca, cb, format=fmt)
            assert_equal(result.format, fmt)
            assert_array_equal(result.toarray(), expected)
            assert isinstance(result, sparray)

        # check that spmatrix returned when both inputs are spmatrix
        result = construct.kron(csr_matrix(a), csr_matrix(b), format=fmt)
        assert_equal(result.format, fmt)
        assert_array_equal(result.toarray(), expected)
        assert isinstance(result, spmatrix)

    def test_kron_large(self):
        n = 2**16
        a = construct.diags_array([1], shape=(1, n), offsets=n-1)
        b = construct.diags_array([1], shape=(n, 1), offsets=1-n)

        construct.kron(a, a)
        construct.kron(b, b)

    def test_kronsum(self):
        cases = []

        cases.append(array([[0]]))
        cases.append(array([[-1]]))
        cases.append(array([[4]]))
        cases.append(array([[10]]))
        cases.append(array([[1,2],[3,4]]))
        cases.append(array([[0,2],[5,0]]))
        cases.append(array([[0,2,-6],[8,0,14],[0,3,0]]))
        cases.append(array([[1,0,0],[0,5,-1],[4,-2,8]]))

        # test all cases with default format
        for a in cases:
            for b in cases:
                result = construct.kronsum(csr_array(a), csr_array(b)).toarray()
                expected = (np.kron(np.eye(b.shape[0]), a)
                            + np.kron(b, np.eye(a.shape[0])))
                assert_array_equal(result, expected)

        # check that spmatrix returned when both inputs are spmatrix
        result = construct.kronsum(csr_matrix(a), csr_matrix(b)).toarray()
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("coo_cls", [coo_matrix, coo_array])
    def test_vstack(self, coo_cls):
        A = coo_cls([[1,2],[3,4]])
        B = coo_cls([[5,6]])

        expected = array([[1, 2],
                          [3, 4],
                          [5, 6]])
        assert_equal(construct.vstack([A, B]).toarray(), expected)
        assert_equal(construct.vstack([A, B], dtype=np.float32).dtype,
                     np.float32)

        assert_equal(construct.vstack([A.tocsr(), B.tocsr()]).toarray(),
                     expected)
        result = construct.vstack([A.tocsr(), B.tocsr()],
                                  format="csr", dtype=np.float32)
        assert_equal(result.dtype, np.float32)
        assert_equal(result.indices.dtype, np.int32)
        assert_equal(result.indptr.dtype, np.int32)

        assert_equal(construct.vstack([A.tocsc(), B.tocsc()]).toarray(),
                     expected)
        result = construct.vstack([A.tocsc(), B.tocsc()],
                                  format="csc", dtype=np.float32)
        assert_equal(result.dtype, np.float32)
        assert_equal(result.indices.dtype, np.int32)
        assert_equal(result.indptr.dtype, np.int32)

    def test_vstack_matrix_or_array(self):
        A = [[1,2],[3,4]]
        B = [[5,6]]
        assert isinstance(construct.vstack([coo_array(A), coo_array(B)]), sparray)
        assert isinstance(construct.vstack([coo_array(A), coo_matrix(B)]), sparray)
        assert isinstance(construct.vstack([coo_matrix(A), coo_array(B)]), sparray)
        assert isinstance(construct.vstack([coo_matrix(A), coo_matrix(B)]), spmatrix)

    @pytest.mark.parametrize("coo_cls", [coo_matrix, coo_array])
    def test_hstack(self,coo_cls):
        A = coo_cls([[1,2],[3,4]])
        B = coo_cls([[5],[6]])

        expected = array([[1, 2, 5],
                          [3, 4, 6]])
        assert_equal(construct.hstack([A, B]).toarray(), expected)
        assert_equal(construct.hstack([A, B], dtype=np.float32).dtype,
                     np.float32)
        assert_equal(construct.hstack([A.tocsc(), B.tocsc()]).toarray(),
                     expected)
        assert_equal(construct.hstack([A.tocsc(), B.tocsc()],
                                      dtype=np.float32).dtype,
                     np.float32)
        assert_equal(construct.hstack([A.tocsr(), B.tocsr()]).toarray(),
                     expected)
        assert_equal(construct.hstack([A.tocsr(), B.tocsr()],
                                      dtype=np.float32).dtype,
                     np.float32)

    def test_hstack_matrix_or_array(self):
        A = [[1,2],[3,4]]
        B = [[5],[6]]
        assert isinstance(construct.hstack([coo_array(A), coo_array(B)]), sparray)
        assert isinstance(construct.hstack([coo_array(A), coo_matrix(B)]), sparray)
        assert isinstance(construct.hstack([coo_matrix(A), coo_array(B)]), sparray)
        assert isinstance(construct.hstack([coo_matrix(A), coo_matrix(B)]), spmatrix)

    @pytest.mark.parametrize("block_array", (construct.bmat, construct.block_array))
    def test_block_creation(self, block_array):

        A = coo_array([[1, 2], [3, 4]])
        B = coo_array([[5],[6]])
        C = coo_array([[7]])
        D = coo_array((0, 0))

        expected = array([[1, 2, 5],
                          [3, 4, 6],
                          [0, 0, 7]])
        assert_equal(block_array([[A, B], [None, C]]).toarray(), expected)
        E = csr_array((1, 2), dtype=np.int32)
        assert_equal(block_array([[A.tocsr(), B.tocsr()],
                                  [E, C.tocsr()]]).toarray(),
                     expected)
        assert_equal(block_array([[A.tocsc(), B.tocsc()],
                                  [E.tocsc(), C.tocsc()]]).toarray(),
                     expected)

        expected = array([[1, 2, 0],
                          [3, 4, 0],
                          [0, 0, 7]])
        assert_equal(block_array([[A, None], [None, C]]).toarray(), expected)
        assert_equal(block_array([[A.tocsr(), E.T.tocsr()],
                                  [E, C.tocsr()]]).toarray(),
                     expected)
        assert_equal(block_array([[A.tocsc(), E.T.tocsc()],
                                  [E.tocsc(), C.tocsc()]]).toarray(),
                     expected)

        Z = csr_array((1, 1), dtype=np.int32)
        expected = array([[0, 5],
                          [0, 6],
                          [7, 0]])
        assert_equal(block_array([[None, B], [C, None]]).toarray(), expected)
        assert_equal(block_array([[E.T.tocsr(), B.tocsr()],
                                  [C.tocsr(), Z]]).toarray(),
                     expected)
        assert_equal(block_array([[E.T.tocsc(), B.tocsc()],
                                  [C.tocsc(), Z.tocsc()]]).toarray(),
                     expected)

        expected = np.empty((0, 0))
        assert_equal(block_array([[None, None]]).toarray(), expected)
        assert_equal(block_array([[None, D], [D, None]]).toarray(),
                     expected)

        # test bug reported in gh-5976
        expected = array([[7]])
        assert_equal(block_array([[None, D], [C, None]]).toarray(),
                     expected)

        # test failure cases
        with assert_raises(ValueError) as excinfo:
            block_array([[A], [B]])
        excinfo.match(r'Got blocks\[1,0\]\.shape\[1\] == 1, expected 2')

        with assert_raises(ValueError) as excinfo:
            block_array([[A.tocsr()], [B.tocsr()]])
        excinfo.match(r'incompatible dimensions for axis 1')

        with assert_raises(ValueError) as excinfo:
            block_array([[A.tocsc()], [B.tocsc()]])
        excinfo.match(r'Mismatching dimensions along axis 1: ({1, 2}|{2, 1})')

        with assert_raises(ValueError) as excinfo:
            block_array([[A, C]])
        excinfo.match(r'Got blocks\[0,1\]\.shape\[0\] == 1, expected 2')

        with assert_raises(ValueError) as excinfo:
            block_array([[A.tocsr(), C.tocsr()]])
        excinfo.match(r'Mismatching dimensions along axis 0: ({1, 2}|{2, 1})')

        with assert_raises(ValueError) as excinfo:
            block_array([[A.tocsc(), C.tocsc()]])
        excinfo.match(r'incompatible dimensions for axis 0')

    def test_block_return_type(self):
        block = construct.block_array

        # csr format ensures we hit _compressed_sparse_stack
        # shape of F,G ensure we hit _stack_along_minor_axis
        # list version ensure we hit the path with neither helper function
        Fl, Gl = [[1, 2],[3, 4]], [[7], [5]]
        Fm, Gm = csr_matrix(Fl), csr_matrix(Gl)
        assert isinstance(block([[None, Fl], [Gl, None]], format="csr"), sparray)
        assert isinstance(block([[None, Fm], [Gm, None]], format="csr"), sparray)
        assert isinstance(block([[Fm, Gm]], format="csr"), sparray)

    def test_bmat_return_type(self):
        """This can be removed after sparse matrix is removed"""
        bmat = construct.bmat
        # check return type. if any input _is_array output array, else matrix
        Fl, Gl = [[1, 2],[3, 4]], [[7], [5]]
        Fm, Gm = csr_matrix(Fl), csr_matrix(Gl)
        Fa, Ga = csr_array(Fl), csr_array(Gl)
        assert isinstance(bmat([[Fa, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Fm, Gm]], format="csr"), spmatrix)
        assert isinstance(bmat([[None, Fa], [Ga, None]], format="csr"), sparray)
        assert isinstance(bmat([[None, Fm], [Ga, None]], format="csr"), sparray)
        assert isinstance(bmat([[None, Fm], [Gm, None]], format="csr"), spmatrix)
        assert isinstance(bmat([[None, Fl], [Gl, None]], format="csr"), spmatrix)

        # type returned by _compressed_sparse_stack (all csr)
        assert isinstance(bmat([[Ga, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Ga, Gm]], format="csr"), sparray)
        assert isinstance(bmat([[Gm, Gm]], format="csr"), spmatrix)
        # shape is 2x2 so no _stack_along_minor_axis
        assert isinstance(bmat([[Fa, Fm]], format="csr"), sparray)
        assert isinstance(bmat([[Fm, Fm]], format="csr"), spmatrix)

        # type returned by _compressed_sparse_stack (all csc)
        assert isinstance(bmat([[Gm.tocsc(), Ga.tocsc()]], format="csc"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Gm.tocsc()]], format="csc"), spmatrix)
        # shape is 2x2 so no _stack_along_minor_axis
        assert isinstance(bmat([[Fa.tocsc(), Fm.tocsc()]], format="csr"), sparray)
        assert isinstance(bmat([[Fm.tocsc(), Fm.tocsc()]], format="csr"), spmatrix)

        # type returned when mixed input
        assert isinstance(bmat([[Gl, Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Ga]], format="csr"), sparray)
        assert isinstance(bmat([[Gm.tocsc(), Gm]], format="csr"), spmatrix)
        assert isinstance(bmat([[Gm, Gm]], format="csc"), spmatrix)

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    def test_concatenate_int32_overflow(self):
        """ test for indptr overflow when concatenating matrices """
        check_free_memory(30000)

        n = 33000
        A = csr_array(np.ones((n, n), dtype=bool))
        B = A.copy()
        C = construct._compressed_sparse_stack((A, B), axis=0,
                                               return_spmatrix=False)

        assert_(np.all(np.equal(np.diff(C.indptr), n)))
        assert_equal(C.indices.dtype, np.int64)
        assert_equal(C.indptr.dtype, np.int64)

    def test_block_diag_basic(self):
        """ basic test for block_diag """
        A = coo_array([[1,2],[3,4]])
        B = coo_array([[5],[6]])
        C = coo_array([[7]])

        expected = array([[1, 2, 0, 0],
                          [3, 4, 0, 0],
                          [0, 0, 5, 0],
                          [0, 0, 6, 0],
                          [0, 0, 0, 7]])

        assert_equal(construct.block_diag((A, B, C)).toarray(), expected)

    def test_block_diag_scalar_1d_args(self):
        """ block_diag with scalar and 1d arguments """
        # one 1d matrix and a scalar
        assert_array_equal(construct.block_diag([[2,3], 4]).toarray(),
                           [[2, 3, 0], [0, 0, 4]])

    def test_block_diag_1(self):
        """ block_diag with one matrix """
        assert_equal(construct.block_diag([[1, 0]]).toarray(),
                     array([[1, 0]]))
        assert_equal(construct.block_diag([[[1, 0]]]).toarray(),
                     array([[1, 0]]))
        assert_equal(construct.block_diag([[[1], [0]]]).toarray(),
                     array([[1], [0]]))
        # just on scalar
        assert_equal(construct.block_diag([1]).toarray(),
                     array([[1]]))

    def test_block_diag_sparse_arrays(self):
        """ block_diag with sparse arrays """

        A = coo_array([[1, 2, 3]], shape=(1, 3))
        B = coo_array([[4, 5]], shape=(1, 2))
        assert_equal(construct.block_diag([A, B]).toarray(),
                     array([[1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))

        A = coo_array([[1], [2], [3]], shape=(3, 1))
        B = coo_array([[4], [5]], shape=(2, 1))
        assert_equal(construct.block_diag([A, B]).toarray(),
                     array([[1, 0], [2, 0], [3, 0], [0, 4], [0, 5]]))

    def test_block_diag_return_type(self):
        A, B = coo_array([[1, 2, 3]]), coo_matrix([[2, 3, 4]])
        assert isinstance(construct.block_diag([A, A]), sparray)
        assert isinstance(construct.block_diag([A, B]), sparray)
        assert isinstance(construct.block_diag([B, A]), sparray)
        assert isinstance(construct.block_diag([B, B]), spmatrix)

    def test_random_sampling(self):
        # Simple sanity checks for sparse random sampling.
        for f in sprand, _sprandn:
            for t in [np.float32, np.float64, np.longdouble,
                      np.int32, np.int64, np.complex64, np.complex128]:
                x = f(5, 10, density=0.1, dtype=t)
                assert_equal(x.dtype, t)
                assert_equal(x.shape, (5, 10))
                assert_equal(x.nnz, 5)

            x1 = f(5, 10, density=0.1, random_state=4321)
            assert_equal(x1.dtype, np.float64)

            x2 = f(5, 10, density=0.1,
                   random_state=np.random.RandomState(4321))

            assert_array_equal(x1.data, x2.data)
            assert_array_equal(x1.row, x2.row)
            assert_array_equal(x1.col, x2.col)

            for density in [0.0, 0.1, 0.5, 1.0]:
                x = f(5, 10, density=density)
                assert_equal(x.nnz, int(density * np.prod(x.shape)))

            for fmt in ['coo', 'csc', 'csr', 'lil']:
                x = f(5, 10, format=fmt)
                assert_equal(x.format, fmt)

            assert_raises(ValueError, lambda: f(5, 10, 1.1))
            assert_raises(ValueError, lambda: f(5, 10, -0.1))

    def test_rand(self):
        # Simple distributional checks for sparse.rand.
        random_states = [None, 4321, np.random.RandomState()]
        try:
            gen = np.random.default_rng()
            random_states.append(gen)
        except AttributeError:
            pass

        for random_state in random_states:
            x = sprand(10, 20, density=0.5, dtype=np.float64,
                       random_state=random_state)
            assert_(np.all(np.less_equal(0, x.data)))
            assert_(np.all(np.less_equal(x.data, 1)))

    def test_randn(self):
        # Simple distributional checks for sparse.randn.
        # Statistically, some of these should be negative
        # and some should be greater than 1.
        random_states = [None, 4321, np.random.RandomState()]
        try:
            gen = np.random.default_rng()
            random_states.append(gen)
        except AttributeError:
            pass

        for rs in random_states:
            x = _sprandn(10, 20, density=0.5, dtype=np.float64, random_state=rs)
            assert_(np.any(np.less(x.data, 0)))
            assert_(np.any(np.less(1, x.data)))
            x = _sprandn_array(10, 20, density=0.5, dtype=np.float64, random_state=rs)
            assert_(np.any(np.less(x.data, 0)))
            assert_(np.any(np.less(1, x.data)))

    def test_random_accept_str_dtype(self):
        # anything that np.dtype can convert to a dtype should be accepted
        # for the dtype
        construct.random(10, 10, dtype='d')
        construct.random_array((10, 10), dtype='d')

    def test_random_sparse_matrix_returns_correct_number_of_non_zero_elements(self):
        # A 10 x 10 matrix, with density of 12.65%, should have 13 nonzero elements.
        # 10 x 10 x 0.1265 = 12.65, which should be rounded up to 13, not 12.
        sparse_matrix = construct.random(10, 10, density=0.1265)
        assert_equal(sparse_matrix.count_nonzero(),13)
        # check random_array
        sparse_array = construct.random_array((10, 10), density=0.1265)
        assert_equal(sparse_array.count_nonzero(),13)
        assert isinstance(sparse_array, sparray)
        # check big size
        shape = (2**33, 2**33)
        sparse_array = construct.random_array(shape, density=2.7105e-17)
        assert_equal(sparse_array.count_nonzero(),2000)


def test_diags_array():
    """Tests of diags_array that do not rely on diags wrapper."""
    diag = np.arange(1, 5)

    assert_array_equal(construct.diags_array(diag).toarray(), np.diag(diag))

    assert_array_equal(
        construct.diags_array(diag, offsets=2).toarray(), np.diag(diag, k=2)
    )

    assert_array_equal(
        construct.diags_array(diag, offsets=2, shape=(4, 4)).toarray(),
        np.diag(diag, k=2)[:4, :4]
    )

    # Offset outside bounds when shape specified
    with pytest.raises(ValueError, match=".*out of bounds"):
        construct.diags(np.arange(1, 5), 5, shape=(4, 4))
