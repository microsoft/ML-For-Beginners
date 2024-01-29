import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, hstack
import pytest


def _check_csr_rowslice(i, sl, X, Xcsr):
    np_slice = X[i, sl]
    csr_slice = Xcsr[i, sl]
    assert_array_almost_equal(np_slice, csr_slice.toarray()[0])
    assert_(type(csr_slice) is csr_matrix)


def test_csr_rowslice():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    slices = [slice(None, None, None),
              slice(None, None, -1),
              slice(1, -2, 2),
              slice(-2, 1, -2)]

    for i in range(N):
        for sl in slices:
            _check_csr_rowslice(i, sl, X, Xcsr)


def test_csr_getrow():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_row = X[i:i + 1, :]
        csr_row = Xcsr.getrow(i)

        assert_array_almost_equal(arr_row, csr_row.toarray())
        assert_(type(csr_row) is csr_matrix)


def test_csr_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_col = X[:, i:i + 1]
        csr_col = Xcsr.getcol(i)

        assert_array_almost_equal(arr_col, csr_col.toarray())
        assert_(type(csr_col) is csr_matrix)

@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      0, (0, 4)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      1, (3, 0)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      'both', (0, 0)),
     (csr_matrix([[0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0]]),
      0, (0, 5))])
def test_csr_empty_slices(matrix_input, axis, expected_shape):
    # see gh-11127 for related discussion
    slice_1 = matrix_input.toarray().shape[0] - 1
    slice_2 = slice_1
    slice_3 = slice_2 - 1

    if axis == 0:
        actual_shape_1 = matrix_input[slice_1:slice_2, :].toarray().shape
        actual_shape_2 = matrix_input[slice_1:slice_3, :].toarray().shape
    elif axis == 1:
        actual_shape_1 = matrix_input[:, slice_1:slice_2].toarray().shape
        actual_shape_2 = matrix_input[:, slice_1:slice_3].toarray().shape
    elif axis == 'both':
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].toarray().shape
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].toarray().shape

    assert actual_shape_1 == expected_shape
    assert actual_shape_1 == actual_shape_2


def test_csr_bool_indexing():
    data = csr_matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    list_indices1 = [False, True, False]
    array_indices1 = np.array(list_indices1)
    list_indices2 = [[False, True, False], [False, True, False], [False, True, False]]
    array_indices2 = np.array(list_indices2)
    list_indices3 = ([False, True, False], [False, True, False])
    array_indices3 = (np.array(list_indices3[0]), np.array(list_indices3[1]))
    slice_list1 = data[list_indices1].toarray()
    slice_array1 = data[array_indices1].toarray()
    slice_list2 = data[list_indices2]
    slice_array2 = data[array_indices2]
    slice_list3 = data[list_indices3]
    slice_array3 = data[array_indices3]
    assert (slice_list1 == slice_array1).all()
    assert (slice_list2 == slice_array2).all()
    assert (slice_list3 == slice_array3).all()


def test_csr_hstack_int64():
    """
    Tests if hstack properly promotes to indices and indptr arrays to np.int64
    when using np.int32 during concatenation would result in either array
    overflowing.
    """
    max_int32 = np.iinfo(np.int32).max

    # First case: indices would overflow with int32
    data = [1.0]
    row = [0]

    max_indices_1 = max_int32 - 1
    max_indices_2 = 3

    # Individual indices arrays are representable with int32
    col_1 = [max_indices_1 - 1]
    col_2 = [max_indices_2 - 1]

    X_1 = csr_matrix((data, (row, col_1)))
    X_2 = csr_matrix((data, (row, col_2)))

    assert max(max_indices_1 - 1, max_indices_2 - 1) < max_int32
    assert X_1.indices.dtype == X_1.indptr.dtype == np.int32
    assert X_2.indices.dtype == X_2.indptr.dtype == np.int32

    # ... but when concatenating their CSR matrices, the resulting indices
    # array can't be represented with int32 and must be promoted to int64.
    X_hs = hstack([X_1, X_2], format="csr")

    assert X_hs.indices.max() == max_indices_1 + max_indices_2 - 1
    assert max_indices_1 + max_indices_2 - 1 > max_int32
    assert X_hs.indices.dtype == X_hs.indptr.dtype == np.int64

    # Even if the matrices are empty, we must account for their size
    # contribution so that we may safely set the final elements.
    X_1_empty = csr_matrix(X_1.shape)
    X_2_empty = csr_matrix(X_2.shape)
    X_hs_empty = hstack([X_1_empty, X_2_empty], format="csr")

    assert X_hs_empty.shape == X_hs.shape
    assert X_hs_empty.indices.dtype == np.int64

    # Should be just small enough to stay in int32 after stack. Note that
    # we theoretically could support indices.max() == max_int32, but due to an
    # edge-case in the underlying sparsetools code
    # (namely the `coo_tocsr` routine),
    # we require that max(X_hs_32.shape) < max_int32 as well.
    # Hence we can only support max_int32 - 1.
    col_3 = [max_int32 - max_indices_1 - 1]
    X_3 = csr_matrix((data, (row, col_3)))
    X_hs_32 = hstack([X_1, X_3], format="csr")
    assert X_hs_32.indices.dtype == np.int32
    assert X_hs_32.indices.max() == max_int32 - 1
