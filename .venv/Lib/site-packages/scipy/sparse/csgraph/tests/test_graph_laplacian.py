import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse

from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong


def check_int_type(mat):
    return np.issubdtype(mat.dtype, np.signedinteger) or np.issubdtype(
        mat.dtype, np_ulong
    )


def test_laplacian_value_error():
    for t in int, float, complex:
        for m in ([1, 1],
                  [[[1]]],
                  [[1, 2, 3], [4, 5, 6]],
                  [[1, 2], [3, 4], [5, 5]]):
            A = np.array(m, dtype=t)
            assert_raises(ValueError, csgraph.laplacian, A)


def _explicit_laplacian(x, normed=False):
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    y = -1.0 * x
    for j in range(y.shape[0]):
        y[j,j] = x[j,j+1:].sum() + x[j,:j].sum()
    if normed:
        d = np.diag(y).copy()
        d[d == 0] = 1.0
        y /= d[:,None]**.5
        y /= d[None,:]**.5
    return y


def _check_symmetric_graph_laplacian(mat, normed, copy=True):
    if not hasattr(mat, 'shape'):
        mat = eval(mat, dict(np=np, sparse=sparse))

    if sparse.issparse(mat):
        sp_mat = mat
        mat = sp_mat.toarray()
    else:
        sp_mat = sparse.csr_matrix(mat)

    mat_copy = np.copy(mat)
    sp_mat_copy = sparse.csr_matrix(sp_mat, copy=True)

    n_nodes = mat.shape[0]
    explicit_laplacian = _explicit_laplacian(mat, normed=normed)
    laplacian = csgraph.laplacian(mat, normed=normed, copy=copy)
    sp_laplacian = csgraph.laplacian(sp_mat, normed=normed,
                                     copy=copy)

    if copy:
        assert_allclose(mat, mat_copy)
        _assert_allclose_sparse(sp_mat, sp_mat_copy)
    else:
        if not (normed and check_int_type(mat)):
            assert_allclose(laplacian, mat)
            if sp_mat.format == 'coo':
                _assert_allclose_sparse(sp_laplacian, sp_mat)

    assert_allclose(laplacian, sp_laplacian.toarray())

    for tested in [laplacian, sp_laplacian.toarray()]:
        if not normed:
            assert_allclose(tested.sum(axis=0), np.zeros(n_nodes))
        assert_allclose(tested.T, tested)
        assert_allclose(tested, explicit_laplacian)


def test_symmetric_graph_laplacian():
    symmetric_mats = (
        'np.arange(10) * np.arange(10)[:, np.newaxis]',
        'np.ones((7, 7))',
        'np.eye(19)',
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4))',
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).toarray()',
        'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).todense()',
        'np.vander(np.arange(4)) + np.vander(np.arange(4)).T'
    )
    for mat in symmetric_mats:
        for normed in True, False:
            for copy in True, False:
                _check_symmetric_graph_laplacian(mat, normed, copy)


def _assert_allclose_sparse(a, b, **kwargs):
    # helper function that can deal with sparse matrices
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = b.toarray()
    assert_allclose(a, b, **kwargs)


def _check_laplacian_dtype_none(
    A, desired_L, desired_d, normed, use_out_degree, copy, dtype, arr_type
):
    mat = arr_type(A, dtype=dtype)
    L, d = csgraph.laplacian(
        mat,
        normed=normed,
        return_diag=True,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=None,
    )
    if normed and check_int_type(mat):
        assert L.dtype == np.float64
        assert d.dtype == np.float64
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)
    else:
        assert L.dtype == dtype
        assert d.dtype == dtype
        desired_L = np.asarray(desired_L).astype(dtype)
        desired_d = np.asarray(desired_d).astype(dtype)
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)

    if not copy:
        if not (normed and check_int_type(mat)):
            if type(mat) is np.ndarray:
                assert_allclose(L, mat)
            elif mat.format == "coo":
                _assert_allclose_sparse(L, mat)


def _check_laplacian_dtype(
    A, desired_L, desired_d, normed, use_out_degree, copy, dtype, arr_type
):
    mat = arr_type(A, dtype=dtype)
    L, d = csgraph.laplacian(
        mat,
        normed=normed,
        return_diag=True,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
    )
    assert L.dtype == dtype
    assert d.dtype == dtype
    desired_L = np.asarray(desired_L).astype(dtype)
    desired_d = np.asarray(desired_d).astype(dtype)
    _assert_allclose_sparse(L, desired_L, atol=1e-12)
    _assert_allclose_sparse(d, desired_d, atol=1e-12)

    if not copy:
        if not (normed and check_int_type(mat)):
            if type(mat) is np.ndarray:
                assert_allclose(L, mat)
            elif mat.format == 'coo':
                _assert_allclose_sparse(L, mat)


INT_DTYPES = {np.intc, np_long, np.longlong}
REAL_DTYPES = {np.float32, np.float64, np.longdouble}
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}
# use sorted list to ensure fixed order of tests
DTYPES = sorted(INT_DTYPES ^ REAL_DTYPES ^ COMPLEX_DTYPES, key=str)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arr_type", [np.array,
                                      sparse.csr_matrix,
                                      sparse.coo_matrix,
                                      sparse.csr_array,
                                      sparse.coo_array])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("use_out_degree", [True, False])
def test_asymmetric_laplacian(use_out_degree, normed,
                              copy, dtype, arr_type):
    # adjacency matrix
    A = [[0, 1, 0],
         [4, 2, 0],
         [0, 0, 0]]
    A = arr_type(np.array(A), dtype=dtype)
    A_copy = A.copy()

    if not normed and use_out_degree:
        # Laplacian matrix using out-degree
        L = [[1, -1, 0],
             [-4, 4, 0],
             [0, 0, 0]]
        d = [1, 4, 0]

    if normed and use_out_degree:
        # normalized Laplacian matrix using out-degree
        L = [[1, -0.5, 0],
             [-2, 1, 0],
             [0, 0, 0]]
        d = [1, 2, 1]

    if not normed and not use_out_degree:
        # Laplacian matrix using in-degree
        L = [[4, -1, 0],
             [-4, 1, 0],
             [0, 0, 0]]
        d = [4, 1, 0]

    if normed and not use_out_degree:
        # normalized Laplacian matrix using in-degree
        L = [[1, -0.5, 0],
             [-2, 1, 0],
             [0, 0, 0]]
        d = [2, 1, 1]

    _check_laplacian_dtype_none(
        A,
        L,
        d,
        normed=normed,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
        arr_type=arr_type,
    )

    _check_laplacian_dtype(
        A_copy,
        L,
        d,
        normed=normed,
        use_out_degree=use_out_degree,
        copy=copy,
        dtype=dtype,
        arr_type=arr_type,
    )


@pytest.mark.parametrize("fmt", ['csr', 'csc', 'coo', 'lil',
                                 'dok', 'dia', 'bsr'])
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_sparse_formats(fmt, normed, copy):
    mat = sparse.diags([1, 1], [-1, 1], shape=(4, 4), format=fmt)
    _check_symmetric_graph_laplacian(mat, normed, copy)


@pytest.mark.parametrize(
    "arr_type", [np.asarray,
                 sparse.csr_matrix,
                 sparse.coo_matrix,
                 sparse.csr_array,
                 sparse.coo_array]
)
@pytest.mark.parametrize("form", ["array", "function", "lo"])
def test_laplacian_symmetrized(arr_type, form):
    # adjacency matrix
    n = 3
    mat = arr_type(np.arange(n * n).reshape(n, n))
    L_in, d_in = csgraph.laplacian(
        mat,
        return_diag=True,
        form=form,
    )
    L_out, d_out = csgraph.laplacian(
        mat,
        return_diag=True,
        use_out_degree=True,
        form=form,
    )
    Ls, ds = csgraph.laplacian(
        mat,
        return_diag=True,
        symmetrized=True,
        form=form,
    )
    Ls_normed, ds_normed = csgraph.laplacian(
        mat,
        return_diag=True,
        symmetrized=True,
        normed=True,
        form=form,
    )
    mat += mat.T
    Lss, dss = csgraph.laplacian(mat, return_diag=True, form=form)
    Lss_normed, dss_normed = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=True,
        form=form,
    )

    assert_allclose(ds, d_in + d_out)
    assert_allclose(ds, dss)
    assert_allclose(ds_normed, dss_normed)

    d = {}
    for L in ["L_in", "L_out", "Ls", "Ls_normed", "Lss", "Lss_normed"]:
        if form == "array":
            d[L] = eval(L)
        else:
            d[L] = eval(L)(np.eye(n, dtype=mat.dtype))

    _assert_allclose_sparse(d["Ls"], d["L_in"] + d["L_out"].T)
    _assert_allclose_sparse(d["Ls"], d["Lss"])
    _assert_allclose_sparse(d["Ls_normed"], d["Lss_normed"])


@pytest.mark.parametrize(
    "arr_type", [np.asarray,
                 sparse.csr_matrix,
                 sparse.coo_matrix,
                 sparse.csr_array,
                 sparse.coo_array]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("normed", [True, False])
@pytest.mark.parametrize("symmetrized", [True, False])
@pytest.mark.parametrize("use_out_degree", [True, False])
@pytest.mark.parametrize("form", ["function", "lo"])
def test_format(dtype, arr_type, normed, symmetrized, use_out_degree, form):
    n = 3
    mat = [[0, 1, 0], [4, 2, 0], [0, 0, 0]]
    mat = arr_type(np.array(mat), dtype=dtype)
    Lo, do = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=normed,
        symmetrized=symmetrized,
        use_out_degree=use_out_degree,
        dtype=dtype,
    )
    La, da = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=normed,
        symmetrized=symmetrized,
        use_out_degree=use_out_degree,
        dtype=dtype,
        form="array",
    )
    assert_allclose(do, da)
    _assert_allclose_sparse(Lo, La)

    L, d = csgraph.laplacian(
        mat,
        return_diag=True,
        normed=normed,
        symmetrized=symmetrized,
        use_out_degree=use_out_degree,
        dtype=dtype,
        form=form,
    )
    assert_allclose(d, do)
    assert d.dtype == dtype
    Lm = L(np.eye(n, dtype=mat.dtype)).astype(dtype)
    _assert_allclose_sparse(Lm, Lo, rtol=2e-7, atol=2e-7)
    x = np.arange(6).reshape(3, 2)
    if not (normed and dtype in INT_DTYPES):
        assert_allclose(L(x), Lo @ x)
    else:
        # Normalized Lo is casted to integer, but L() is not
        pass


def test_format_error_message():
    with pytest.raises(ValueError, match="Invalid form: 'toto'"):
        _ = csgraph.laplacian(np.eye(1), form='toto')
