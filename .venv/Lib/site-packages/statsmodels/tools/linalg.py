"""
Linear Algebra solvers and other helpers
"""
import numpy as np

__all__ = ["logdet_symm", "stationary_solve", "transf_constraints",
           "matrix_sqrt"]


def logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array_like
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    from scipy import linalg
    if check_symm:
        if not np.all(m == m.T):  # would be nice to short-circuit check
            raise ValueError("m is not symmetric.")
    c, _ = linalg.cho_factor(m, lower=True)
    return 2*np.sum(np.log(c.diagonal()))


def stationary_solve(r, b):
    """
    Solve a linear system for a Toeplitz correlation matrix.

    A Toeplitz correlation matrix represents the covariance of a
    stationary series with unit variance.

    Parameters
    ----------
    r : array_like
        A vector describing the coefficient matrix.  r[0] is the first
        band next to the diagonal, r[1] is the second band, etc.
    b : array_like
        The right-hand side for which we are solving, i.e. we solve
        Tx = b and return b, where T is the Toeplitz coefficient matrix.

    Returns
    -------
    The solution to the linear system.
    """

    db = r[0:1]

    dim = b.ndim
    if b.ndim == 1:
        b = b[:, None]
    x = b[0:1, :]

    for j in range(1, len(b)):
        rf = r[0:j][::-1]
        a = (b[j, :] - np.dot(rf, x)) / (1 - np.dot(rf, db[::-1]))
        z = x - np.outer(db[::-1], a)
        x = np.concatenate((z, a[None, :]), axis=0)

        if j == len(b) - 1:
            break

        rn = r[j]
        a = (rn - np.dot(rf, db)) / (1 - np.dot(rf, db[::-1]))
        z = db - a*db[::-1]
        db = np.concatenate((z, np.r_[a]))

    if dim == 1:
        x = x[:, 0]

    return x


def transf_constraints(constraints):
    """use QR to get transformation matrix to impose constraint

    Parameters
    ----------
    constraints : ndarray, 2-D
        restriction matrix with one constraints in rows

    Returns
    -------
    transf : ndarray
        transformation matrix to reparameterize so that constraint is
        imposed

    Notes
    -----
    This is currently and internal helper function for GAM.
    API not stable and will most likely change.

    The code for this function was taken from patsy spline handling, and
    corresponds to the reparameterization used by Wood in R's mgcv package.

    See Also
    --------
    statsmodels.base._constraints.TransformRestriction : class to impose
        constraints by reparameterization used by `_fit_constrained`.
    """

    from scipy import linalg

    m = constraints.shape[0]
    q, _ = linalg.qr(np.transpose(constraints))
    transf = q[:, m:]
    return transf


def matrix_sqrt(mat, inverse=False, full=False, nullspace=False,
                threshold=1e-15):
    """matrix square root for symmetric matrices

    Usage is for decomposing a covariance function S into a square root R
    such that

        R' R = S if inverse is False, or
        R' R = pinv(S) if inverse is True

    Parameters
    ----------
    mat : array_like, 2-d square
        symmetric square matrix for which square root or inverse square
        root is computed.
        There is no checking for whether the matrix is symmetric.
        A warning is issued if some singular values are negative, i.e.
        below the negative of the threshold.
    inverse : bool
        If False (default), then the matrix square root is returned.
        If inverse is True, then the matrix square root of the inverse
        matrix is returned.
    full : bool
        If full is False (default, then the square root has reduce number
        of rows if the matrix is singular, i.e. has singular values below
        the threshold.
    nullspace : bool
        If nullspace is true, then the matrix square root of the null space
        of the matrix is returned.
    threshold : float
        Singular values below the threshold are dropped.

    Returns
    -------
    msqrt : ndarray
        matrix square root or square root of inverse matrix.
    """
    # see also scipy.linalg null_space
    u, s, v = np.linalg.svd(mat)
    if np.any(s < -threshold):
        import warnings
        warnings.warn('some singular values are negative')

    if not nullspace:
        mask = s > threshold
        s[s < threshold] = 0
    else:
        mask = s < threshold
        s[s > threshold] = 0

    sqrt_s = np.sqrt(s[mask])
    if inverse:
        sqrt_s = 1 / np.sqrt(s[mask])

    if full:
        b = np.dot(u[:, mask], np.dot(np.diag(sqrt_s), v[mask]))
    else:
        b = np.dot(np.diag(sqrt_s), v[mask])
    return b
