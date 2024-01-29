import numpy as np
from scipy.linalg import solve


__all__ = ['nnls']


def nnls(A, b, maxiter=None, *, atol=None):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``.

    This problem, often called as NonNegative Least Squares, is a convex
    optimization problem with convex constraints. It typically arises when
    the ``x`` models quantities for which only nonnegative values are
    attainable; weight of ingredients, component costs and so on.

    Parameters
    ----------
    A : (m, n) ndarray
        Coefficient array
    b : (m,) ndarray, float
        Right-hand side vector.
    maxiter: int, optional
        Maximum number of iterations, optional. Default value is ``3 * n``.
    atol: float
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(A.T @ (A x - b)`` entries. Increasing this
        value relaxes the solution constraints. A typical relaxation value can
        be selected as ``max(m, n) * np.linalg.norm(a, 1) * np.spacing(1.)``.
        This value is not set as default since the norm operation becomes
        expensive for large problems hence can be used only when necessary.

    Returns
    -------
    x : ndarray
        Solution vector.
    rnorm : float
        The 2-norm of the residual, ``|| Ax-b ||_2``.

    See Also
    --------
    lsq_linear : Linear least squares with bounds on the variables

    Notes
    -----
    The code is based on [2]_ which is an improved version of the classical
    algorithm of [1]_. It utilizes an active set method and solves the KKT
    (Karush-Kuhn-Tucker) conditions for the non-negative least squares problem.

    References
    ----------
    .. [1] : Lawson C., Hanson R.J., "Solving Least Squares Problems", SIAM,
       1995, :doi:`10.1137/1.9781611971217`
    .. [2] : Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-
       Constrained Least Squares Algorithm", Journal Of Chemometrics, 1997,
       :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`

     Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import nnls
    ...
    >>> A = np.array([[1, 0], [1, 0], [0, 1]])
    >>> b = np.array([2, 1, 1])
    >>> nnls(A, b)
    (array([1.5, 1. ]), 0.7071067811865475)

    >>> b = np.array([-1, -1, -1])
    >>> nnls(A, b)
    (array([0., 0.]), 1.7320508075688772)

    """

    A = np.asarray_chkfinite(A)
    b = np.asarray_chkfinite(b)

    if len(A.shape) != 2:
        raise ValueError("Expected a two-dimensional array (matrix)" +
                         f", but the shape of A is {A.shape}")
    if len(b.shape) != 1:
        raise ValueError("Expected a one-dimensional array (vector)" +
                         f", but the shape of b is {b.shape}")

    m, n = A.shape

    if m != b.shape[0]:
        raise ValueError(
                "Incompatible dimensions. The first dimension of " +
                f"A is {m}, while the shape of b is {(b.shape[0], )}")

    x, rnorm, mode = _nnls(A, b, maxiter, tol=atol)
    if mode != 1:
        raise RuntimeError("Maximum number of iterations reached.")

    return x, rnorm


def _nnls(A, b, maxiter=None, tol=None):
    """
    This is a single RHS algorithm from ref [2] above. For multiple RHS
    support, the algorithm is given in  :doi:`10.1002/cem.889`
    """
    m, n = A.shape

    AtA = A.T @ A
    Atb = b @ A  # Result is 1D - let NumPy figure it out

    if not maxiter:
        maxiter = 3*n
    if tol is None:
        tol = 10 * max(m, n) * np.spacing(1.)

    # Initialize vars
    x = np.zeros(n, dtype=np.float64)

    # Inactive constraint switches
    P = np.zeros(n, dtype=bool)

    # Projected residual
    resid = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

    # Overall iteration counter
    # Outer loop is not counted, inner iter is counted across outer spins
    iter = 0

    while (not P.all()) and (resid[~P] > tol).any():  # B
        # Get the "most" active coeff index and move to inactive set
        resid[P] = -np.inf
        k = np.argmax(resid)  # B.2
        P[k] = True  # B.3

        # Iteration solution
        s = np.zeros(n, dtype=np.float64)
        P_ind = P.nonzero()[0]
        s[P] = solve(AtA[P_ind[:, None], P_ind[None, :]], Atb[P],
                     assume_a='sym', check_finite=False)  # B.4

        # Inner loop
        while (iter < maxiter) and (s[P].min() <= tol):  # C.1
            alpha_ind = ((s < tol) & P).nonzero()
            alpha = (x[alpha_ind] / (x[alpha_ind] - s[alpha_ind])).min()  # C.2
            x *= (1 - alpha)
            x += alpha*s
            P[x < tol] = False
            s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym',
                         check_finite=False)
            s[~P] = 0  # C.6
            iter += 1

        x[:] = s[:]
        resid = Atb - AtA @ x

        if iter == maxiter:
            # Typically following line should return
            # return x, np.linalg.norm(A@x - b), -1
            # however at the top level, -1 raises an exception wasting norm
            # Instead return dummy number 0.
            return x, 0., -1

    return x, np.linalg.norm(A@x - b), 1
