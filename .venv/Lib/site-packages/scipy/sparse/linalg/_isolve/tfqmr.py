import numpy as np
from .utils import make_system


__all__ = ['tfqmr']


def tfqmr(A, b, x0=None, tol=1e-5, maxiter=None, M=None,
          callback=None, atol=None, show=False):
    """
    Use Transpose-Free Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        `scipy.sparse.linalg.LinearOperator`.
    b : {ndarray}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : {ndarray}
        Starting guess for the solution.
    tol, atol : float, optional
        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b-Ax0), atol)``.
        The default for `tol` is 1.0e-5.
        The default for `atol` is ``tol * norm(b-Ax0)``.

        .. warning::

           The default value for `atol` will be changed in a future release.
           For future compatibility, specify `atol` explicitly.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
        Default is ``min(10000, ndofs * 10)``, where ``ndofs = A.shape[0]``.
    M : {sparse matrix, ndarray, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as `callback(xk)`, where `xk` is the current solution vector.
    show : bool, optional
        Specify ``show = True`` to show the convergence, ``show = False`` is
        to close the output of the convergence.
        Default is `False`.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The Transpose-Free QMR algorithm is derived from the CGS algorithm.
    However, unlike CGS, the convergence curves for the TFQMR method is
    smoothed by computing a quasi minimization of the residual norm. The
    implementation supports left preconditioner, and the "residual norm"
    to compute in convergence criterion is actually an upper bound on the
    actual residual norm ``||b - Axk||``.

    References
    ----------
    .. [1] R. W. Freund, A Transpose-Free Quasi-Minimal Residual Algorithm for
           Non-Hermitian Linear Systems, SIAM J. Sci. Comput., 14(2), 470-482,
           1993.
    .. [2] Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd edition,
           SIAM, Philadelphia, 2003.
    .. [3] C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations,
           number 16 in Frontiers in Applied Mathematics, SIAM, Philadelphia,
           1995.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import tfqmr
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = tfqmr(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """

    # Check data type
    dtype = A.dtype
    if np.issubdtype(dtype, np.int64):
        dtype = float
        A = A.astype(dtype)
    if np.issubdtype(b.dtype, np.int64):
        b = b.astype(dtype)

    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Check if the R.H.S is a zero vector
    if np.linalg.norm(b) == 0.:
        x = b.copy()
        return (postprocess(x), 0)

    ndofs = A.shape[0]
    if maxiter is None:
        maxiter = min(10000, ndofs * 10)

    if x0 is None:
        r = b.copy()
    else:
        r = b - A.matvec(x)
    u = r
    w = r.copy()
    # Take rstar as b - Ax0, that is rstar := r = b - Ax0 mathematically
    rstar = r
    v = M.matvec(A.matvec(r))
    uhat = v
    d = theta = eta = 0.
    rho = np.inner(rstar.conjugate(), r)
    rhoLast = rho
    r0norm = np.sqrt(rho)
    tau = r0norm
    if r0norm == 0:
        return (postprocess(x), 0)

    if atol is None:
        atol = tol * r0norm
    else:
        atol = max(atol, tol * r0norm)

    for iter in range(maxiter):
        even = iter % 2 == 0
        if (even):
            vtrstar = np.inner(rstar.conjugate(), v)
            # Check breakdown
            if vtrstar == 0.:
                return (postprocess(x), -1)
            alpha = rho / vtrstar
            uNext = u - alpha * v  # [1]-(5.6)
        w -= alpha * uhat  # [1]-(5.8)
        d = u + (theta**2 / alpha) * eta * d  # [1]-(5.5)
        # [1]-(5.2)
        theta = np.linalg.norm(w) / tau
        c = np.sqrt(1. / (1 + theta**2))
        tau *= theta * c
        # Calculate step and direction [1]-(5.4)
        eta = (c**2) * alpha
        z = M.matvec(d)
        x += eta * z

        if callback is not None:
            callback(x)

        # Convergence criteron
        if tau * np.sqrt(iter+1) < atol:
            if (show):
                print("TFQMR: Linear solve converged due to reach TOL "
                      "iterations {}".format(iter+1))
            return (postprocess(x), 0)

        if (not even):
            # [1]-(5.7)
            rho = np.inner(rstar.conjugate(), w)
            beta = rho / rhoLast
            u = w + beta * u
            v = beta * uhat + (beta**2) * v
            uhat = M.matvec(A.matvec(u))
            v += uhat
        else:
            uhat = M.matvec(A.matvec(uNext))
            u = uNext
            rhoLast = rho

    if (show):
        print("TFQMR: Linear solve not converged due to reach MAXIT "
              "iterations {}".format(iter+1))
    return (postprocess(x), maxiter)
