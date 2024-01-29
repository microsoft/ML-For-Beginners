# -*- coding: utf-8 -*-
"""

Created on Fri Aug 17 13:10:52 2012

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings

from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
    IterationLimitWarning, iteration_limit_doc)


def clip_evals(x, value=0):  # threshold=0, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < value)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return x_new, clipped


def corr_nearest(corr, threshold=1e-15, n_fact=100):
    '''
    Find the nearest correlation matrix that is positive semi-definite.

    The function iteratively adjust the correlation matrix by clipping the
    eigenvalues of a difference matrix. The diagonal elements are set to one.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations. The maximum
        number of iterations is the integer part of the number of columns in
        the correlation matrix times n_fact.

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix

    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite or positive definite, so that smallest eigenvalue is above
    threshold. In this case, the returned array is not the original, but
    is equal to it within numerical precision.

    See Also
    --------
    corr_clipped
    cov_nearest

    '''
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError("matrix is not square")

    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)

    for ii in range(int(len(corr) * n_fact)):
        x_adj = x_new - diff
        x_psd, clipped = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        warnings.warn(iteration_limit_doc, IterationLimitWarning)

    return x_new


def corr_clipped(corr, threshold=1e-15):
    '''
    Find a near correlation matrix that is positive semi-definite

    This function clips the eigenvalues, replacing eigenvalues smaller than
    the threshold by the threshold. The new matrix is normalized, so that the
    diagonal elements are one.
    Compared to corr_nearest, the distance between the original correlation
    matrix and the positive definite correlation matrix is larger, however,
    it is much faster since it only computes eigenvalues once.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix


    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``. In examples, the
    smallest eigenvalue can be by a factor of 10 smaller than the threshold,
    e.g. threshold 1e-8 can result in smallest eigenvalue in the range
    between 1e-9 and 1e-8.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric. The diagonal elements of
    returned correlation matrix is set to ones.

    If the correlation matrix is already positive semi-definite given the
    threshold, then the original correlation matrix is returned.

    ``cov_clipped`` is 40 or more times faster than ``cov_nearest`` in simple
    example, but has a slightly larger approximation error.

    See Also
    --------
    corr_nearest
    cov_nearest

    '''
    x_new, clipped = clip_evals(corr, value=threshold)
    if not clipped:
        return corr

    # cov2corr
    x_std = np.sqrt(np.diag(x_new))
    x_new = x_new / x_std / x_std[:, None]
    return x_new


def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100,
                return_all=False):
    """
    Find the nearest covariance matrix that is positive (semi-) definite

    This leaves the diagonal, i.e. the variance, unchanged

    Parameters
    ----------
    cov : ndarray, (k,k)
        initial covariance matrix
    method : str
        if "clipped", then the faster but less accurate ``corr_clipped`` is
        used.if "nearest", then ``corr_nearest`` is used
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations in
        ``corr_nearest``. See its doc string
    return_all : bool
        if False (default), then only the covariance matrix is returned.
        If True, then correlation matrix and standard deviation are
        additionally returned.

    Returns
    -------
    cov_ : ndarray
        corrected covariance matrix
    corr_ : ndarray, (optional)
        corrected correlation matrix
    std_ : ndarray, (optional)
        standard deviation


    Notes
    -----
    This converts the covariance matrix to a correlation matrix. Then, finds
    the nearest correlation matrix that is positive semidefinite and converts
    it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input covariance matrix is symmetric.

    See Also
    --------
    corr_nearest
    corr_clipped
    """

    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    cov_, std_ = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    else:  # method == 'nearest'
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)

    cov_ = corr2cov(corr_, std_)

    if return_all:
        return cov_, corr_, std_
    else:
        return cov_


def _nmono_linesearch(obj, grad, x, d, obj_hist, M=10, sig1=0.1,
                      sig2=0.9, gam=1e-4, maxiter=100):
    """
    Implements the non-monotone line search of Grippo et al. (1986),
    as described in Birgin, Martinez and Raydan (2013).

    Parameters
    ----------
    obj : real-valued function
        The objective function, to be minimized
    grad : vector-valued function
        The gradient of the objective function
    x : array_like
        The starting point for the line search
    d : array_like
        The search direction
    obj_hist : array_like
        Objective function history (must contain at least one value)
    M : positive int
        Number of previous function points to consider (see references
        for details).
    sig1 : real
        Tuning parameter, see references for details.
    sig2 : real
        Tuning parameter, see references for details.
    gam : real
        Tuning parameter, see references for details.
    maxiter : int
        The maximum number of iterations; returns Nones if convergence
        does not occur by this point

    Returns
    -------
    alpha : real
        The step value
    x : Array_like
        The function argument at the final step
    obval : Real
        The function value at the final step
    g : Array_like
        The gradient at the final step

    Notes
    -----
    The basic idea is to take a big step in the direction of the
    gradient, even if the function value is not decreased (but there
    is a maximum allowed increase in terms of the recent history of
    the iterates).

    References
    ----------
    Grippo L, Lampariello F, Lucidi S (1986). A Nonmonotone Line
    Search Technique for Newton's Method. SIAM Journal on Numerical
    Analysis, 23, 707-716.

    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).
    """

    alpha = 1.
    last_obval = obj(x)
    obj_max = max(obj_hist[-M:])

    for iter in range(maxiter):

        obval = obj(x + alpha*d)
        g = grad(x)
        gtd = (g * d).sum()

        if obval <= obj_max + gam*alpha*gtd:
            return alpha, x + alpha*d, obval, g

        a1 = -0.5*alpha**2*gtd / (obval - last_obval - alpha*gtd)

        if (sig1 <= a1) and (a1 <= sig2*alpha):
            alpha = a1
        else:
            alpha /= 2.

        last_obval = obval

    return None, None, None, None


def _spg_optim(func, grad, start, project, maxiter=1e4, M=10,
               ctol=1e-3, maxiter_nmls=200, lam_min=1e-30,
               lam_max=1e30, sig1=0.1, sig2=0.9, gam=1e-4):
    """
    Implements the spectral projected gradient method for minimizing a
    differentiable function on a convex domain.

    Parameters
    ----------
    func : real valued function
        The objective function to be minimized.
    grad : real array-valued function
        The gradient of the objective function
    start : array_like
        The starting point
    project : function
        In-place projection of the argument to the domain
        of func.
    ... See notes regarding additional arguments

    Returns
    -------
    rslt : Bunch
        rslt.params is the final iterate, other fields describe
        convergence status.

    Notes
    -----
    This can be an effective heuristic algorithm for problems where no
    guaranteed algorithm for computing a global minimizer is known.

    There are a number of tuning parameters, but these generally
    should not be changed except for `maxiter` (positive integer) and
    `ctol` (small positive real).  See the Birgin et al reference for
    more information about the tuning parameters.

    Reference
    ---------
    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).  Available at:
    http://www.ime.usp.br/~egbirgin/publications/bmr5.pdf
    """

    lam = min(10*lam_min, lam_max)

    params = start.copy()
    gval = grad(params)

    obj_hist = [func(params), ]

    for itr in range(int(maxiter)):

        # Check convergence
        df = params - gval
        project(df)
        df -= params
        if np.max(np.abs(df)) < ctol:
            return Bunch(**{"Converged": True, "params": params,
                            "objective_values": obj_hist,
                            "Message": "Converged successfully"})

        # The line search direction
        d = params - lam*gval
        project(d)
        d -= params

        # Carry out the nonmonotone line search
        alpha, params1, fval, gval1 = _nmono_linesearch(
            func,
            grad,
            params,
            d,
            obj_hist,
            M=M,
            sig1=sig1,
            sig2=sig2,
            gam=gam,
            maxiter=maxiter_nmls)

        if alpha is None:
            return Bunch(**{"Converged": False, "params": params,
                            "objective_values": obj_hist,
                            "Message": "Failed in nmono_linesearch"})

        obj_hist.append(fval)
        s = params1 - params
        y = gval1 - gval

        sy = (s*y).sum()
        if sy <= 0:
            lam = lam_max
        else:
            ss = (s*s).sum()
            lam = max(lam_min, min(ss/sy, lam_max))

        params = params1
        gval = gval1

    return Bunch(**{"Converged": False, "params": params,
                    "objective_values": obj_hist,
                    "Message": "spg_optim did not converge"})


def _project_correlation_factors(X):
    """
    Project a matrix into the domain of matrices whose row-wise sums
    of squares are less than or equal to 1.

    The input matrix is modified in-place.
    """
    nm = np.sqrt((X*X).sum(1))
    ii = np.flatnonzero(nm > 1)
    if len(ii) > 0:
        X[ii, :] /= nm[ii][:, None]


class FactoredPSDMatrix:
    """
    Representation of a positive semidefinite matrix in factored form.

    The representation is constructed based on a vector `diag` and
    rectangular matrix `root`, such that the PSD matrix represented by
    the class instance is Diag + root * root', where Diag is the
    square diagonal matrix with `diag` on its main diagonal.

    Parameters
    ----------
    diag : 1d array_like
        See above
    root : 2d array_like
        See above

    Notes
    -----
    The matrix is represented internally in the form Diag^{1/2}(I +
    factor * scales * factor')Diag^{1/2}, where `Diag` and `scales`
    are diagonal matrices, and `factor` is an orthogonal matrix.
    """

    def __init__(self, diag, root):
        self.diag = diag
        self.root = root
        root = root / np.sqrt(diag)[:, None]
        u, s, vt = np.linalg.svd(root, 0)
        self.factor = u
        self.scales = s**2

    def to_matrix(self):
        """
        Returns the PSD matrix represented by this instance as a full
        (square) matrix.
        """
        return np.diag(self.diag) + np.dot(self.root, self.root.T)

    def decorrelate(self, rhs):
        """
        Decorrelate the columns of `rhs`.

        Parameters
        ----------
        rhs : array_like
            A 2 dimensional array with the same number of rows as the
            PSD matrix represented by the class instance.

        Returns
        -------
        C^{-1/2} * rhs, where C is the covariance matrix represented
        by this class instance.

        Notes
        -----
        The returned matrix has the identity matrix as its row-wise
        population covariance matrix.

        This function exploits the factor structure for efficiency.
        """

        # I + factor * qval * factor' is the inverse square root of
        # the covariance matrix in the homogeneous case where diag =
        # 1.
        qval = -1 + 1 / np.sqrt(1 + self.scales)

        # Decorrelate in the general case.
        rhs = rhs / np.sqrt(self.diag)[:, None]
        rhs1 = np.dot(self.factor.T, rhs)
        rhs1 *= qval[:, None]
        rhs1 = np.dot(self.factor, rhs1)
        rhs += rhs1

        return rhs

    def solve(self, rhs):
        """
        Solve a linear system of equations with factor-structured
        coefficients.

        Parameters
        ----------
        rhs : array_like
            A 2 dimensional array with the same number of rows as the
            PSD matrix represented by the class instance.

        Returns
        -------
        C^{-1} * rhs, where C is the covariance matrix represented
        by this class instance.

        Notes
        -----
        This function exploits the factor structure for efficiency.
        """

        qval = -self.scales / (1 + self.scales)
        dr = np.sqrt(self.diag)
        rhs = rhs / dr[:, None]
        mat = qval[:, None] * np.dot(self.factor.T, rhs)
        rhs = rhs + np.dot(self.factor, mat)
        return rhs / dr[:, None]

    def logdet(self):
        """
        Returns the logarithm of the determinant of a
        factor-structured matrix.
        """

        logdet = np.sum(np.log(self.diag))
        logdet += np.sum(np.log(self.scales))
        logdet += np.sum(np.log(1 + 1 / self.scales))

        return logdet


def corr_nearest_factor(corr, rank, ctol=1e-6, lam_min=1e-30,
                        lam_max=1e30, maxiter=1000):
    """
    Find the nearest correlation matrix with factor structure to a
    given square matrix.

    Parameters
    ----------
    corr : square array
        The target matrix (to which the nearest correlation matrix is
        sought).  Must be square, but need not be positive
        semidefinite.
    rank : int
        The rank of the factor structure of the solution, i.e., the
        number of linearly independent columns of X.
    ctol : positive real
        Convergence criterion.
    lam_min : float
        Tuning parameter for spectral projected gradient optimization
        (smallest allowed step in the search direction).
    lam_max : float
        Tuning parameter for spectral projected gradient optimization
        (largest allowed step in the search direction).
    maxiter : int
        Maximum number of iterations in spectral projected gradient
        optimization.

    Returns
    -------
    rslt : Bunch
        rslt.corr is a FactoredPSDMatrix defining the estimated
        correlation structure.  Other fields of `rslt` contain
        returned values from spg_optim.

    Notes
    -----
    A correlation matrix has factor structure if it can be written in
    the form I + XX' - diag(XX'), where X is n x k with linearly
    independent columns, and with each row having sum of squares at
    most equal to 1.  The approximation is made in terms of the
    Frobenius norm.

    This routine is useful when one has an approximate correlation
    matrix that is not positive semidefinite, and there is need to
    estimate the inverse, square root, or inverse square root of the
    population correlation matrix.  The factor structure allows these
    tasks to be done without constructing any n x n matrices.

    This is a non-convex problem with no known guaranteed globally
    convergent algorithm for computing the solution.  Borsdof, Higham
    and Raydan (2010) compared several methods for this problem and
    found the spectral projected gradient (SPG) method (used here) to
    perform best.

    The input matrix `corr` can be a dense numpy array or any scipy
    sparse matrix.  The latter is useful if the input matrix is
    obtained by thresholding a very large sample correlation matrix.
    If `corr` is sparse, the calculations are optimized to save
    memory, so no working matrix with more than 10^6 elements is
    constructed.

    References
    ----------
    .. [*] R Borsdof, N Higham, M Raydan (2010).  Computing a nearest
       correlation matrix with factor structure. SIAM J Matrix Anal Appl,
       31:5, 2603-2622.
       http://eprints.ma.man.ac.uk/1523/01/covered/MIMS_ep2009_87.pdf

    Examples
    --------
    Hard thresholding a correlation matrix may result in a matrix that
    is not positive semidefinite.  We can approximate a hard
    thresholded correlation matrix with a PSD matrix as follows, where
    `corr` is the input correlation matrix.

    >>> import numpy as np
    >>> from statsmodels.stats.correlation_tools import corr_nearest_factor
    >>> np.random.seed(1234)
    >>> b = 1.5 - np.random.rand(10, 1)
    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)
    >>> corr = np.corrcoef(x.T)
    >>> corr = corr * (np.abs(corr) >= 0.3)
    >>> rslt = corr_nearest_factor(corr, 3)
    """

    p, _ = corr.shape

    # Starting values (following the PCA method in BHR).
    u, s, vt = svds(corr, rank)
    X = u * np.sqrt(s)
    nm = np.sqrt((X**2).sum(1))
    ii = np.flatnonzero(nm > 1e-5)
    X[ii, :] /= nm[ii][:, None]

    # Zero the diagonal
    corr1 = corr.copy()
    if type(corr1) is np.ndarray:
        np.fill_diagonal(corr1, 0)
    elif sparse.issparse(corr1):
        corr1.setdiag(np.zeros(corr1.shape[0]))
        corr1.eliminate_zeros()
        corr1.sort_indices()
    else:
        raise ValueError("Matrix type not supported")

    # The gradient, from lemma 4.1 of BHR.
    def grad(X):
        gr = np.dot(X, np.dot(X.T, X))
        if type(corr1) is np.ndarray:
            gr -= np.dot(corr1, X)
        else:
            gr -= corr1.dot(X)
        gr -= (X*X).sum(1)[:, None] * X
        return 4*gr

    # The objective function (sum of squared deviations between fitted
    # and observed arrays).
    def func(X):
        if type(corr1) is np.ndarray:
            M = np.dot(X, X.T)
            np.fill_diagonal(M, 0)
            M -= corr1
            fval = (M*M).sum()
            return fval
        else:
            fval = 0.
            # Control the size of intermediates
            max_ws = 1e6
            bs = int(max_ws / X.shape[0])
            ir = 0
            while ir < X.shape[0]:
                ir2 = min(ir+bs, X.shape[0])
                u = np.dot(X[ir:ir2, :], X.T)
                ii = np.arange(u.shape[0])
                u[ii, ir+ii] = 0
                u -= np.asarray(corr1[ir:ir2, :].todense())
                fval += (u*u).sum()
                ir += bs
            return fval

    rslt = _spg_optim(func, grad, X, _project_correlation_factors, ctol=ctol,
                      lam_min=lam_min, lam_max=lam_max, maxiter=maxiter)
    root = rslt.params
    diag = 1 - (root**2).sum(1)
    soln = FactoredPSDMatrix(diag, root)
    rslt.corr = soln
    del rslt.params
    return rslt


def cov_nearest_factor_homog(cov, rank):
    """
    Approximate an arbitrary square matrix with a factor-structured
    matrix of the form k*I + XX'.

    Parameters
    ----------
    cov : array_like
        The input array, must be square but need not be positive
        semidefinite
    rank : int
        The rank of the fitted factor structure

    Returns
    -------
    A FactoredPSDMatrix instance containing the fitted matrix

    Notes
    -----
    This routine is useful if one has an estimated covariance matrix
    that is not SPD, and the ultimate goal is to estimate the inverse,
    square root, or inverse square root of the true covariance
    matrix. The factor structure allows these tasks to be performed
    without constructing any n x n matrices.

    The calculations use the fact that if k is known, then X can be
    determined from the eigen-decomposition of cov - k*I, which can
    in turn be easily obtained form the eigen-decomposition of `cov`.
    Thus the problem can be reduced to a 1-dimensional search for k
    that does not require repeated eigen-decompositions.

    If the input matrix is sparse, then cov - k*I is also sparse, so
    the eigen-decomposition can be done efficiently using sparse
    routines.

    The one-dimensional search for the optimal value of k is not
    convex, so a local minimum could be obtained.

    Examples
    --------
    Hard thresholding a covariance matrix may result in a matrix that
    is not positive semidefinite.  We can approximate a hard
    thresholded covariance matrix with a PSD matrix as follows:

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> b = 1.5 - np.random.rand(10, 1)
    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)
    >>> cov = np.cov(x)
    >>> cov = cov * (np.abs(cov) >= 0.3)
    >>> rslt = cov_nearest_factor_homog(cov, 3)
    """

    m, n = cov.shape

    Q, Lambda, _ = svds(cov, rank)

    if sparse.issparse(cov):
        QSQ = np.dot(Q.T, cov.dot(Q))
        ts = cov.diagonal().sum()
        tss = cov.dot(cov).diagonal().sum()
    else:
        QSQ = np.dot(Q.T, np.dot(cov, Q))
        ts = np.trace(cov)
        tss = np.trace(np.dot(cov, cov))

    def fun(k):
        Lambda_t = Lambda - k
        v = tss + m*(k**2) + np.sum(Lambda_t**2) - 2*k*ts
        v += 2*k*np.sum(Lambda_t) - 2*np.sum(np.diag(QSQ) * Lambda_t)
        return v

    # Get the optimal decomposition
    k_opt = fminbound(fun, 0, 1e5)
    Lambda_opt = Lambda - k_opt
    fac_opt = Q * np.sqrt(Lambda_opt)

    diag = k_opt * np.ones(m, dtype=np.float64)  # - (fac_opt**2).sum(1)
    return FactoredPSDMatrix(diag, fac_opt)


def corr_thresholded(data, minabs=None, max_elt=1e7):
    r"""
    Construct a sparse matrix containing the thresholded row-wise
    correlation matrix from a data array.

    Parameters
    ----------
    data : array_like
        The data from which the row-wise thresholded correlation
        matrix is to be computed.
    minabs : non-negative real
        The threshold value; correlation coefficients smaller in
        magnitude than minabs are set to zero.  If None, defaults
        to 1 / sqrt(n), see Notes for more information.

    Returns
    -------
    cormat : sparse.coo_matrix
        The thresholded correlation matrix, in COO format.

    Notes
    -----
    This is an alternative to C = np.corrcoef(data); C \*= (np.abs(C)
    >= absmin), suitable for very tall data matrices.

    If the data are jointly Gaussian, the marginal sampling
    distributions of the elements of the sample correlation matrix are
    approximately Gaussian with standard deviation 1 / sqrt(n).  The
    default value of ``minabs`` is thus equal to 1 standard error, which
    will set to zero approximately 68% of the estimated correlation
    coefficients for which the population value is zero.

    No intermediate matrix with more than ``max_elt`` values will be
    constructed.  However memory use could still be high if a large
    number of correlation values exceed `minabs` in magnitude.

    The thresholded matrix is returned in COO format, which can easily
    be converted to other sparse formats.

    Examples
    --------
    Here X is a tall data matrix (e.g. with 100,000 rows and 50
    columns).  The row-wise correlation matrix of X is calculated
    and stored in sparse form, with all entries smaller than 0.3
    treated as 0.

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> b = 1.5 - np.random.rand(10, 1)
    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)
    >>> cmat = corr_thresholded(x, 0.3)
    """

    nrow, ncol = data.shape

    if minabs is None:
        minabs = 1. / float(ncol)

    # Row-standardize the data
    data = data.copy()
    data -= data.mean(1)[:, None]
    sd = data.std(1, ddof=1)
    ii = np.flatnonzero(sd > 1e-5)
    data[ii, :] /= sd[ii][:, None]
    ii = np.flatnonzero(sd <= 1e-5)
    data[ii, :] = 0

    # Number of rows to process in one pass
    bs = int(np.floor(max_elt / nrow))

    ipos_all, jpos_all, cor_values = [], [], []

    ir = 0
    while ir < nrow:
        ir2 = min(data.shape[0], ir + bs)
        cm = np.dot(data[ir:ir2, :], data.T) / (ncol - 1)
        cma = np.abs(cm)
        ipos, jpos = np.nonzero(cma >= minabs)
        ipos_all.append(ipos + ir)
        jpos_all.append(jpos)
        cor_values.append(cm[ipos, jpos])
        ir += bs

    ipos = np.concatenate(ipos_all)
    jpos = np.concatenate(jpos_all)
    cor_values = np.concatenate(cor_values)

    cmat = sparse.coo_matrix((cor_values, (ipos, jpos)), (nrow, nrow))

    return cmat


class MultivariateKernel:
    """
    Base class for multivariate kernels.

    An instance of MultivariateKernel implements a `call` method having
    signature `call(x, loc)`, returning the kernel weights comparing `x`
    (a 1d ndarray) to each row of `loc` (a 2d ndarray).
    """

    def call(self, x, loc):
        raise NotImplementedError

    def set_bandwidth(self, bw):
        """
        Set the bandwidth to the given vector.

        Parameters
        ----------
        bw : array_like
            A vector of non-negative bandwidth values.
        """

        self.bw = bw
        self._setup()

    def _setup(self):

        # Precompute the squared bandwidth values.
        self.bwk = np.prod(self.bw)
        self.bw2 = self.bw * self.bw

    def set_default_bw(self, loc, bwm=None):
        """
        Set default bandwiths based on domain values.

        Parameters
        ----------
        loc : array_like
            Values from the domain to which the kernel will
            be applied.
        bwm : scalar, optional
            A non-negative scalar that is used to multiply
            the default bandwidth.
        """

        sd = loc.std(0)
        q25, q75 = np.percentile(loc, [25, 75], axis=0)
        iqr = (q75 - q25) / 1.349
        bw = np.where(iqr < sd, iqr, sd)
        bw *= 0.9 / loc.shape[0] ** 0.2

        if bwm is not None:
            bw *= bwm

        # The final bandwidths
        self.bw = np.asarray(bw, dtype=np.float64)

        self._setup()


class GaussianMultivariateKernel(MultivariateKernel):
    """
    The Gaussian (squared exponential) multivariate kernel.
    """

    def call(self, x, loc):
        return np.exp(-(x - loc)**2 / (2 * self.bw2)).sum(1) / self.bwk


def kernel_covariance(exog, loc, groups, kernel=None, bw=None):
    """
    Use kernel averaging to estimate a multivariate covariance function.

    The goal is to estimate a covariance function C(x, y) =
    cov(Z(x), Z(y)) where x, y are vectors in R^p (e.g. representing
    locations in time or space), and Z(.) represents a multivariate
    process on R^p.

    The data used for estimation can be observed at arbitrary values of the
    position vector, and there can be multiple independent observations
    from the process.

    Parameters
    ----------
    exog : array_like
        The rows of exog are realizations of the process obtained at
        specified points.
    loc : array_like
        The rows of loc are the locations (e.g. in space or time) at
        which the rows of exog are observed.
    groups : array_like
        The values of groups are labels for distinct independent copies
        of the process.
    kernel : MultivariateKernel instance, optional
        An instance of MultivariateKernel, defaults to
        GaussianMultivariateKernel.
    bw : array_like or scalar
        A bandwidth vector, or bandwidth multiplier.  If a 1d array, it
        contains kernel bandwidths for each component of the process, and
        must have length equal to the number of columns of exog.  If a scalar,
        bw is a bandwidth multiplier used to adjust the default bandwidth; if
        None, a default bandwidth is used.

    Returns
    -------
    A real-valued function C(x, y) that returns an estimate of the covariance
    between values of the process located at x and y.

    References
    ----------
    .. [1] Genton M, W Kleiber (2015).  Cross covariance functions for
        multivariate geostatics.  Statistical Science 30(2).
        https://arxiv.org/pdf/1507.08017.pdf
    """

    exog = np.asarray(exog)
    loc = np.asarray(loc)
    groups = np.asarray(groups)

    if loc.ndim == 1:
        loc = loc[:, None]

    v = [exog.shape[0], loc.shape[0], len(groups)]
    if min(v) != max(v):
        msg = "exog, loc, and groups must have the same number of rows"
        raise ValueError(msg)

    # Map from group labels to the row indices in each group.
    ix = {}
    for i, g in enumerate(groups):
        if g not in ix:
            ix[g] = []
        ix[g].append(i)
    for g in ix.keys():
        ix[g] = np.sort(ix[g])

    if kernel is None:
        kernel = GaussianMultivariateKernel()

    if bw is None:
        kernel.set_default_bw(loc)
    elif np.isscalar(bw):
        kernel.set_default_bw(loc, bwm=bw)
    else:
        kernel.set_bandwidth(bw)

    def cov(x, y):

        kx = kernel.call(x, loc)
        ky = kernel.call(y, loc)

        cm, cw = 0., 0.

        for g, ii in ix.items():

            m = len(ii)
            j1, j2 = np.indices((m, m))
            j1 = ii[j1.flat]
            j2 = ii[j2.flat]
            w = kx[j1] * ky[j2]

            # TODO: some other form of broadcasting may be faster than
            # einsum here
            cm += np.einsum("ij,ik,i->jk", exog[j1, :], exog[j2, :], w)
            cw += w.sum()

        if cw < 1e-10:
            msg = ("Effective sample size is 0.  The bandwidth may be too " +
                   "small, or you are outside the range of your data.")
            warnings.warn(msg)
            return np.nan * np.ones_like(cm)

        return cm / cw

    return cov
