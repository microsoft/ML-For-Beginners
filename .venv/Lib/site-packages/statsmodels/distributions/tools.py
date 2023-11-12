# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:19:30 2021

Author: Josef Perktold
License: BSD-3

"""
import warnings

import numpy as np
from scipy import interpolate, stats

# helper functions to work on a grid of cdf and pdf, histogram

class _Grid:
    """Create Grid values and indices, grid in [0, 1]^d

    This class creates a regular grid in a d dimensional hyper cube.

    Intended for internal use, implementation might change without warning.


    Parameters
    ----------
    k_grid : tuple or array_like
        number of elements for axes, this defines k_grid - 1 equal sized
        intervals of [0, 1] for each axis.
    eps : float
        If eps is not zero, then x values will be clipped to [eps, 1 - eps],
        i.e. to the interior of the unit interval or hyper cube.


    Attributes
    ----------
    k_grid : list of number of grid points
    x_marginal: list of 1-dimensional marginal values
    idx_flat: integer array with indices
    x_flat: flattened grid values,
        rows are grid points, columns represent variables or axis.
        ``x_flat`` is currently also 2-dim in the univariate 1-dim grid case.

    """

    def __init__(self, k_grid, eps=0):
        self.k_grid = k_grid

        x_marginal = [np.arange(ki) / (ki - 1) for ki in k_grid]

        idx_flat = np.column_stack(
                np.unravel_index(np.arange(np.prod(k_grid)), k_grid)
                ).astype(float)
        x_flat = idx_flat / idx_flat.max(0)
        if eps != 0:
            x_marginal = [np.clip(xi, eps, 1 - eps) for xi in x_marginal]
            x_flat = np.clip(x_flat, eps, 1 - eps)

        self.x_marginal = x_marginal
        self.idx_flat = idx_flat
        self.x_flat = x_flat


def prob2cdf_grid(probs):
    """Cumulative probabilities from cell provabilites on a grid

    Parameters
    ----------
    probs : array_like
        Rectangular grid of cell probabilities.

    Returns
    -------
    cdf : ndarray
        Grid of cumulative probabilities with same shape as probs.
    """
    cdf = np.asarray(probs).copy()
    k = cdf.ndim
    for i in range(k):
        cdf = cdf.cumsum(axis=i)

    return cdf


def cdf2prob_grid(cdf, prepend=0):
    """Cell probabilities from cumulative probabilities on a grid.

    Parameters
    ----------
    cdf : array_like
        Grid of cumulative probabilities with same shape as probs.

    Returns
    -------
    probs : ndarray
        Rectangular grid of cell probabilities.

    """
    if prepend is None:
        prepend = np._NoValue
    prob = np.asarray(cdf).copy()
    k = prob.ndim
    for i in range(k):
        prob = np.diff(prob, prepend=prepend, axis=i)

    return prob


def average_grid(values, coords=None, _method="slicing"):
    """Compute average for each cell in grid using endpoints

    Parameters
    ----------
    values : array_like
        Values on a grid that will average over corner points of each cell.
    coords : None or list of array_like
        Grid coordinates for each axis use to compute volumne of cell.
        If None, then averaged values are not rescaled.
    _method : {"slicing", "convolve"}
        Grid averaging is implemented using numpy "slicing" or using
        scipy.signal "convolve".

    Returns
    -------
    Grid with averaged cell values.
    """
    k_dim = values.ndim
    if _method == "slicing":
        p = values.copy()

        for d in range(k_dim):
            # average (p[:-1] + p[1:]) / 2 over each axis
            sl1 = [slice(None, None, None)] * k_dim
            sl2 = [slice(None, None, None)] * k_dim
            sl1[d] = slice(None, -1, None)
            sl2[d] = slice(1, None, None)
            sl1 = tuple(sl1)
            sl2 = tuple(sl2)

            p = (p[sl1] + p[sl2]) / 2

    elif _method == "convolve":
        from scipy import signal
        p = signal.convolve(values, 0.5**k_dim * np.ones([2] * k_dim),
                            mode="valid")

    if coords is not None:
        dx = np.array(1)
        for d in range(k_dim):
            dx = dx[..., None] * np.diff(coords[d])

        p = p * dx

    return p


def nearest_matrix_margins(mat, maxiter=100, tol=1e-8):
    """nearest matrix with uniform margins

    Parameters
    ----------
    mat : array_like, 2-D
        Matrix that will be converted to have uniform margins.
        Currently, `mat` has to be two dimensional.
    maxiter : in
        Maximum number of iterations.
    tol : float
        Tolerance for convergence, defined for difference between largest and
        smallest margin in each dimension.

    Returns
    -------
    ndarray, nearest matrix with uniform margins.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.

    changed in 0.14 to support k_dim > 2.


    """
    pc = np.asarray(mat)
    converged = False

    for _ in range(maxiter):
        pc0 = pc.copy()
        for ax in range(pc.ndim):
            axs = tuple([i for i in range(pc.ndim) if not i == ax])
            pc0 /= pc.sum(axis=axs, keepdims=True)
        pc = pc0
        pc /= pc.sum()

        # check convergence
        mptps = []
        for ax in range(pc.ndim):
            axs = tuple([i for i in range(pc.ndim) if not i == ax])
            marg = pc.sum(axis=axs, keepdims=False)
            mptps.append(np.ptp(marg))
        if max(mptps) < tol:
            converged = True
            break

    if not converged:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.warn("Iterations did not converge, maxiter reached",
                      ConvergenceWarning)
    return pc


def _rankdata_no_ties(x):
    """rankdata without ties for 2-d array

    This is a simplified version for ranking data if there are no ties.
    Works vectorized across columns.

    See Also
    --------
    scipy.stats.rankdata

    """
    nobs, k_vars = x.shape
    ranks = np.ones((nobs, k_vars))
    sidx = np.argsort(x, axis=0)
    ranks[sidx, np.arange(k_vars)] = np.arange(1, nobs + 1)[:, None]
    return ranks


def frequencies_fromdata(data, k_bins, use_ranks=True):
    """count of observations in bins (histogram)

    currently only for bivariate data

    Parameters
    ----------
    data : array_like
        Bivariate data with observations in rows and two columns. Binning is
        in unit rectangle [0, 1]^2. If use_rank is False, then data should be
        in unit interval.
    k_bins : int
        Number of bins along each dimension in the histogram
    use_ranks : bool
        If use_rank is True, then data will be converted to ranks without
        tie handling.

    Returns
    -------
    bin counts : ndarray
        Frequencies are the number of observations in a given bin.
        Bin counts are a 2-dim array with k_bins rows and k_bins columns.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    data = np.asarray(data)
    k_dim = data.shape[-1]
    k = k_bins + 1
    g2 = _Grid([k] * k_dim, eps=0)
    if use_ranks:
        data = _rankdata_no_ties(data) / (data.shape[0] + 1)
        # alternatives: scipy handles ties, but uses np.apply_along_axis
        # rvs = stats.rankdata(rvs, axis=0) / (rvs.shape[0] + 1)
        # rvs = (np.argsort(np.argsort(rvs, axis=0), axis=0) + 1
        #                              ) / (rvs.shape[0] + 1)
    freqr, _ = np.histogramdd(data, bins=g2.x_marginal)
    return freqr


def approx_copula_pdf(copula, k_bins=10, force_uniform=True, use_pdf=False):
    """Histogram probabilities as approximation to a copula density.

    Parameters
    ----------
    copula : instance
        Instance of a copula class. Only the ``pdf`` method is used.
    k_bins : int
        Number of bins along each dimension in the approximating histogram.
    force_uniform : bool
        If true, then the pdf grid will be adjusted to have uniform margins
        using `nearest_matrix_margin`.
        If false, then no adjustment is done and the margins may not be exactly
        uniform.
    use_pdf : bool
        If false, then the grid cell probabilities will be computed from the
        copula cdf.
        If true, then the density, ``pdf``, is used and cell probabilities
        are approximated by averaging the pdf of the cell corners. This is
        only useful if the cdf is not available.

    Returns
    -------
    bin probabilites : ndarray
        Probability that random variable falls in given bin. This corresponds
        to a discrete distribution, and is not scaled to bin size to form a
        piecewise uniform, histogram density.
        Bin probabilities are a k-dim array with k_bins segments in each
        dimensionrows.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    k_dim = copula.k_dim
    k = k_bins + 1
    ks = tuple([k] * k_dim)

    if use_pdf:
        g = _Grid([k] * k_dim, eps=0.1 / k_bins)
        pdfg = copula.pdf(g.x_flat).reshape(*ks)
        # correct for bin size
        pdfg *= 1 / k**k_dim
        ag = average_grid(pdfg)
        if force_uniform:
            pdf_grid = nearest_matrix_margins(ag, maxiter=100, tol=1e-8)
        else:
            pdf_grid = ag / ag.sum()
    else:
        g = _Grid([k] * k_dim, eps=1e-6)
        cdfg = copula.cdf(g.x_flat).reshape(*ks)
        # correct for bin size
        pdf_grid = cdf2prob_grid(cdfg, prepend=None)
        # TODO: check boundary approximation, eg. undefined at zero
        # for now just normalize
        pdf_grid /= pdf_grid.sum()

    return pdf_grid


# functions to evaluate bernstein polynomials

def _eval_bernstein_1d(x, fvals, method="binom"):
    """Evaluate 1-dimensional bernstein polynomial given grid of values.

    experimental, comparing methods

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.
    method: "binom", "beta" or "bpoly"
        Method to construct Bernstein polynomial basis, used for comparison
        of parameterizations.

        - "binom" uses pmf of Binomial distribution
        - "beta" uses pdf of Beta distribution
        - "bpoly" uses one interval in scipy.interpolate.BPoly

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    k_terms = fvals.shape[-1]
    xx = np.asarray(x)
    k = np.arange(k_terms).astype(float)
    n = k_terms - 1.

    if method.lower() == "binom":
        # Divide by 0 RuntimeWarning here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poly_base = stats.binom.pmf(k, n, xx[..., None])
        bp_values = (fvals * poly_base).sum(-1)
    elif method.lower() == "bpoly":
        bpb = interpolate.BPoly(fvals[:, None], [0., 1])
        bp_values = bpb(x)
    elif method.lower() == "beta":
        # Divide by 0 RuntimeWarning here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poly_base = stats.beta.pdf(xx[..., None], k + 1, n - k + 1) / (n + 1)
        bp_values = (fvals * poly_base).sum(-1)
    else:
        raise ValueError("method not recogized")

    return bp_values


def _eval_bernstein_2d(x, fvals):
    """Evaluate 2-dimensional bernstein polynomial given grid of values

    experimental

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    k_terms = fvals.shape
    k_dim = fvals.ndim
    if k_dim != 2:
        raise ValueError("`fval` needs to be 2-dimensional")
    xx = np.atleast_2d(x)
    if xx.shape[1] != 2:
        raise ValueError("x needs to be bivariate and have 2 columns")

    x1, x2 = xx.T
    n1, n2 = k_terms[0] - 1, k_terms[1] - 1
    k1 = np.arange(k_terms[0]).astype(float)
    k2 = np.arange(k_terms[1]).astype(float)

    # we are building a nobs x n1 x n2 array
    poly_base = (stats.binom.pmf(k1[None, :, None], n1, x1[:, None, None]) *
                 stats.binom.pmf(k2[None, None, :], n2, x2[:, None, None]))
    bp_values = (fvals * poly_base).sum(-1).sum(-1)

    return bp_values


def _eval_bernstein_dd(x, fvals):
    """Evaluate d-dimensional bernstein polynomial given grid of valuesv

    experimental

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the Bernstein polynomial.
    fvals : ndarray
        Grid values of coefficients for Bernstein polynomial basis in the
        weighted sum.

    Returns
    -------
    Bernstein polynomial at evaluation points, weighted sum of Bernstein
    polynomial basis.
    """
    k_terms = fvals.shape
    k_dim = fvals.ndim
    xx = np.atleast_2d(x)

    # The following loop is a tricky
    # we add terms for each x and expand dimension of poly base in each
    # iteration using broadcasting

    poly_base = np.zeros(x.shape[0])
    for i in range(k_dim):
        ki = np.arange(k_terms[i]).astype(float)
        for _ in range(i+1):
            ki = ki[..., None]
        ni = k_terms[i] - 1
        xi = xx[:, i]
        poly_base = poly_base[None, ...] + stats.binom._logpmf(ki, ni, xi)

    poly_base = np.exp(poly_base)
    bp_values = fvals.T[..., None] * poly_base

    for i in range(k_dim):
        bp_values = bp_values.sum(0)

    return bp_values


def _ecdf_mv(data, method="seq", use_ranks=True):
    """
    Multivariate empiricial distribution function, empirical copula


    Notes
    -----
    Method "seq" is faster than method "brute", but supports mainly bivariate
    case. Speed advantage of "seq" is increasing in number of observations
    and decreasing in number of variables.
    (see Segers ...)

    Warning: This does not handle ties. The ecdf is based on univariate ranks
    without ties. The assignment of ranks to ties depends on the sorting
    algorithm and the initial ordering of the data.

    When the original data is used instead of ranks, then method "brute"
    computes the correct ecdf counts even in the case of ties.

    """
    x = np.asarray(data)
    n = x.shape[0]
    if use_ranks:
        x = _rankdata_no_ties(x) / n
    if method == "brute":
        count = [((x <= x[i]).all(1)).sum() for i in range(n)]
        count = np.asarray(count)
    elif method.startswith("seq"):
        sort_idx0 = np.argsort(x[:, 0])
        x_s0 = x[sort_idx0]
        x1 = x_s0[:, 1:]
        count_smaller = [(x1[:i] <= x1[i]).all(1).sum() + 1 for i in range(n)]
        count = np.empty(x.shape[0])
        count[sort_idx0] = count_smaller
    else:
        raise ValueError("method not available")

    return count, x
