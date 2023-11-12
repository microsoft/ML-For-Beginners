# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:35:23 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats

from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
        _Grid, cdf2prob_grid, prob2cdf_grid,
        _eval_bernstein_dd, _eval_bernstein_2d, _eval_bernstein_1d)


class BernsteinDistribution:
    """Distribution based on Bernstein Polynomials on unit hypercube.

    Parameters
    ----------
    cdf_grid : array_like
        cdf values on a equal spaced grid of the unit hypercube [0, 1]^d.
        The dimension of the arrays define how many random variables are
        included in the multivariate distribution.

    Attributes
    ----------
    cdf_grid : grid of cdf values
    prob_grid : grid of cell or bin probabilities
    k_dim : (int) number of components, dimension of random variable
    k_grid : (tuple) shape of cdf_grid
    k_grid_product : (int) total number of bins in grid
    _grid : Grid instance with helper methods and attributes
    """

    def __init__(self, cdf_grid):
        self.cdf_grid = cdf_grid = np.asarray(cdf_grid)
        self.k_dim = cdf_grid.ndim
        self.k_grid = cdf_grid.shape
        self.k_grid_product = np.prod([i-1 for i in self.k_grid])
        self._grid = _Grid(self.k_grid)

    @classmethod
    def from_data(cls, data, k_bins):
        """Create distribution instance from data using histogram binning.

        Classmethod to construct a distribution instance.

        Parameters
        ----------
        data : array_like
            Data with observation in rows and random variables in columns.
            Data can be 1-dimensional in the univariate case.
        k_bins : int or list
            Number or edges of bins to be used in numpy histogramdd.
            If k_bins is a scalar int, then the number of bins of each
            component will be equal to it.

        Returns
        -------
        Instance of a Bernstein distribution
        """
        data = np.asarray(data)
        if np.any(data < 0) or np.any(data > 1):
            raise ValueError("data needs to be in [0, 1]")

        if data.ndim == 1:
            data = data[:, None]

        k_dim = data.shape[1]
        if np.size(k_bins) == 1:
            k_bins = [k_bins] * k_dim
        bins = [np.linspace(-1 / ni, 1, ni + 2) for ni in k_bins]
        c, e = np.histogramdd(data, bins=bins, density=False)
        # TODO: check when we have zero observations, which bin?
        # check bins start at 0 exept leading bin
        assert all([ei[1] == 0 for ei in e])
        c /= len(data)

        cdf_grid = prob2cdf_grid(c)
        return cls(cdf_grid)

    @cache_readonly
    def prob_grid(self):
        return cdf2prob_grid(self.cdf_grid, prepend=None)

    def cdf(self, x):
        """cdf values evaluated at x.

        Parameters
        ----------
        x : array_like
            Points of multivariate random variable at which cdf is evaluated.
            This can be a single point with length equal to the dimension of
            the random variable, or two dimensional with points (observations)
            in rows and random variables in columns.
            In the univariate case, a 1-dimensional x will be interpreted as
            different points for evaluation.

        Returns
        -------
        pdf values

        Notes
        -----
        Warning: 2-dim x with many points can be memory intensive because
        currently the bernstein polynomials will be evaluated in a fully
        vectorized computation.
        """
        x = np.asarray(x)
        if x.ndim == 1 and self.k_dim == 1:
            x = x[:, None]
        cdf_ = _eval_bernstein_dd(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        """pdf values evaluated at x.

        Parameters
        ----------
        x : array_like
            Points of multivariate random variable at which pdf is evaluated.
            This can be a single point with length equal to the dimension of
            the random variable, or two dimensional with points (observations)
            in rows and random variables in columns.
            In the univariate case, a 1-dimensional x will be interpreted as
            different points for evaluation.

        Returns
        -------
        cdf values

        Notes
        -----
        Warning: 2-dim x with many points can be memory intensive because
        currently the bernstein polynomials will be evaluated in a fully
        vectorized computation.
        """
        x = np.asarray(x)
        if x.ndim == 1 and self.k_dim == 1:
            x = x[:, None]
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_dd(x, self.prob_grid)
        return pdf_

    def get_marginal(self, idx):
        """Get marginal BernsteinDistribution.

        Parameters
        ----------
        idx : int or list of int
            Index or indices of the component for which the marginal
            distribution is returned.

        Returns
        -------
        BernsteinDistribution instance for the marginal distribution.
        """

        # univariate
        if self.k_dim == 1:
            return self

        sl = [-1] * self.k_dim
        if np.shape(idx) == ():
            idx = [idx]
        for ii in idx:
            sl[ii] = slice(None, None, None)
        cdf_m = self.cdf_grid[tuple(sl)]
        bpd_marginal = BernsteinDistribution(cdf_m)
        return bpd_marginal

    def rvs(self, nobs):
        """Generate random numbers from distribution.

        Parameters
        ----------
        nobs : int
            Number of random observations to generate.
        """
        rvs_mnl = np.random.multinomial(nobs, self.prob_grid.flatten())
        k_comp = self.k_dim
        rvs_m = []
        for i in range(len(rvs_mnl)):
            if rvs_mnl[i] != 0:
                idx = np.unravel_index(i, self.prob_grid.shape)
                rvsi = []
                for j in range(k_comp):
                    n = self.k_grid[j]
                    xgi = self._grid.x_marginal[j][idx[j]]
                    # Note: x_marginal starts at 0
                    #       x_marginal ends with 1 but that is not used by idx
                    rvsi.append(stats.beta.rvs(n * xgi + 1, n * (1-xgi) + 0,
                                               size=rvs_mnl[i]))
                rvs_m.append(np.column_stack(rvsi))

        rvsm = np.concatenate(rvs_m)
        return rvsm


class BernsteinDistributionBV(BernsteinDistribution):

    def cdf(self, x):
        cdf_ = _eval_bernstein_2d(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_2d(x, self.prob_grid)
        return pdf_


class BernsteinDistributionUV(BernsteinDistribution):

    def cdf(self, x, method="binom"):

        cdf_ = _eval_bernstein_1d(x, self.cdf_grid, method=method)
        return cdf_

    def pdf(self, x, method="binom"):
        # TODO: check usage of k_grid_product. Should this go into eval?
        pdf_ = self.k_grid_product * _eval_bernstein_1d(x, self.prob_grid,
                                                        method=method)
        return pdf_
