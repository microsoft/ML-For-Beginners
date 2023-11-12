# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 23:44:18 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats

from statsmodels.distributions.copula.api import (
        CopulaDistribution, ArchimedeanCopula)
from statsmodels.distributions.copula.api import transforms as tra
import statsmodels.distributions.tools as dt
from statsmodels.distributions.bernstein import (
    BernsteinDistribution, BernsteinDistributionBV, BernsteinDistributionUV)


def test_bernstein_distribution_1d():
    grid = dt._Grid([501])
    loc = grid.x_flat == 0
    grid.x_flat[loc] = grid.x_flat[~loc].min() / 2
    grid.x_flat[grid.x_flat == 1] = 1 - grid.x_flat.min()

    distr = stats.beta(3, 5)
    cdf_g = distr.cdf(np.squeeze(grid.x_flat))
    bpd = BernsteinDistribution(cdf_g)

    cdf_bp = bpd.cdf(grid.x_flat)
    assert_allclose(cdf_bp, cdf_g, atol=0.005)
    assert_array_less(np.median(np.abs(cdf_bp - cdf_g)), 0.001)

    pdfv = distr.pdf(np.squeeze(grid.x_flat))
    pdf_bp = bpd.pdf(grid.x_flat)
    assert_allclose(pdf_bp, pdfv, atol=0.02)
    assert_array_less(np.median(np.abs(pdf_bp - pdfv)), 0.01)

    # compare with UV class
    xf = np.squeeze(grid.x_flat)  # UV returns column if x is column
    bpd1 = BernsteinDistributionUV(cdf_g)
    cdf_bp1 = bpd1.cdf(xf)
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf)
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)

    cdf_bp1 = bpd1.cdf(xf, method="beta")
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf, method="beta")
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)

    cdf_bp1 = bpd1.cdf(xf, method="bpoly")
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf, method="bpoly")
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)

    # check rvs
    # currently smoke test
    rvs = bpd.rvs(100)
    assert len(rvs) == 100


def test_bernstein_distribution_2d():
    grid = dt._Grid([51, 51])

    cop_tr = tra.TransfFrank
    args = (2,)
    ca = ArchimedeanCopula(cop_tr())
    distr1 = stats.uniform
    distr2 = stats.uniform
    cad = CopulaDistribution(ca, [distr1, distr2], cop_args=args)
    cdfv = cad.cdf(grid.x_flat, args)
    cdf_g = cdfv.reshape(grid.k_grid)

    bpd = BernsteinDistribution(cdf_g)

    cdf_bp = bpd.cdf(grid.x_flat)
    assert_allclose(cdf_bp, cdfv, atol=0.005)
    assert_array_less(np.median(np.abs(cdf_bp - cdfv)), 0.001)

    grid_eps = dt._Grid([51, 51], eps=1e-8)
    pdfv = cad.pdf(grid_eps.x_flat)
    pdf_bp = bpd.pdf(grid_eps.x_flat)
    assert_allclose(pdf_bp, pdfv, atol=0.01, rtol=0.04)
    assert_array_less(np.median(np.abs(pdf_bp - pdfv)), 0.05)

    # check marginal cdfs
    # get marginal cdf
    xx = np.column_stack((np.linspace(0, 1, 5), np.ones(5)))
    cdf_m1 = bpd.cdf(xx)
    assert_allclose(cdf_m1, xx[:, 0], atol=1e-13)
    xx = np.column_stack((np.ones(5), np.linspace(0, 1, 5)))
    cdf_m2 = bpd.cdf(xx)
    assert_allclose(cdf_m2, xx[:, 1], atol=1e-13)

    xx_ = np.linspace(0, 1, 5)
    xx = xx_[:, None]  # currently requires 2-dim
    bpd_m1 = bpd.get_marginal(0)
    cdf_m1 = bpd_m1.cdf(xx)
    assert_allclose(cdf_m1, xx_, atol=1e-13)
    pdf_m1 = bpd_m1.pdf(xx)
    assert_allclose(pdf_m1, np.ones(len(xx)), atol=1e-13)

    bpd_m = bpd.get_marginal(1)
    cdf_m = bpd_m.cdf(xx)
    assert_allclose(cdf_m, xx_, atol=1e-13)
    pdf_m = bpd_m.pdf(xx)
    assert_allclose(pdf_m, np.ones(len(xx)), atol=1e-13)


class TestBernsteinBeta2d:

    @classmethod
    def setup_class(cls):
        grid = dt._Grid([91, 101])

        cop_tr = tra.TransfFrank
        args = (2,)
        ca = ArchimedeanCopula(cop_tr())
        distr1 = stats.beta(4, 3)
        distr2 = stats.beta(4, 4)  # (5, 2)
        cad = CopulaDistribution(ca, [distr1, distr2], cop_args=args)
        cdfv = cad.cdf(grid.x_flat, args)
        cdf_g = cdfv.reshape(grid.k_grid)

        cls.grid = grid
        cls.cdfv = cdfv
        cls.distr = cad
        cls.bpd = BernsteinDistributionBV(cdf_g)

    def test_basic(self):
        bpd = self.bpd
        grid = self.grid
        cdfv = self.cdfv
        distr = self.distr

        if grid.x_flat.shape[0] < 51**2:
            cdf_bp = bpd.cdf(grid.x_flat)
            assert_allclose(cdf_bp, cdfv, atol=0.05)
            assert_array_less(np.median(np.abs(cdf_bp - cdfv)), 0.01)

        grid_eps = dt._Grid([51, 51], eps=1e-2)
        cdfv = distr.cdf(grid_eps.x_flat)
        cdf_bp = bpd.cdf(grid_eps.x_flat)
        assert_allclose(cdf_bp, cdfv, atol=0.01, rtol=0.01)
        assert_array_less(np.median(np.abs(cdf_bp - cdfv)), 0.05)

        pdfv = distr.pdf(grid_eps.x_flat)
        pdf_bp = bpd.pdf(grid_eps.x_flat)
        assert_allclose(pdf_bp, pdfv, atol=0.06, rtol=0.1)
        assert_array_less(np.median(np.abs(pdf_bp - pdfv)), 0.05)

    def test_rvs(self):
        # currently smoke test
        rvs = self.bpd.rvs(100)
        assert len(rvs) == 100


class TestBernsteinBeta2dd(TestBernsteinBeta2d):

    @classmethod
    def setup_class(cls):
        grid = dt._Grid([91, 101])

        cop_tr = tra.TransfFrank
        args = (2,)
        ca = ArchimedeanCopula(cop_tr())
        distr1 = stats.beta(4, 3)
        distr2 = stats.beta(4, 4)  # (5, 2)
        cad = CopulaDistribution(ca, [distr1, distr2], cop_args=args)
        cdfv = cad.cdf(grid.x_flat, args)
        cdf_g = cdfv.reshape(grid.k_grid)

        cls.grid = grid
        cls.cdfv = cdfv
        cls.distr = cad
        cls.bpd = BernsteinDistribution(cdf_g)
