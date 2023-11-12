# -*- coding: utf-8 -*-
"""Unit tests for Gram-Charlier exansion

No reference results, test based on consistency and normal case.

Created on Wed Feb 19 12:39:49 2014

Author: Josef Perktold
"""

import pytest
import numpy as np
from scipy import stats

from numpy.testing import assert_allclose, assert_array_less

from statsmodels.sandbox.distributions.extras import NormExpan_gen


class CheckDistribution:

    @pytest.mark.smoke
    def test_dist1(self):
        self.dist1.rvs(size=10)
        self.dist1.pdf(np.linspace(-4, 4, 11))

    def test_cdf_ppf_roundtrip(self):
        # round trip
        probs = np.linspace(0.001, 0.999, 6)
        ppf = self.dist2.ppf(probs)
        cdf = self.dist2.cdf(ppf)
        assert_allclose(cdf, probs, rtol=1e-6)
        sf = self.dist2.sf(ppf)
        assert_allclose(sf, 1 - probs, rtol=1e-6)


class CheckExpandNorm(CheckDistribution):

    def test_pdf(self):
        scale = getattr(self, 'scale', 1)
        x = np.linspace(-4, 4, 11) * scale
        pdf2 = self.dist2.pdf(x)
        pdf1 = self.dist1.pdf(x)
        atol_pdf = getattr(self, 'atol_pdf', 0)
        assert_allclose(((pdf2 - pdf1)**2).mean(), 0, rtol=1e-6, atol=atol_pdf)
        assert_allclose(pdf2, pdf1, rtol=1e-6, atol=atol_pdf)

    def test_mvsk(self):
        #compare defining mvsk with numerical integration, generic stats
        mvsk2 = self.dist2.mvsk
        mvsk1 = self.dist2.stats(moments='mvsk')
        assert_allclose(mvsk2, mvsk1, rtol=1e-6, atol=1e-13)

        # check mvsk that was used to generate distribution
        assert_allclose(self.dist2.mvsk, self.mvsk, rtol=1e-12)



class TestExpandNormMom(CheckExpandNorm):
    # compare with normal, skew=0, excess_kurtosis=0

    @classmethod
    def setup_class(cls):
        cls.scale = 2
        cls.dist1 = stats.norm(1, 2)
        cls.mvsk = [1., 2**2, 0, 0]
        cls.dist2 = NormExpan_gen(cls.mvsk, mode='mvsk')


class TestExpandNormSample:
    # do not subclass CheckExpandNorm,
    # precision not high enough because of mvsk from data

    @classmethod
    def setup_class(cls):
        cls.dist1 = dist1 = stats.norm(1, 2)
        np.random.seed(5999)
        cls.rvs = dist1.rvs(size=200)
        #rvs = np.concatenate([rvs, -rvs])
        # fix mean and std of sample
        #rvs = (rvs - rvs.mean())/rvs.std(ddof=1) * np.sqrt(2) + 1
        cls.dist2 = NormExpan_gen(cls.rvs, mode='sample')

        cls.scale = 2
        cls.atol_pdf = 1e-3

    def test_ks(self):
        # cdf is slow
        # Kolmogorov-Smirnov test against generating sample
        stat, pvalue = stats.kstest(self.rvs, self.dist2.cdf)
        assert_array_less(0.25, pvalue)

    def test_mvsk(self):
        mvsk = stats.describe(self.rvs)[-4:]
        assert_allclose(self.dist2.mvsk, mvsk, rtol=1e-12)
