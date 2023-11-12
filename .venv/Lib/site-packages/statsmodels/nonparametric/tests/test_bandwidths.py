# -*- coding: utf-8 -*-
"""

Tests for bandwidth selection and calculation.

Author: Padarn Wilson
"""

import numpy as np
from scipy import stats

from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference


from numpy.testing import assert_allclose
import pytest

# setup test data

np.random.seed(12345)
Xi = mixture_rvs([.25,.75], size=200, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))


class TestBandwidthCalculation:

    def test_calculate_bandwidth_gaussian(self):

        bw_expected = [0.29774853596742024,
                       0.25304408155871411,
                       0.29781147113698891]

        kern = kernels.Gaussian()

        bw_calc = [0, 0, 0]
        for ii, bw in enumerate(['scott','silverman','normal_reference']):
            bw_calc[ii] = select_bandwidth(Xi, bw, kern)

        assert_allclose(bw_expected, bw_calc)

    def test_calculate_normal_reference_bandwidth(self):
        # Should be the same as the Gaussian Kernel
        bw_expected = 0.29781147113698891
        bw = bw_normal_reference(Xi)
        assert_allclose(bw, bw_expected)


class CheckNormalReferenceConstant:

    def test_calculate_normal_reference_constant(self):
        const = self.constant
        kern = self.kern
        assert_allclose(const, kern.normal_reference_constant, 1e-2)


class TestEpanechnikov(CheckNormalReferenceConstant):

    kern = kernels.Epanechnikov()
    constant = 2.34


class TestGaussian(CheckNormalReferenceConstant):

    kern = kernels.Gaussian()
    constant = 1.06


class TestBiweight(CheckNormalReferenceConstant):

    kern = kernels.Biweight()
    constant = 2.78


class TestTriweight(CheckNormalReferenceConstant):

    kern = kernels.Triweight()
    constant = 3.15


class BandwidthZero:

    def test_bandwidth_zero(self):

        kern = kernels.Gaussian()
        for bw in ['scott', 'silverman', 'normal_reference']:
            with pytest.raises(RuntimeError,
                               match="Selected KDE bandwidth is 0"):
                select_bandwidth(self.xx, bw, kern)


class TestAllBandwidthZero(BandwidthZero):

    xx = np.ones((100, 3))


class TestAnyBandwidthZero(BandwidthZero):

    xx = np.random.normal(size=(100, 3))
    xx[:, 0] = 1.0
