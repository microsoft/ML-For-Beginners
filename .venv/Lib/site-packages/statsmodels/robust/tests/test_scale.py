"""
Test functions for models.robust.scale
"""

import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian

import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad

# Example from Section 5.5, Venables & Ripley (2002)


DECIMAL = 4
# TODO: Can replicate these tests using stackloss data and R if this
#  data is a problem


class TestChem:
    @classmethod
    def setup_class(cls):
        cls.chem = np.array(
            [
                2.20,
                2.20,
                2.4,
                2.4,
                2.5,
                2.7,
                2.8,
                2.9,
                3.03,
                3.03,
                3.10,
                3.37,
                3.4,
                3.4,
                3.4,
                3.5,
                3.6,
                3.7,
                3.7,
                3.7,
                3.7,
                3.77,
                5.28,
                28.95,
            ]
        )

    def test_mean(self):
        assert_almost_equal(np.mean(self.chem), 4.2804, DECIMAL)

    def test_median(self):
        assert_almost_equal(np.median(self.chem), 3.385, DECIMAL)

    def test_mad(self):
        assert_almost_equal(scale.mad(self.chem), 0.52632, DECIMAL)

    def test_iqr(self):
        assert_almost_equal(scale.iqr(self.chem), 0.68570, DECIMAL)

    def test_qn(self):
        assert_almost_equal(scale.qn_scale(self.chem), 0.73231, DECIMAL)

    def test_huber_scale(self):
        assert_almost_equal(scale.huber(self.chem)[0], 3.20549, DECIMAL)

    def test_huber_location(self):
        assert_almost_equal(scale.huber(self.chem)[1], 0.67365, DECIMAL)

    def test_huber_huberT(self):
        n = scale.norms.HuberT()
        n.t = 1.5
        h = scale.Huber(norm=n)
        assert_almost_equal(
            scale.huber(self.chem)[0], h(self.chem)[0], DECIMAL
        )
        assert_almost_equal(
            scale.huber(self.chem)[1], h(self.chem)[1], DECIMAL
        )

    def test_huber_Hampel(self):
        hh = scale.Huber(norm=scale.norms.Hampel())
        assert_almost_equal(hh(self.chem)[0], 3.17434, DECIMAL)
        assert_almost_equal(hh(self.chem)[1], 0.66782, DECIMAL)


class TestMad:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_mad(self):
        m = scale.mad(self.X)
        assert_equal(m.shape, (10,))

    def test_mad_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.mad(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.mad(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.mad(empty, axis=-1), np.empty((100, 100, 0)))

    def test_mad_center(self):
        n = scale.mad(self.X, center=0)
        assert_equal(n.shape, (10,))
        with pytest.raises(TypeError):
            scale.mad(self.X, center=None)
        assert_almost_equal(
            scale.mad(self.X, center=1),
            np.median(np.abs(self.X - 1), axis=0) / Gaussian.ppf(3 / 4.0),
            DECIMAL,
        )


class TestMadAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.mad(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.mad(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.mad(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.mad(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestIqr:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_iqr(self):
        m = scale.iqr(self.X)
        assert_equal(m.shape, (10,))

    def test_iqr_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.iqr(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.iqr(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.iqr(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)


class TestIqrAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.iqr(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.iqr(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.iqr(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.iqr(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestQn:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.normal = standard_normal(size=40)
        cls.range = np.arange(0, 40)
        cls.exponential = np.random.exponential(size=40)
        cls.stackloss = sm.datasets.stackloss.load_pandas().data
        cls.sunspot = sm.datasets.sunspots.load_pandas().data.SUNACTIVITY

    def test_qn_naive(self):
        assert_almost_equal(
            scale.qn_scale(self.normal), scale._qn_naive(self.normal), DECIMAL
        )
        assert_almost_equal(
            scale.qn_scale(self.range), scale._qn_naive(self.range), DECIMAL
        )
        assert_almost_equal(
            scale.qn_scale(self.exponential),
            scale._qn_naive(self.exponential),
            DECIMAL,
        )

    def test_qn_robustbase(self):
        # from R's robustbase with finite.corr = FALSE
        assert_almost_equal(scale.qn_scale(self.range), 13.3148, DECIMAL)
        assert_almost_equal(
            scale.qn_scale(self.stackloss),
            np.array([8.87656, 8.87656, 2.21914, 4.43828]),
            DECIMAL,
        )
        # sunspot.year from datasets in R only goes up to 289
        assert_almost_equal(
            scale.qn_scale(self.sunspot[0:289]), 33.50901, DECIMAL
        )

    def test_qn_empty(self):
        empty = np.empty(0)
        assert np.isnan(scale.qn_scale(empty))
        empty = np.empty((10, 100, 0))
        assert_equal(scale.qn_scale(empty, axis=1), np.empty((10, 0)))
        empty = np.empty((100, 100, 0, 0))
        assert_equal(scale.qn_scale(empty, axis=-1), np.empty((100, 100, 0)))
        empty = np.empty(shape=())
        with pytest.raises(ValueError):
            scale.iqr(empty)


class TestQnAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))

    def test_axis0(self):
        m = scale.qn_scale(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m = scale.qn_scale(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m = scale.qn_scale(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m = scale.qn_scale(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


class TestHuber:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10))

    def test_huber_result_shape(self):
        h = scale.Huber(maxiter=100)
        m, s = h(self.X)
        assert_equal(m.shape, (10,))


class TestHuberAxes:
    @classmethod
    def setup_class(cls):
        np.random.seed(54321)
        cls.X = standard_normal((40, 10, 30))
        cls.h = scale.Huber(maxiter=1000, tol=1.0e-05)

    def test_default(self):
        m, s = self.h(self.X, axis=0)
        assert_equal(m.shape, (10, 30))

    def test_axis1(self):
        m, s = self.h(self.X, axis=1)
        assert_equal(m.shape, (40, 30))

    def test_axis2(self):
        m, s = self.h(self.X, axis=2)
        assert_equal(m.shape, (40, 10))

    def test_axisneg1(self):
        m, s = self.h(self.X, axis=-1)
        assert_equal(m.shape, (40, 10))


def test_mad_axis_none():
    # GH 7027
    a = np.array([[0, 1, 2], [2, 3, 2]])

    def m(x):
        return np.median(x)

    direct = mad(a=a, axis=None)
    custom = mad(a=a, axis=None, center=m)
    axis0 = mad(a=a.ravel(), axis=0)

    np.testing.assert_allclose(direct, custom)
    np.testing.assert_allclose(direct, axis0)
