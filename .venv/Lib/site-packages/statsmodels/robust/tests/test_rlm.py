"""
Test functions for sm.rlm
"""
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


def load_stackloss():
    from statsmodels.datasets.stackloss import load
    data = load()
    data.endog = np.asarray(data.endog)
    data.exog = np.asarray(data.exog)
    return data


class CheckRlmResultsMixin:
    '''
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to results.results_rlm

    Covariance matrices were obtained from SAS and are imported from
    results.results_rlm
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    decimal_standarderrors = DECIMAL_4

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse,
                            self.decimal_standarderrors)

    # TODO: get other results from SAS, though if it works for one...
    def test_confidenceintervals(self):
        if not hasattr(self.res2, 'conf_int'):
            pytest.skip("Results from R")

        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(),
                            DECIMAL_4)

    decimal_scale = DECIMAL_4

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                            self.decimal_scale)

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL_4)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model, self.res2.df_model,
                            DECIMAL_4)
        assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid,
                            DECIMAL_4)

    def test_bcov_unscaled(self):
        if not hasattr(self.res2, 'bcov_unscaled'):
            pytest.skip("No unscaled cov matrix from SAS")

        assert_almost_equal(self.res1.bcov_unscaled,
                            self.res2.bcov_unscaled, DECIMAL_4)

    decimal_bcov_scaled = DECIMAL_4

    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled, self.res2.h1,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h2, self.res2.h2,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h3, self.res2.h3,
                            self.decimal_bcov_scaled)

    def test_tvalues(self):
        if not hasattr(self.res2, 'tvalues'):
            pytest.skip("No tvalues in benchmark")

        assert_allclose(self.res1.tvalues, self.res2.tvalues, rtol=0.003)

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))

        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)


class TestRlm(CheckRlmResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1
        cls.decimal_scale = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        cls.model = model
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Huber
        self.res2 = Huber()

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        self.res1.summary2()

    @pytest.mark.smoke
    def test_chisq(self):
        assert isinstance(self.res1.chisq, np.ndarray)

    @pytest.mark.smoke
    def test_predict(self):
        assert isinstance(self.model.predict(self.res1.params), np.ndarray)


class TestHampel(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestHampel, cls).setup_class()
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_2
        cls.decimal_scale = DECIMAL_3
        cls.decimal_bcov_scaled = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.Hampel())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Hampel
        self.res2 = Hampel()


class TestRlmBisquare(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestRlmBisquare, cls).setup_class()
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1

        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BiSquare
        self.res2 = BiSquare()


class TestRlmAndrews(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestRlmAndrews, cls).setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.AndrewWave())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Andrews
        self.res2 = Andrews()


# --------------------------------------------------------------------
# tests with Huber scaling

class TestRlmHuber(CheckRlmResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import HuberHuber
        self.res2 = HuberHuber()


class TestHampelHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestHampelHuber, cls).setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.Hampel())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import HampelHuber
        self.res2 = HampelHuber()


class TestRlmBisquareHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestRlmBisquareHuber, cls).setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BisquareHuber
        self.res2 = BisquareHuber()


class TestRlmAndrewsHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super(TestRlmAndrewsHuber, cls).setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.AndrewWave())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import AndrewsHuber
        self.res2 = AndrewsHuber()


class TestRlmSresid(CheckRlmResultsMixin):
    # Check GH#187
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1
        cls.decimal_scale = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        results = model.fit(conv='sresid')
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Huber
        self.res2 = Huber()


@pytest.mark.smoke
def test_missing():
    # see GH#2083
    import statsmodels.formula.api as smf

    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan]}
    smf.rlm('Foo ~ Bar', data=d)


def test_rlm_start_values():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())
    results = model.fit()
    start_params = [0.7156402, 1.29528612, -0.15212252, -39.91967442]
    result_sv = model.fit(start_params=start_params)
    assert_allclose(results.params, result_sv.params)


def test_rlm_start_values_errors():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())
    start_params = [0.7156402, 1.29528612, -0.15212252]
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)

    start_params = np.array([start_params, start_params]).T
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)


@pytest.fixture(scope='module',
                params=[norms.AndrewWave, norms.LeastSquares, norms.HuberT,
                        norms.TrimmedMean, norms.TukeyBiweight, norms.Hampel,
                        norms.RamsayE])
def norm(request):
    return request.param()


@pytest.fixture(scope='module')
def perfect_fit_data(request):
    from statsmodels.tools.tools import Bunch
    rs = np.random.RandomState(1249328932)
    exog = rs.standard_normal((1000, 1))
    endog = exog + exog ** 2
    exog = sm.add_constant(np.c_[exog, exog ** 2])
    return Bunch(endog=endog, exog=exog, const=(3.2 * np.ones_like(endog)))


def test_perfect_fit(perfect_fit_data, norm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RLM(perfect_fit_data.endog, perfect_fit_data.exog, M=norm).fit()
    assert_allclose(res.params, np.array([0, 1, 1]), atol=1e-8)


def test_perfect_const(perfect_fit_data, norm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RLM(perfect_fit_data.const, perfect_fit_data.exog, M=norm).fit()
    assert_allclose(res.params, np.array([3.2, 0, 0]), atol=1e-8)


@pytest.mark.parametrize('conv', ('weights', 'coefs', 'sresid'))
def test_alt_criterion(conv):
    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    base = RLM(data.endog, data.exog, M=norms.HuberT()).fit()
    alt = RLM(data.endog, data.exog, M=norms.HuberT()).fit(conv=conv)
    assert_allclose(base.params, alt.params)


def test_bad_criterion():
    data = load_stackloss()
    data.exog = np.asarray(data.exog)
    data.endog = np.asarray(data.endog)
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = RLM(data.endog, data.exog, M=norms.HuberT())
    with pytest.raises(ValueError, match='Convergence argument unknown'):
        mod.fit(conv='unknown')
