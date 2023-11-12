import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
    biweight_chamberlain, biweight_hsheather, biweight_bofinger,
    cosine_chamberlain, cosine_hsheather, cosine_bofinger,
    gaussian_chamberlain, gaussian_hsheather, gaussian_bofinger,
    epan2_chamberlain, epan2_hsheather, epan2_bofinger,
    parzen_chamberlain, parzen_hsheather, parzen_bofinger,
    # rectangle_chamberlain, rectangle_hsheather, rectangle_bofinger,
    # triangle_chamberlain, triangle_hsheather, triangle_bofinger,
    # epanechnikov_chamberlain, epanechnikov_hsheather, epanechnikov_bofinger,
    epanechnikov_hsheather_q75, Rquantreg)

idx = ['income', 'Intercept']


class CheckModelResultsMixin:
    def test_params(self):
        assert_allclose(np.ravel(self.res1.params.loc[idx]),
                        self.res2.table[:, 0], rtol=1e-3)

    def test_bse(self):
        assert_equal(self.res1.scale, 1)
        assert_allclose(np.ravel(self.res1.bse.loc[idx]),
                        self.res2.table[:, 1], rtol=1e-3)

    def test_tvalues(self):
        assert_allclose(np.ravel(self.res1.tvalues.loc[idx]),
                        self.res2.table[:, 2], rtol=1e-2)

    def test_pvalues(self):
        pvals_stata = scipy.stats.t.sf(self.res2.table[:, 2], self.res2.df_r)
        assert_allclose(np.ravel(self.res1.pvalues.loc[idx]),
                        pvals_stata, rtol=1.1)

        # test that we use the t distribution for the p-values
        pvals_t = scipy.stats.t.sf(self.res1.tvalues, self.res2.df_r) * 2
        assert_allclose(np.ravel(self.res1.pvalues),
                        pvals_t, rtol=1e-9, atol=1e-10)

    def test_conf_int(self):
        assert_allclose(self.res1.conf_int().loc[idx],
                        self.res2.table[:, -2:], rtol=1e-3)

    def test_nobs(self):
        assert_allclose(self.res1.nobs, self.res2.N, rtol=1e-3)

    def test_df_model(self):
        assert_allclose(self.res1.df_model, self.res2.df_m, rtol=1e-3)

    def test_df_resid(self):
        assert_allclose(self.res1.df_resid, self.res2.df_r, rtol=1e-3)

    def test_prsquared(self):
        assert_allclose(self.res1.prsquared, self.res2.psrsquared, rtol=1e-3)

    def test_sparsity(self):
        assert_allclose(np.array(self.res1.sparsity),
                        self.res2.sparsity, rtol=1e-3)

    def test_bandwidth(self):
        assert_allclose(np.array(self.res1.bandwidth),
                        self.res2.kbwidth, rtol=1e-3)


d = {('biw', 'bofinger'): biweight_bofinger,
     ('biw', 'chamberlain'): biweight_chamberlain,
     ('biw', 'hsheather'): biweight_hsheather,
     ('cos', 'bofinger'): cosine_bofinger,
     ('cos', 'chamberlain'): cosine_chamberlain,
     ('cos', 'hsheather'): cosine_hsheather,
     ('gau', 'bofinger'): gaussian_bofinger,
     ('gau', 'chamberlain'): gaussian_chamberlain,
     ('gau', 'hsheather'): gaussian_hsheather,
     ('par', 'bofinger'): parzen_bofinger,
     ('par', 'chamberlain'): parzen_chamberlain,
     ('par', 'hsheather'): parzen_hsheather,
     # ('rec','bofinger'): rectangle_bofinger,
     # ('rec','chamberlain'): rectangle_chamberlain,
     # ('rec','hsheather'): rectangle_hsheather,
     # ('tri','bofinger'): triangle_bofinger,
     # ('tri','chamberlain'): triangle_chamberlain,
     # ('tri','hsheather'): triangle_hsheather,
     ('epa', 'bofinger'): epan2_bofinger,
     ('epa', 'chamberlain'): epan2_chamberlain,
     ('epa', 'hsheather'): epan2_hsheather
     # ('epa2', 'bofinger'): epan2_bofinger,
     # ('epa2', 'chamberlain'): epan2_chamberlain,
     # ('epa2', 'hsheather'): epan2_hsheather
     }


def setup_fun(kernel='gau', bandwidth='bofinger'):
    data = sm.datasets.engel.load_pandas().data
    y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
    statsm = QuantReg(y, X).fit(vcov='iid', kernel=kernel, bandwidth=bandwidth)
    stata = d[(kernel, bandwidth)]
    return statsm, stata


def test_fitted_residuals():
    data = sm.datasets.engel.load_pandas().data
    y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
    res = QuantReg(y, X).fit(q=.1)
    # Note: maxabs relative error with fitted is 1.789e-09
    assert_almost_equal(np.array(res.fittedvalues), Rquantreg.fittedvalues, 5)
    assert_almost_equal(np.array(res.predict()), Rquantreg.fittedvalues, 5)
    assert_almost_equal(np.array(res.resid), Rquantreg.residuals, 5)


class TestEpanechnikovHsheatherQ75(CheckModelResultsMixin):
    # Vincent Arel-Bundock also spot-checked q=.1
    @classmethod
    def setup_class(cls):
        data = sm.datasets.engel.load_pandas().data
        y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
        cls.res1 = QuantReg(y, X).fit(q=.75, vcov='iid', kernel='epa',
                                      bandwidth='hsheather')
        cls.res2 = epanechnikov_hsheather_q75


class TestEpanechnikovBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'bofinger')


class TestEpanechnikovChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'chamberlain')


class TestEpanechnikovHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('epa', 'hsheather')


class TestGaussianBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'bofinger')


class TestGaussianChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'chamberlain')


class TestGaussianHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('gau', 'hsheather')


class TestBiweightBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'bofinger')


class TestBiweightChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'chamberlain')


class TestBiweightHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('biw', 'hsheather')


class TestCosineBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'bofinger')


class TestCosineChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'chamberlain')


class TestCosineHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('cos', 'hsheather')


class TestParzeneBofinger(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('par', 'bofinger')


class TestParzeneChamberlain(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('par', 'chamberlain')


class TestParzeneHsheather(CheckModelResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.res1, cls.res2 = setup_fun('par', 'hsheather')

# class TestTriangleBofinger(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'bofinger')

# class TestTriangleChamberlain(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'chamberlain')

# class TestTriangleHsheather(CheckModelResultsMixin):
#    @classmethod
#    def setup_class(cls):
#        cls.res1, cls.res2 = setup_fun('tri', 'hsheather')


def test_zero_resid():
    # smoke and regression tests

    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain')  # 'bofinger')
    res.summary()

    assert_allclose(res.params,
                    np.array([0.0, 0.96774163]),
                    rtol=1e-4, atol=1e-20)
    assert_allclose(res.bse,
                    np.array([0.0447576, 0.01154867]),
                    rtol=1e-4, atol=1e-20)
    assert_allclose(res.resid,
                    np.array([0.0, 3.22583680e-02,
                              -3.22574272e-02, 9.40732912e-07]),
                    rtol=1e-4, atol=1e-20)

    X = np.array([[1, 0], [0.1, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain')
    res.summary()

    assert_allclose(res.params, np.array([9.99982796e-08, 9.67741630e-01]),
                    rtol=1e-4, atol=1e-20)
    assert_allclose(res.bse, np.array([0.04455029, 0.01155251]), rtol=1e-4,
                    atol=1e-20)
    assert_allclose(res.resid, np.array([-9.99982796e-08, 3.22583598e-02,
                                         -3.22574234e-02, 9.46361860e-07]),
                    rtol=1e-4, atol=1e-20)


def test_use_t_summary():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain', use_t=True)
    summ = res.summary()
    assert 'P>|t|' in str(summ)
    assert 'P>|z|' not in str(summ)


def test_alpha_summary():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain', use_t=True)
    summ_20 = res.summary(alpha=.2)
    assert '[0.025      0.975]' not in str(summ_20)
    assert '[0.1        0.9]' in str(summ_20)


def test_remove_data():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res = QuantReg(y, X).fit(0.5)
    res.remove_data()


def test_collinear_matrix():
    X = np.array([[1, 0, .5], [1, 0, .8],
                  [1, 0, 1.5], [1, 0, .25]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    res_collinear = QuantReg(y, X).fit(0.5)
    assert len(res_collinear.params) == X.shape[1]


def test_nontrivial_singular_matrix():
    x_one = np.random.random(1000)
    x_two = np.random.random(1000)*10
    x_three = np.random.random(1000)
    intercept = np.ones(1000)

    y = np.random.random(1000)*5
    X = np.column_stack((intercept, x_one, x_two, x_three, x_one))

    assert np.linalg.matrix_rank(X) < X.shape[1]
    res_singular = QuantReg(y, X).fit(0.5)
    assert len(res_singular.params) == X.shape[1]
    assert np.linalg.matrix_rank(res_singular.cov_params()) == X.shape[1] - 1

    # prediction is correct even with singular exog
    res_ns = QuantReg(y, X[:, :-1]).fit(0.5)
    assert_allclose(res_singular.fittedvalues, res_ns.fittedvalues, rtol=0.01)
