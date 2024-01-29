"""
Test functions for GEE

External comparisons are to R and Stata.  The statsmodels GEE
implementation should generally agree with the R GEE implementation
for the independence and exchangeable correlation structures.  For
other correlation structures, the details of the correlation
estimation differ among implementations and the results will not agree
exactly.
"""

from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_array_less, assert_raises, assert_warns,
                           assert_)
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete

import pandas as pd
from scipy.stats.distributions import norm
import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

pdf_output = False

if pdf_output:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("test_glm.pdf")
else:
    pdf = None


def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)


def load_data(fname, icept=True):
    """
    Load a data set from the results directory.  The data set should
    be a CSV file with the following format:

    Column 0: Group indicator
    Column 1: endog variable
    Columns 2-end: exog variables

    If `icept` is True, an intercept is prepended to the exog
    variables.
    """

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    Z = np.genfromtxt(os.path.join(cur_dir, 'results', fname),
                      delimiter=",")

    group = Z[:, 0]
    endog = Z[:, 1]
    exog = Z[:, 2:]

    if icept:
        exog = np.concatenate((np.ones((exog.shape[0], 1)), exog),
                              axis=1)

    return endog, exog, group


def check_wrapper(results):
    # check wrapper
    assert_(isinstance(results.params, pd.Series))
    assert_(isinstance(results.fittedvalues, pd.Series))
    assert_(isinstance(results.resid, pd.Series))
    assert_(isinstance(results.centered_resid, pd.Series))

    assert_(isinstance(results._results.params, np.ndarray))
    assert_(isinstance(results._results.fittedvalues, np.ndarray))
    assert_(isinstance(results._results.resid, np.ndarray))
    assert_(isinstance(results._results.centered_resid, np.ndarray))


class TestGEE:

    def test_margins_gaussian(self):
        # Check marginal effects for a Gaussian GEE fit.  Marginal
        # effects and ordinary effects should be equal.

        n = 40
        np.random.seed(34234)
        exog = np.random.normal(size=(n, 3))
        exog[:, 0] = 1

        groups = np.kron(np.arange(n / 4), np.r_[1, 1, 1, 1])
        endog = exog[:, 1] + np.random.normal(size=n)

        model = gee.GEE(endog, exog, groups)
        result = model.fit(
            start_params=[-4.88085602e-04, 1.18501903, 4.78820100e-02])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, result.params[1:])
        assert_allclose(marg.margeff_se, result.bse[1:])

        # smoke test
        marg.summary()

    def test_margins_gaussian_lists_tuples(self):
        # Check marginal effects for a Gaussian GEE fit using lists and
        # tuples. Marginal effects and ordinary effects should be equal.

        n = 40
        np.random.seed(34234)
        exog_arr = np.random.normal(size=(n, 3))
        exog_arr[:, 0] = 1

        groups_arr = np.kron(np.arange(n / 4), np.r_[1, 1, 1, 1])
        endog_arr = exog_arr[:, 1] + np.random.normal(size=n)

        # check that GEE accepts lists
        exog_list = [list(row) for row in exog_arr]
        groups_list = list(groups_arr)
        endog_list = list(endog_arr)

        model = gee.GEE(endog_list, exog_list, groups_list)
        result = model.fit(
            start_params=[-4.88085602e-04, 1.18501903, 4.78820100e-02])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, result.params[1:])
        assert_allclose(marg.margeff_se, result.bse[1:])

        # check that GEE accepts tuples
        exog_tuple = tuple(tuple(row) for row in exog_arr)
        groups_tuple = tuple(groups_arr)
        endog_tuple = tuple(endog_arr)

        model = gee.GEE(endog_tuple, exog_tuple, groups_tuple)
        result = model.fit(
            start_params=[-4.88085602e-04, 1.18501903, 4.78820100e-02])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, result.params[1:])
        assert_allclose(marg.margeff_se, result.bse[1:])

    def test_margins_logistic(self):
        # Check marginal effects for a binomial GEE fit.  Comparison
        # comes from Stata.

        np.random.seed(34234)
        endog = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
        exog = np.ones((8, 2))
        exog[:, 1] = np.r_[1, 2, 1, 1, 2, 1, 2, 2]

        groups = np.arange(8)

        model = gee.GEE(endog, exog, groups, family=families.Binomial())
        result = model.fit(
            cov_type='naive', start_params=[-3.29583687,  2.19722458])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, np.r_[0.4119796])
        assert_allclose(marg.margeff_se, np.r_[0.1379962], rtol=1e-6)

    def test_margins_multinomial(self):
        # Check marginal effects for a 2-class multinomial GEE fit,
        # which should be equivalent to logistic regression.  Comparison
        # comes from Stata.

        np.random.seed(34234)
        endog = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
        exog = np.ones((8, 2))
        exog[:, 1] = np.r_[1, 2, 1, 1, 2, 1, 2, 2]

        groups = np.arange(8)

        model = gee.NominalGEE(endog, exog, groups)
        result = model.fit(cov_type='naive', start_params=[
                           3.295837, -2.197225])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, np.r_[-0.41197961], rtol=1e-5)
        assert_allclose(marg.margeff_se, np.r_[0.1379962], rtol=1e-6)

    @pytest.mark.smoke
    @pytest.mark.matplotlib
    def test_nominal_plot(self, close_figures):
        np.random.seed(34234)
        endog = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
        exog = np.ones((8, 2))
        exog[:, 1] = np.r_[1, 2, 1, 1, 2, 1, 2, 2]

        groups = np.arange(8)

        model = gee.NominalGEE(endog, exog, groups)
        result = model.fit(cov_type='naive',
                           start_params=[3.295837, -2.197225])

        fig = result.plot_distribution()
        assert_equal(isinstance(fig, plt.Figure), True)

    def test_margins_poisson(self):
        # Check marginal effects for a Poisson GEE fit.

        np.random.seed(34234)
        endog = np.r_[10, 15, 12, 13, 20, 18, 26, 29]
        exog = np.ones((8, 2))
        exog[:, 1] = np.r_[0, 0, 0, 0, 1, 1, 1, 1]

        groups = np.arange(8)

        model = gee.GEE(endog, exog, groups, family=families.Poisson())
        result = model.fit(cov_type='naive', start_params=[
                           2.52572864, 0.62057649])

        marg = result.get_margeff()

        assert_allclose(marg.margeff, np.r_[11.0928], rtol=1e-6)
        assert_allclose(marg.margeff_se, np.r_[3.269015], rtol=1e-6)

    def test_multinomial(self):
        """
        Check the 2-class multinomial (nominal) GEE fit against
        logistic regression.
        """

        np.random.seed(34234)
        endog = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
        exog = np.ones((8, 2))
        exog[:, 1] = np.r_[1, 2, 1, 1, 2, 1, 2, 2]

        groups = np.arange(8)

        model = gee.NominalGEE(endog, exog, groups)
        results = model.fit(cov_type='naive', start_params=[
                            3.295837, -2.197225])

        logit_model = gee.GEE(endog, exog, groups,
                              family=families.Binomial())
        logit_results = logit_model.fit(cov_type='naive')

        assert_allclose(results.params, -logit_results.params, rtol=1e-5)
        assert_allclose(results.bse, logit_results.bse, rtol=1e-5)

    def test_weighted(self):

        # Simple check where the answer can be computed by hand.
        exog = np.ones(20)
        weights = np.ones(20)
        weights[0:10] = 2
        endog = np.zeros(20)
        endog[0:10] += 1
        groups = np.kron(np.arange(10), np.r_[1, 1])
        model = gee.GEE(endog, exog, groups, weights=weights)
        result = model.fit()
        assert_allclose(result.params, np.r_[2 / 3.])

        # Comparison against stata using groups with different sizes.
        weights = np.ones(20)
        weights[10:] = 2
        endog = np.r_[1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6,
                      7, 8, 7, 8]
        exog1 = np.r_[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                      3, 3, 3, 3]
        groups = np.r_[1, 1, 2, 2, 2, 2, 4, 4, 5, 5, 6, 6, 6, 6,
                       8, 8, 9, 9, 10, 10]
        exog = np.column_stack((np.ones(20), exog1))

        # Comparison using independence model
        model = gee.GEE(endog, exog, groups, weights=weights,
                        cov_struct=cov_struct.Independence())
        g = np.mean([2, 4, 2, 2, 4, 2, 2, 2])
        fac = 20 / float(20 - g)
        result = model.fit(ddof_scale=0, scaling_factor=fac)

        assert_allclose(result.params, np.r_[1.247573, 1.436893], atol=1e-6)
        assert_allclose(result.scale, 1.808576)

        # Stata multiples robust SE by sqrt(N / (N - g)), where N is
        # the total sample size and g is the average group size.
        assert_allclose(result.bse, np.r_[0.895366, 0.3425498], atol=1e-5)

        # Comparison using exchangeable model
        # Smoke test for now
        model = gee.GEE(endog, exog, groups, weights=weights,
                        cov_struct=cov_struct.Exchangeable())
        model.fit(ddof_scale=0)

    # This is in the release announcement for version 0.6.
    def test_poisson_epil(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(cur_dir, "results", "epil.csv")
        data = pd.read_csv(fname)

        fam = families.Poisson()
        ind = cov_struct.Independence()
        mod1 = gee.GEE.from_formula("y ~ age + trt + base", data["subject"],
                                    data, cov_struct=ind, family=fam)
        rslt1 = mod1.fit(cov_type='naive')

        # Coefficients should agree with GLM
        from statsmodels.genmod.generalized_linear_model import GLM

        mod2 = GLM.from_formula("y ~ age + trt + base", data,
                                family=families.Poisson())
        rslt2 = mod2.fit()

        # do not use wrapper, asserts_xxx do not work
        rslt1 = rslt1._results
        rslt2 = rslt2._results

        assert_allclose(rslt1.params, rslt2.params, rtol=1e-6, atol=1e-6)
        assert_allclose(rslt1.bse, rslt2.bse, rtol=1e-6, atol=1e-6)

    def test_missing(self):
        # Test missing data handling for calling from the api.  Missing
        # data handling does not currently work for formulas.

        np.random.seed(34234)
        endog = np.random.normal(size=100)
        exog = np.random.normal(size=(100, 3))
        exog[:, 0] = 1
        groups = np.kron(lrange(20), np.ones(5))

        endog[0] = np.nan
        endog[5:7] = np.nan
        exog[10:12, 1] = np.nan

        mod1 = gee.GEE(endog, exog, groups, missing='drop')
        rslt1 = mod1.fit()

        assert_almost_equal(len(mod1.endog), 95)
        assert_almost_equal(np.asarray(mod1.exog.shape), np.r_[95, 3])

        ii = np.isfinite(endog) & np.isfinite(exog).all(1)

        mod2 = gee.GEE(endog[ii], exog[ii, :], groups[ii], missing='none')
        rslt2 = mod2.fit()

        assert_almost_equal(rslt1.params, rslt2.params)
        assert_almost_equal(rslt1.bse, rslt2.bse)

    def test_missing_formula(self):
        # Test missing data handling for formulas.

        np.random.seed(34234)
        endog = np.random.normal(size=100)
        exog1 = np.random.normal(size=100)
        exog2 = np.random.normal(size=100)
        exog3 = np.random.normal(size=100)
        groups = np.kron(lrange(20), np.ones(5))

        endog[0] = np.nan
        endog[5:7] = np.nan
        exog2[10:12] = np.nan

        data0 = pd.DataFrame({"endog": endog, "exog1": exog1, "exog2": exog2,
                              "exog3": exog3, "groups": groups})

        for k in 0, 1:
            data = data0.copy()
            kwargs = {}
            if k == 1:
                data["offset"] = 0
                data["time"] = 0
                kwargs["offset"] = "offset"
                kwargs["time"] = "time"

            mod1 = gee.GEE.from_formula("endog ~ exog1 + exog2 + exog3",
                                        groups="groups", data=data,
                                        missing='drop', **kwargs)
            rslt1 = mod1.fit()

            assert_almost_equal(len(mod1.endog), 95)
            assert_almost_equal(np.asarray(mod1.exog.shape), np.r_[95, 4])

            data = data.dropna()

            kwargs = {}
            if k == 1:
                kwargs["offset"] = data["offset"]
                kwargs["time"] = data["time"]

            mod2 = gee.GEE.from_formula("endog ~ exog1 + exog2 + exog3",
                                        groups=data["groups"], data=data,
                                        missing='none', **kwargs)
            rslt2 = mod2.fit()

            assert_almost_equal(rslt1.params.values, rslt2.params.values)
            assert_almost_equal(rslt1.bse.values, rslt2.bse.values)

    @pytest.mark.parametrize("k1", [False, True])
    @pytest.mark.parametrize("k2", [False, True])
    def test_invalid_args(self, k1, k2):

        for j in range(3):

            p = [20, 20, 20]
            p[j] = 18

            endog = np.zeros(p[0])
            exog = np.zeros((p[1], 2))

            kwargs = {}
            kwargs["groups"] = np.zeros(p[2])
            if k1:
                kwargs["exposure"] = np.zeros(18)
            if k2:
                kwargs["time"] = np.zeros(18)
            with assert_raises(ValueError):
                gee.GEE(endog, exog, **kwargs)

    def test_default_time(self):
        # Check that the time defaults work correctly.

        endog, exog, group = load_data("gee_logistic_1.csv")

        # Time values for the autoregressive model
        T = np.zeros(len(endog))
        idx = set(group)
        for ii in idx:
            jj = np.flatnonzero(group == ii)
            T[jj] = lrange(len(jj))

        family = families.Binomial()
        va = cov_struct.Autoregressive(grid=False)

        md1 = gee.GEE(endog, exog, group, family=family, cov_struct=va)
        mdf1 = md1.fit()

        md2 = gee.GEE(endog, exog, group, time=T, family=family,
                      cov_struct=va)
        mdf2 = md2.fit()

        assert_almost_equal(mdf1.params, mdf2.params, decimal=6)
        assert_almost_equal(mdf1.standard_errors(),
                            mdf2.standard_errors(), decimal=6)

    def test_logistic(self):
        # R code for comparing results:

        # library(gee)
        # Z = read.csv("results/gee_logistic_1.csv", header=FALSE)
        # Y = Z[,2]
        # Id = Z[,1]
        # X1 = Z[,3]
        # X2 = Z[,4]
        # X3 = Z[,5]

        # mi = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
        #         corstr="independence")
        # smi = summary(mi)
        # u = coefficients(smi)
        # cfi = paste(u[,1], collapse=",")
        # sei = paste(u[,4], collapse=",")

        # me = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
        #         corstr="exchangeable")
        # sme = summary(me)
        # u = coefficients(sme)
        # cfe = paste(u[,1], collapse=",")
        # see = paste(u[,4], collapse=",")

        # ma = gee(Y ~ X1 + X2 + X3, id=Id, family=binomial,
        #         corstr="AR-M")
        # sma = summary(ma)
        # u = coefficients(sma)
        # cfa = paste(u[,1], collapse=",")
        # sea = paste(u[,4], collapse=",")

        # sprintf("cf = [[%s],[%s],[%s]]", cfi, cfe, cfa)
        # sprintf("se = [[%s],[%s],[%s]]", sei, see, sea)

        endog, exog, group = load_data("gee_logistic_1.csv")

        # Time values for the autoregressive model
        T = np.zeros(len(endog))
        idx = set(group)
        for ii in idx:
            jj = np.flatnonzero(group == ii)
            T[jj] = lrange(len(jj))

        family = families.Binomial()
        ve = cov_struct.Exchangeable()
        vi = cov_struct.Independence()
        va = cov_struct.Autoregressive(grid=False)

        # From R gee
        cf = [[0.0167272965285882, 1.13038654425893,
               -1.86896345082962, 1.09397608331333],
              [0.0178982283915449, 1.13118798191788,
               -1.86133518416017, 1.08944256230299],
              [0.0109621937947958, 1.13226505028438,
               -1.88278757333046, 1.09954623769449]]
        se = [[0.127291720283049, 0.166725808326067,
               0.192430061340865, 0.173141068839597],
              [0.127045031730155, 0.165470678232842,
               0.192052750030501, 0.173174779369249],
              [0.127240302296444, 0.170554083928117,
               0.191045527104503, 0.169776150974586]]

        for j, v in enumerate((vi, ve, va)):
            md = gee.GEE(endog, exog, group, T, family, v)
            mdf = md.fit()
            if id(v) != id(va):
                assert_almost_equal(mdf.params, cf[j], decimal=6)
                assert_almost_equal(mdf.standard_errors(), se[j],
                                    decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:, None], group[:, None], exog[:, 1:]),
                           axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y", "Id", ] + ["X%d" % (k + 1)
                                     for k in range(exog.shape[1] - 1)]
        for j, v in enumerate((vi, ve)):
            md = gee.GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D,
                                      family=family, cov_struct=v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=6)
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=6)

        # FIXME: do not leave commented-out
        # Check for run-time exceptions in summary
        # print(mdf.summary())

    def test_autoregressive(self):

        dep_params_true = [0, 0.589208623896, 0.559823804948]

        params_true = [[1.08043787, 1.12709319, 0.90133927],
                       [0.9613677, 1.05826987, 0.90832055],
                       [1.05370439, 0.96084864, 0.93923374]]

        np.random.seed(342837482)

        num_group = 100
        ar_param = 0.5
        k = 3

        ga = families.Gaussian()

        for gsize in 1, 2, 3:

            ix = np.arange(gsize)[:, None] - np.arange(gsize)[None, :]
            ix = np.abs(ix)
            cmat = ar_param ** ix
            cmat_r = np.linalg.cholesky(cmat)

            endog = []
            exog = []
            groups = []
            for i in range(num_group):
                x = np.random.normal(size=(gsize, k))
                exog.append(x)
                expval = x.sum(1)
                errors = np.dot(cmat_r, np.random.normal(size=gsize))
                endog.append(expval + errors)
                groups.append(i * np.ones(gsize))

            endog = np.concatenate(endog)
            groups = np.concatenate(groups)
            exog = np.concatenate(exog, axis=0)

            ar = cov_struct.Autoregressive(grid=False)
            md = gee.GEE(endog, exog, groups, family=ga, cov_struct=ar)
            mdf = md.fit()
            assert_almost_equal(ar.dep_params, dep_params_true[gsize - 1])
            assert_almost_equal(mdf.params, params_true[gsize - 1])

    def test_post_estimation(self):

        family = families.Gaussian()
        endog, exog, group = load_data("gee_linear_1.csv")

        ve = cov_struct.Exchangeable()

        md = gee.GEE(endog, exog, group, None, family, ve)
        mdf = md.fit()

        assert_almost_equal(np.dot(exog, mdf.params),
                            mdf.fittedvalues)
        assert_almost_equal(endog - np.dot(exog, mdf.params),
                            mdf.resid)

    def test_scoretest(self):
        # Regression tests

        np.random.seed(6432)
        n = 200  # Must be divisible by 4
        exog = np.random.normal(size=(n, 4))
        endog = exog[:, 0] + exog[:, 1] + exog[:, 2]
        endog += 3 * np.random.normal(size=n)
        group = np.kron(np.arange(n / 4), np.ones(4))

        # Test under the null.
        L = np.array([[1., -1, 0, 0]])
        R = np.array([0., ])
        family = families.Gaussian()
        va = cov_struct.Independence()
        mod1 = gee.GEE(endog, exog, group, family=family,
                       cov_struct=va, constraint=(L, R))
        res1 = mod1.fit()
        assert_almost_equal(res1.score_test()["statistic"],
                            1.08126334)
        assert_almost_equal(res1.score_test()["p-value"],
                            0.2984151086)

        # Test under the alternative.
        L = np.array([[1., -1, 0, 0]])
        R = np.array([1.0, ])
        family = families.Gaussian()
        va = cov_struct.Independence()
        mod2 = gee.GEE(endog, exog, group, family=family,
                       cov_struct=va, constraint=(L, R))
        res2 = mod2.fit()
        assert_almost_equal(res2.score_test()["statistic"],
                            3.491110965)
        assert_almost_equal(res2.score_test()["p-value"],
                            0.0616991659)

        # Compare to Wald tests
        exog = np.random.normal(size=(n, 2))
        L = np.array([[1, -1]])
        R = np.array([0.])
        f = np.r_[1, -1]
        for i in range(10):
            endog = exog[:, 0] + (0.5 + i / 10.) * exog[:, 1] +\
                np.random.normal(size=n)
            family = families.Gaussian()
            va = cov_struct.Independence()
            mod0 = gee.GEE(endog, exog, group, family=family,
                           cov_struct=va)
            rslt0 = mod0.fit()
            family = families.Gaussian()
            va = cov_struct.Independence()
            mod1 = gee.GEE(endog, exog, group, family=family,
                           cov_struct=va, constraint=(L, R))
            res1 = mod1.fit()
            se = np.sqrt(np.dot(f, np.dot(rslt0.cov_params(), f)))
            wald_z = np.dot(f, rslt0.params) / se
            wald_p = 2 * norm.cdf(-np.abs(wald_z))
            score_p = res1.score_test()["p-value"]
            assert_array_less(np.abs(wald_p - score_p), 0.02)

    @pytest.mark.parametrize("cov_struct", [cov_struct.Independence,
                                            cov_struct.Exchangeable])
    def test_compare_score_test(self, cov_struct):

        np.random.seed(6432)
        n = 200  # Must be divisible by 4
        exog = np.random.normal(size=(n, 4))
        group = np.kron(np.arange(n / 4), np.ones(4))

        exog_sub = exog[:, [0, 3]]
        endog = exog_sub.sum(1) + 3 * np.random.normal(size=n)

        L = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0]])
        R = np.zeros(2)
        mod_lr = gee.GEE(endog, exog, group, constraint=(L, R),
                         cov_struct=cov_struct())
        mod_lr.fit()

        mod_sub = gee.GEE(endog, exog_sub, group, cov_struct=cov_struct())
        res_sub = mod_sub.fit()

        for call_fit in [False, True]:
            mod = gee.GEE(endog, exog, group, cov_struct=cov_struct())
            if call_fit:
                # Should work with or without fitting the parent model
                mod.fit()
            score_results = mod.compare_score_test(res_sub)
            assert_almost_equal(
                score_results["statistic"],
                mod_lr.score_test_results["statistic"])
            assert_almost_equal(
                score_results["p-value"],
                mod_lr.score_test_results["p-value"])
            assert_almost_equal(
                score_results["df"],
                mod_lr.score_test_results["df"])

    def test_compare_score_test_warnings(self):

        np.random.seed(6432)
        n = 200  # Must be divisible by 4
        exog = np.random.normal(size=(n, 4))
        group = np.kron(np.arange(n / 4), np.ones(4))
        exog_sub = exog[:, [0, 3]]
        endog = exog_sub.sum(1) + 3 * np.random.normal(size=n)

        # Mismatched cov_struct
        with assert_warns(UserWarning):
            mod_sub = gee.GEE(endog, exog_sub, group,
                              cov_struct=cov_struct.Exchangeable())
            res_sub = mod_sub.fit()
            mod = gee.GEE(endog, exog, group,
                          cov_struct=cov_struct.Independence())
            mod.compare_score_test(res_sub)  # smoketest

        # Mismatched family
        with assert_warns(UserWarning):
            mod_sub = gee.GEE(endog, exog_sub, group,
                              family=families.Gaussian())
            res_sub = mod_sub.fit()
            mod = gee.GEE(endog, exog, group, family=families.Poisson())
            mod.compare_score_test(res_sub)  # smoketest

        # Mismatched size
        with assert_raises(Exception):
            mod_sub = gee.GEE(endog, exog_sub, group)
            res_sub = mod_sub.fit()
            mod = gee.GEE(endog[0:100], exog[:100, :], group[0:100])
            mod.compare_score_test(res_sub)  # smoketest

        # Mismatched weights
        with assert_warns(UserWarning):
            w = np.random.uniform(size=n)
            mod_sub = gee.GEE(endog, exog_sub, group, weights=w)
            res_sub = mod_sub.fit()
            mod = gee.GEE(endog, exog, group)
            mod.compare_score_test(res_sub)  # smoketest

        # Parent and submodel are the same dimension
        with pytest.warns(UserWarning):
            w = np.random.uniform(size=n)
            mod_sub = gee.GEE(endog, exog, group)
            res_sub = mod_sub.fit()
            mod = gee.GEE(endog, exog, group)
            mod.compare_score_test(res_sub)  # smoketest

    def test_constraint_covtype(self):
        # Test constraints with different cov types
        np.random.seed(6432)
        n = 200
        exog = np.random.normal(size=(n, 4))
        endog = exog[:, 0] + exog[:, 1] + exog[:, 2]
        endog += 3 * np.random.normal(size=n)
        group = np.kron(np.arange(n / 4), np.ones(4))
        L = np.array([[1., -1, 0, 0]])
        R = np.array([0., ])
        family = families.Gaussian()
        va = cov_struct.Independence()
        for cov_type in "robust", "naive", "bias_reduced":
            model = gee.GEE(endog, exog, group, family=family,
                            cov_struct=va, constraint=(L, R))
            result = model.fit(cov_type=cov_type)
            result.standard_errors(cov_type=cov_type)
            assert_allclose(result.cov_robust.shape, np.r_[4, 4])
            assert_allclose(result.cov_naive.shape, np.r_[4, 4])
            if cov_type == "bias_reduced":
                assert_allclose(result.cov_robust_bc.shape, np.r_[4, 4])

    def test_linear(self):
        # library(gee)

        # Z = read.csv("results/gee_linear_1.csv", header=FALSE)
        # Y = Z[,2]
        # Id = Z[,1]
        # X1 = Z[,3]
        # X2 = Z[,4]
        # X3 = Z[,5]
        # mi = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian,
        #         corstr="independence", tol=1e-8, maxit=100)
        # smi = summary(mi)
        # u = coefficients(smi)

        # cfi = paste(u[,1], collapse=",")
        # sei = paste(u[,4], collapse=",")

        # me = gee(Y ~ X1 + X2 + X3, id=Id, family=gaussian,
        #         corstr="exchangeable", tol=1e-8, maxit=100)
        # sme = summary(me)
        # u = coefficients(sme)

        # cfe = paste(u[,1], collapse=",")
        # see = paste(u[,4], collapse=",")

        # sprintf("cf = [[%s],[%s]]", cfi, cfe)
        # sprintf("se = [[%s],[%s]]", sei, see)

        family = families.Gaussian()

        endog, exog, group = load_data("gee_linear_1.csv")

        vi = cov_struct.Independence()
        ve = cov_struct.Exchangeable()

        # From R gee
        cf = [[-0.01850226507491, 0.81436304278962,
               -1.56167635393184, 0.794239361055003],
              [-0.0182920577154767, 0.814898414022467,
               -1.56194040106201, 0.793499517527478]]
        se = [[0.0440733554189401, 0.0479993639119261,
               0.0496045952071308, 0.0479467597161284],
              [0.0440369906460754, 0.0480069787567662,
               0.049519758758187, 0.0479760443027526]]

        for j, v in enumerate((vi, ve)):
            md = gee.GEE(endog, exog, group, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=10)

        # Test with formulas
        D = np.concatenate((endog[:, None], group[:, None], exog[:, 1:]),
                           axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y", "Id", ] + ["X%d" % (k + 1)
                                     for k in range(exog.shape[1] - 1)]
        for j, v in enumerate((vi, ve)):
            md = gee.GEE.from_formula("Y ~ X1 + X2 + X3", "Id", D,
                                      family=family, cov_struct=v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=10)
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=10)

    def test_linear_constrained(self):

        family = families.Gaussian()

        np.random.seed(34234)
        exog = np.random.normal(size=(300, 4))
        exog[:, 0] = 1
        endog = np.dot(exog, np.r_[1, 1, 0, 0.2]) +\
            np.random.normal(size=300)
        group = np.kron(np.arange(100), np.r_[1, 1, 1])

        vi = cov_struct.Independence()
        ve = cov_struct.Exchangeable()

        L = np.r_[[[0, 0, 0, 1]]]
        R = np.r_[0, ]

        for j, v in enumerate((vi, ve)):
            md = gee.GEE(endog, exog, group, None, family, v,
                         constraint=(L, R))
            mdf = md.fit()
            assert_almost_equal(mdf.params[3], 0, decimal=10)

    def test_nested_linear(self):

        family = families.Gaussian()

        endog, exog, group = load_data("gee_nested_linear_1.csv")

        group_n = []
        for i in range(endog.shape[0] // 10):
            group_n.extend([0, ] * 5)
            group_n.extend([1, ] * 5)
        group_n = np.array(group_n)[:, None]

        dp = cov_struct.Independence()
        md = gee.GEE(endog, exog, group, None, family, dp)
        mdf1 = md.fit()

        # From statsmodels.GEE (not an independent test)
        cf = np.r_[-0.1671073,  1.00467426, -2.01723004,  0.97297106]
        se = np.r_[0.08629606,  0.04058653,  0.04067038,  0.03777989]
        assert_almost_equal(mdf1.params, cf, decimal=6)
        assert_almost_equal(mdf1.standard_errors(), se,
                            decimal=6)

        ne = cov_struct.Nested()
        md = gee.GEE(endog, exog, group, None, family, ne,
                     dep_data=group_n)
        mdf2 = md.fit(start_params=mdf1.params)

        # From statsmodels.GEE (not an independent test)
        cf = np.r_[-0.16655319,  1.02183688, -2.00858719,  1.00101969]
        se = np.r_[0.08632616,  0.02913582,  0.03114428,  0.02893991]
        assert_almost_equal(mdf2.params, cf, decimal=6)
        assert_almost_equal(mdf2.standard_errors(), se,
                            decimal=6)

        smry = mdf2.cov_struct.summary()
        assert_allclose(
            smry.Variance,
            np.r_[1.043878, 0.611656, 1.421205],
            atol=1e-5, rtol=1e-5)

    def test_nested_pandas(self):

        np.random.seed(4234)
        n = 10000

        # Outer groups
        groups = np.kron(np.arange(n // 100), np.ones(100)).astype(int)

        # Inner groups
        groups1 = np.kron(np.arange(n // 50), np.ones(50)).astype(int)
        groups2 = np.kron(np.arange(n // 10), np.ones(10)).astype(int)

        # Group effects
        groups_e = np.random.normal(size=n // 100)
        groups1_e = 2 * np.random.normal(size=n // 50)
        groups2_e = 3 * np.random.normal(size=n // 10)

        y = groups_e[groups] + groups1_e[groups1] + groups2_e[groups2]
        y += 0.5 * np.random.normal(size=n)

        df = pd.DataFrame({"y": y, "TheGroups": groups,
                           "groups1": groups1, "groups2": groups2})

        model = gee.GEE.from_formula("y ~ 1", groups="TheGroups",
                                     dep_data="0 + groups1 + groups2",
                                     cov_struct=cov_struct.Nested(),
                                     data=df)
        result = model.fit()

        # The true variances are 1, 4, 9, 0.25
        smry = result.cov_struct.summary()
        assert_allclose(
            smry.Variance,
            np.r_[1.437299, 4.421543, 8.905295, 0.258480],
            atol=1e-5, rtol=1e-5)

    def test_ordinal(self):

        family = families.Binomial()

        endog, exog, groups = load_data("gee_ordinal_1.csv",
                                        icept=False)

        va = cov_struct.GlobalOddsRatio("ordinal")

        mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        rslt = mod.fit()

        # Regression test
        cf = np.r_[1.09250002, 0.0217443, -0.39851092, -0.01812116,
                   0.03023969, 1.18258516, 0.01803453, -1.10203381]
        assert_almost_equal(rslt.params, cf, decimal=5)

        # Regression test
        se = np.r_[0.10883461, 0.10330197, 0.11177088, 0.05486569,
                   0.05997153, 0.09168148, 0.05953324, 0.0853862]
        assert_almost_equal(rslt.bse, se, decimal=5)

        # Check that we get the correct results type
        assert_equal(type(rslt), gee.OrdinalGEEResultsWrapper)
        assert_equal(type(rslt._results), gee.OrdinalGEEResults)

    @pytest.mark.smoke
    def test_ordinal_formula(self):

        np.random.seed(434)
        n = 40
        y = np.random.randint(0, 3, n)
        groups = np.arange(n)
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)

        df = pd.DataFrame({"y": y, "groups": groups, "x1": x1, "x2": x2})

        model = gee.OrdinalGEE.from_formula("y ~ 0 + x1 + x2", groups, data=df)
        model.fit()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gee.NominalGEE.from_formula("y ~ 0 + x1 + x2", groups,
                                                data=df)
            model.fit()

    @pytest.mark.smoke
    def test_ordinal_independence(self):

        np.random.seed(434)
        n = 40
        y = np.random.randint(0, 3, n)
        groups = np.kron(np.arange(n / 2), np.r_[1, 1])
        x = np.random.normal(size=(n, 1))

        odi = cov_struct.OrdinalIndependence()
        model1 = gee.OrdinalGEE(y, x, groups, cov_struct=odi)
        model1.fit()

    @pytest.mark.smoke
    def test_nominal_independence(self):

        np.random.seed(434)
        n = 40
        y = np.random.randint(0, 3, n)
        groups = np.kron(np.arange(n / 2), np.r_[1, 1])
        x = np.random.normal(size=(n, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nmi = cov_struct.NominalIndependence()
            model1 = gee.NominalGEE(y, x, groups, cov_struct=nmi)
            model1.fit()

    @pytest.mark.smoke
    @pytest.mark.matplotlib
    def test_ordinal_plot(self, close_figures):
        family = families.Binomial()

        endog, exog, groups = load_data("gee_ordinal_1.csv",
                                        icept=False)

        va = cov_struct.GlobalOddsRatio("ordinal")

        mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        rslt = mod.fit()

        fig = rslt.plot_distribution()
        assert_equal(isinstance(fig, plt.Figure), True)

    def test_nominal(self):

        endog, exog, groups = load_data("gee_nominal_1.csv",
                                        icept=False)

        # Test with independence correlation
        va = cov_struct.Independence()
        mod1 = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        rslt1 = mod1.fit()

        # Regression test
        cf1 = np.r_[0.450009, 0.451959, -0.918825, -0.468266]
        se1 = np.r_[0.08915936, 0.07005046, 0.12198139, 0.08281258]
        assert_allclose(rslt1.params, cf1, rtol=1e-5, atol=1e-5)
        assert_allclose(rslt1.standard_errors(), se1, rtol=1e-5, atol=1e-5)

        # Test with global odds ratio dependence
        va = cov_struct.GlobalOddsRatio("nominal")
        mod2 = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        rslt2 = mod2.fit(start_params=rslt1.params)

        # Regression test
        cf2 = np.r_[0.455365, 0.415334, -0.916589, -0.502116]
        se2 = np.r_[0.08803614, 0.06628179, 0.12259726, 0.08411064]
        assert_allclose(rslt2.params, cf2, rtol=1e-5, atol=1e-5)
        assert_allclose(rslt2.standard_errors(), se2, rtol=1e-5, atol=1e-5)

        # Make sure we get the correct results type
        assert_equal(type(rslt1), gee.NominalGEEResultsWrapper)
        assert_equal(type(rslt1._results), gee.NominalGEEResults)

    def test_poisson(self):
        # library(gee)
        # Z = read.csv("results/gee_poisson_1.csv", header=FALSE)
        # Y = Z[,2]
        # Id = Z[,1]
        # X1 = Z[,3]
        # X2 = Z[,4]
        # X3 = Z[,5]
        # X4 = Z[,6]
        # X5 = Z[,7]

        # mi = gee(Y ~ X1 + X2 + X3 + X4 + X5, id=Id, family=poisson,
        #        corstr="independence", scale.fix=TRUE)
        # smi = summary(mi)
        # u = coefficients(smi)
        # cfi = paste(u[,1], collapse=",")
        # sei = paste(u[,4], collapse=",")

        # me = gee(Y ~ X1 + X2 + X3 + X4 + X5, id=Id, family=poisson,
        #        corstr="exchangeable", scale.fix=TRUE)
        # sme = summary(me)

        # u = coefficients(sme)
        # cfe = paste(u[,1], collapse=",")
        # see = paste(u[,4], collapse=",")

        # sprintf("cf = [[%s],[%s]]", cfi, cfe)
        # sprintf("se = [[%s],[%s]]", sei, see)

        family = families.Poisson()

        endog, exog, group_n = load_data("gee_poisson_1.csv")

        vi = cov_struct.Independence()
        ve = cov_struct.Exchangeable()

        # From R gee
        cf = [[-0.0364450410793481, -0.0543209391301178,
               0.0156642711741052, 0.57628591338724,
               -0.00465659951186211, -0.477093153099256],
              [-0.0315615554826533, -0.0562589480840004,
               0.0178419412298561, 0.571512795340481,
               -0.00363255566297332, -0.475971696727736]]
        se = [[0.0611309237214186, 0.0390680524493108,
               0.0334234174505518, 0.0366860768962715,
               0.0304758505008105, 0.0316348058881079],
              [0.0610840153582275, 0.0376887268649102,
               0.0325168379415177, 0.0369786751362213,
               0.0296141014225009, 0.0306115470200955]]

        for j, v in enumerate((vi, ve)):
            md = gee.GEE(endog, exog, group_n, None, family, v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=5)
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=6)

        # Test with formulas
        D = np.concatenate((endog[:, None], group_n[:, None],
                            exog[:, 1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y", "Id", ] + ["X%d" % (k + 1)
                                     for k in range(exog.shape[1] - 1)]
        for j, v in enumerate((vi, ve)):
            md = gee.GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", "Id",
                                      D, family=family, cov_struct=v)
            mdf = md.fit()
            assert_almost_equal(mdf.params, cf[j], decimal=5)
            assert_almost_equal(mdf.standard_errors(), se[j],
                                decimal=6)
            # print(mdf.params)

    def test_groups(self):
        # Test various group structures (nonconsecutive, different
        # group sizes, not ordered, string labels)

        np.random.seed(234)
        n = 40
        x = np.random.normal(size=(n, 2))
        y = np.random.normal(size=n)

        # groups with unequal group sizes
        groups = np.kron(np.arange(n / 4), np.ones(4))
        groups[8:12] = 3
        groups[34:36] = 9

        model1 = gee.GEE(y, x, groups=groups)
        result1 = model1.fit()

        # Unordered groups
        ix = np.random.permutation(n)
        y1 = y[ix]
        x1 = x[ix, :]
        groups1 = groups[ix]

        model2 = gee.GEE(y1, x1, groups=groups1)
        result2 = model2.fit()

        assert_allclose(result1.params, result2.params)
        assert_allclose(result1.tvalues, result2.tvalues)

        # group labels are strings
        mp = {}
        import string
        for j, g in enumerate(set(groups)):
            mp[g] = string.ascii_letters[j:j + 4]
        groups2 = [mp[g] for g in groups]

        model3 = gee.GEE(y, x, groups=groups2)
        result3 = model3.fit()

        assert_allclose(result1.params, result3.params)
        assert_allclose(result1.tvalues, result3.tvalues)

    def test_compare_OLS(self):
        # Gaussian GEE with independence correlation should agree
        # exactly with OLS for parameter estimates and standard errors
        # derived from the naive covariance estimate.

        vs = cov_struct.Independence()
        family = families.Gaussian()

        np.random.seed(34234)
        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(lrange(20), np.ones(5))

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = gee.GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                  family=family, cov_struct=vs)
        mdf = md.fit()

        ols = lm.OLS.from_formula("Y ~ X1 + X2 + X3", data=D).fit()

        # do not use wrapper, asserts_xxx do not work
        ols = ols._results

        assert_almost_equal(ols.params, mdf.params, decimal=10)

        se = mdf.standard_errors(cov_type="naive")
        assert_almost_equal(ols.bse, se, decimal=10)

        naive_tvalues = mdf.params / np.sqrt(np.diag(mdf.cov_naive))
        assert_almost_equal(naive_tvalues, ols.tvalues, decimal=10)

    def test_formulas(self):
        # Check formulas, especially passing groups and time as either
        # variable names or arrays.

        n = 100
        np.random.seed(34234)
        Y = np.random.normal(size=n)
        X1 = np.random.normal(size=n)
        mat = np.concatenate((np.ones((n, 1)), X1[:, None]), axis=1)
        Time = np.random.uniform(size=n)
        groups = np.kron(lrange(20), np.ones(5))

        data = pd.DataFrame({"Y": Y, "X1": X1, "Time": Time, "groups": groups})

        va = cov_struct.Autoregressive(grid=False)
        family = families.Gaussian()

        mod1 = gee.GEE(Y, mat, groups, time=Time, family=family,
                       cov_struct=va)
        rslt1 = mod1.fit()

        mod2 = gee.GEE.from_formula("Y ~ X1", groups, data, time=Time,
                                    family=family, cov_struct=va)
        rslt2 = mod2.fit()

        mod3 = gee.GEE.from_formula("Y ~ X1", groups, data, time="Time",
                                    family=family, cov_struct=va)
        rslt3 = mod3.fit()

        mod4 = gee.GEE.from_formula("Y ~ X1", "groups", data, time=Time,
                                    family=family, cov_struct=va)
        rslt4 = mod4.fit()

        mod5 = gee.GEE.from_formula("Y ~ X1", "groups", data, time="Time",
                                    family=family, cov_struct=va)
        rslt5 = mod5.fit()

        assert_almost_equal(rslt1.params, rslt2.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt3.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt4.params, decimal=8)
        assert_almost_equal(rslt1.params, rslt5.params, decimal=8)

        check_wrapper(rslt2)

    def test_compare_logit(self):

        vs = cov_struct.Independence()
        family = families.Binomial()

        np.random.seed(34234)
        Y = 1 * (np.random.normal(size=100) < 0)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = gee.GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                    family=family, cov_struct=vs)
        rslt1 = mod1.fit()

        mod2 = discrete.Logit.from_formula("Y ~ X1 + X2 + X3", data=D)
        rslt2 = mod2.fit(disp=False)

        assert_almost_equal(rslt1.params.values, rslt2.params.values,
                            decimal=10)

    def test_compare_poisson(self):

        vs = cov_struct.Independence()
        family = families.Poisson()

        np.random.seed(34234)
        Y = np.ceil(-np.log(np.random.uniform(size=100)))
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = gee.GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                    family=family, cov_struct=vs)
        rslt1 = mod1.fit()

        mod2 = discrete.Poisson.from_formula("Y ~ X1 + X2 + X3", data=D)
        rslt2 = mod2.fit(disp=False)

        assert_almost_equal(rslt1.params.values, rslt2.params.values,
                            decimal=10)

    def test_predict(self):

        n = 50
        np.random.seed(4324)
        X1 = np.random.normal(size=n)
        X2 = np.random.normal(size=n)
        groups = np.kron(np.arange(n / 2), np.r_[1, 1])
        offset = np.random.uniform(1, 2, size=n)
        Y = np.random.normal(0.1 * (X1 + X2) + offset, size=n)
        data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "groups": groups,
                             "offset": offset})

        fml = "Y ~ X1 + X2"
        model = gee.GEE.from_formula(fml, groups, data,
                                     family=families.Gaussian(),
                                     offset="offset")
        result = model.fit(start_params=[0, 0.1, 0.1])
        assert_equal(result.converged, True)

        pred1 = result.predict()
        pred2 = result.predict(offset=data.offset)
        pred3 = result.predict(exog=data[["X1", "X2"]], offset=data.offset)
        pred4 = result.predict(exog=data[["X1", "X2"]], offset=0 * data.offset)
        pred5 = result.predict(offset=0 * data.offset)

        assert_allclose(pred1, pred2)
        assert_allclose(pred1, pred3)
        assert_allclose(pred1, pred4 + data.offset)
        assert_allclose(pred1, pred5 + data.offset)

        x1_new = np.random.normal(size=10)
        x2_new = np.random.normal(size=10)
        new_exog = pd.DataFrame({"X1": x1_new, "X2": x2_new})
        pred6 = result.predict(exog=new_exog)
        params = np.asarray(result.params)
        pred6_correct = params[0] + params[1] * x1_new + params[2] * x2_new
        assert_allclose(pred6, pred6_correct)

    def test_stationary_grid(self):

        endog = np.r_[4, 2, 3, 1, 4, 5, 6, 7, 8, 3, 2, 4.]
        exog = np.r_[2, 3, 1, 4, 3, 2, 5, 4, 5, 6, 3, 2]
        group = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        exog = tools.add_constant(exog)

        cs = cov_struct.Stationary(max_lag=2, grid=True)
        model = gee.GEE(endog, exog, group, cov_struct=cs)
        result = model.fit()
        se = result.bse * np.sqrt(12 / 9.)  # Stata adjustment

        assert_allclose(cs.covariance_matrix(np.r_[1, 1, 1], 0)[0].sum(),
                        6.4633538285149452)

        # Obtained from Stata using:
        # xtgee y x, i(g) vce(robust) corr(Stationary2)
        assert_allclose(result.params, np.r_[
                        4.463968, -0.0386674], rtol=1e-5, atol=1e-5)
        assert_allclose(se, np.r_[0.5217202, 0.2800333], rtol=1e-5, atol=1e-5)

    def test_stationary_nogrid(self):

        # First test special case where the data follow a grid but we
        # fit using nogrid
        endog = np.r_[4, 2, 3, 1, 4, 5, 6, 7, 8, 3, 2, 4.]
        exog = np.r_[2, 3, 1, 4, 3, 2, 5, 4, 5, 6, 3, 2]
        time = np.r_[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        group = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

        exog = tools.add_constant(exog)

        model = gee.GEE(endog, exog, group,
                        cov_struct=cov_struct.Stationary(max_lag=2,
                                                         grid=False))
        result = model.fit()
        se = result.bse * np.sqrt(12 / 9.)  # Stata adjustment

        # Obtained from Stata using:
        # xtgee y x, i(g) vce(robust) corr(Stationary2)
        assert_allclose(result.params, np.r_[
                        4.463968, -0.0386674], rtol=1e-5, atol=1e-5)
        assert_allclose(se, np.r_[0.5217202, 0.2800333], rtol=1e-5, atol=1e-5)

        # Smoke test for no grid  # TODO: pytest.mark.smoke>
        time = np.r_[0, 1, 3, 0, 2, 3, 0, 2, 3, 0, 1, 2][:, None]
        model = gee.GEE(endog, exog, group, time=time,
                        cov_struct=cov_struct.Stationary(max_lag=4,
                                                         grid=False))
        model.fit()

    def test_predict_exposure(self):

        n = 50
        np.random.seed(34234)
        X1 = np.random.normal(size=n)
        X2 = np.random.normal(size=n)
        groups = np.kron(np.arange(25), np.r_[1, 1])
        offset = np.random.uniform(1, 2, size=n)
        exposure = np.random.uniform(1, 2, size=n)
        Y = np.random.poisson(0.1 * (X1 + X2) + offset +
                              np.log(exposure), size=n)
        data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "groups": groups,
                             "offset": offset, "exposure": exposure})

        fml = "Y ~ X1 + X2"
        model = gee.GEE.from_formula(fml, groups, data,
                                     family=families.Poisson(),
                                     offset="offset", exposure="exposure")
        result = model.fit()
        assert_equal(result.converged, True)

        pred1 = result.predict()
        pred2 = result.predict(offset=data["offset"])
        pred3 = result.predict(exposure=data["exposure"])
        pred4 = result.predict(
            offset=data["offset"], exposure=data["exposure"])
        pred5 = result.predict(exog=data[-10:],
                               offset=data["offset"][-10:],
                               exposure=data["exposure"][-10:])
        # without patsy
        pred6 = result.predict(exog=result.model.exog[-10:],
                               offset=data["offset"][-10:],
                               exposure=data["exposure"][-10:],
                               transform=False)
        assert_allclose(pred1, pred2)
        assert_allclose(pred1, pred3)
        assert_allclose(pred1, pred4)
        assert_allclose(pred1[-10:], pred5)
        assert_allclose(pred1[-10:], pred6)

    def test_predict_exposure_lists(self):

        n = 50
        np.random.seed(34234)
        exog = [[1, np.random.normal(), np.random.normal()] for _ in range(n)]
        groups = list(np.kron(np.arange(25), np.r_[1, 1]))
        offset = list(np.random.uniform(1, 2, size=n))
        exposure = list(np.random.uniform(1, 2, size=n))
        endog = [
            np.random.poisson(
                0.1 * (exog_i[1] + exog_i[2]) + offset_i + np.log(exposure_i)
            )
            for exog_i, offset_i, exposure_i in zip(exog, offset, exposure)
        ]

        model = gee.GEE(
            endog,
            exog,
            groups=groups,
            family=families.Poisson(),
            offset=offset,
            exposure=exposure,
        )
        result = model.fit()

        pred1 = result.predict()
        pred2 = result.predict(exog=exog, offset=offset, exposure=exposure)

        assert_allclose(pred1, pred2)

    def test_offset_formula(self):
        # Test various ways of passing offset and exposure to `from_formula`.

        n = 50
        np.random.seed(34234)
        X1 = np.random.normal(size=n)
        X2 = np.random.normal(size=n)
        groups = np.kron(np.arange(25), np.r_[1, 1])
        offset = np.random.uniform(1, 2, size=n)
        exposure = np.exp(offset)
        Y = np.random.poisson(0.1 * (X1 + X2) + 2 * offset, size=n)
        data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "groups": groups,
                             "offset": offset, "exposure": exposure})

        fml = "Y ~ X1 + X2"
        model1 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      offset="offset")
        result1 = model1.fit()
        assert_equal(result1.converged, True)

        model2 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      offset=offset)
        result2 = model2.fit(start_params=result1.params)
        assert_allclose(result1.params, result2.params)
        assert_equal(result2.converged, True)

        model3 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      exposure=exposure)
        result3 = model3.fit(start_params=result1.params)
        assert_allclose(result1.params, result3.params)
        assert_equal(result3.converged, True)

        model4 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      exposure="exposure")
        result4 = model4.fit(start_params=result1.params)
        assert_allclose(result1.params, result4.params)
        assert_equal(result4.converged, True)

        model5 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      exposure="exposure", offset="offset")
        result5 = model5.fit()
        assert_equal(result5.converged, True)

        model6 = gee.GEE.from_formula(fml, groups, data,
                                      family=families.Poisson(),
                                      offset=2 * offset)
        result6 = model6.fit(start_params=result5.params)
        assert_allclose(result5.params, result6.params)
        assert_equal(result6.converged, True)

    def test_sensitivity(self):

        va = cov_struct.Exchangeable()
        family = families.Gaussian()

        np.random.seed(34234)
        n = 100
        Y = np.random.normal(size=n)
        X1 = np.random.normal(size=n)
        X2 = np.random.normal(size=n)
        groups = np.kron(np.arange(50), np.r_[1, 1])

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2})

        mod = gee.GEE.from_formula("Y ~ X1 + X2", groups, D,
                                   family=family, cov_struct=va)
        rslt = mod.fit()
        ps = rslt.params_sensitivity(0, 0.5, 2)
        assert_almost_equal(len(ps), 2)
        assert_almost_equal([x.cov_struct.dep_params for x in ps],
                            [0.0, 0.5])

        # Regression test
        assert_almost_equal([np.asarray(x.params)[0] for x in ps],
                            [0.1696214707458818, 0.17836097387799127])

    def test_equivalence(self):
        """
        The Equivalence covariance structure can represent an
        exchangeable covariance structure.  Here we check that the
        results are identical using the two approaches.
        """

        np.random.seed(3424)
        endog = np.random.normal(size=20)
        exog = np.random.normal(size=(20, 2))
        exog[:, 0] = 1
        groups = np.kron(np.arange(5), np.ones(4))
        groups[12:] = 3  # Create unequal size groups

        # Set up an Equivalence covariance structure to mimic an
        # Exchangeable covariance structure.
        pairs = {}
        start = [0, 4, 8, 12]
        for k in range(4):
            pairs[k] = {}

            # Diagonal values (variance parameters)
            if k < 3:
                pairs[k][0] = (start[k] + np.r_[0, 1, 2, 3],
                               start[k] + np.r_[0, 1, 2, 3])
            else:
                pairs[k][0] = (start[k] + np.r_[0, 1, 2, 3, 4, 5, 6, 7],
                               start[k] + np.r_[0, 1, 2, 3, 4, 5, 6, 7])

            # Off-diagonal pairs (covariance parameters)
            if k < 3:
                a, b = np.tril_indices(4, -1)
                pairs[k][1] = (start[k] + a, start[k] + b)
            else:
                a, b = np.tril_indices(8, -1)
                pairs[k][1] = (start[k] + a, start[k] + b)

        ex = cov_struct.Exchangeable()
        model1 = gee.GEE(endog, exog, groups, cov_struct=ex)
        result1 = model1.fit()

        for return_cov in False, True:

            ec = cov_struct.Equivalence(pairs, return_cov=return_cov)
            model2 = gee.GEE(endog, exog, groups, cov_struct=ec)
            result2 = model2.fit()

            # Use large atol/rtol for the correlation case since there
            # are some small differences in the results due to degree
            # of freedom differences.
            if return_cov is True:
                atol, rtol = 1e-6, 1e-6
            else:
                atol, rtol = 1e-3, 1e-3
            assert_allclose(result1.params, result2.params,
                            atol=atol, rtol=rtol)
            assert_allclose(result1.bse, result2.bse, atol=atol, rtol=rtol)
            assert_allclose(result1.scale, result2.scale, atol=atol, rtol=rtol)

    def test_equivalence_from_pairs(self):

        np.random.seed(3424)
        endog = np.random.normal(size=50)
        exog = np.random.normal(size=(50, 2))
        exog[:, 0] = 1
        groups = np.kron(np.arange(5), np.ones(10))
        groups[30:] = 3  # Create unequal size groups

        # Set up labels.
        labels = np.kron(np.arange(5), np.ones(10)).astype(np.int32)
        labels = labels[np.random.permutation(len(labels))]

        eq = cov_struct.Equivalence(labels=labels, return_cov=True)
        model1 = gee.GEE(endog, exog, groups, cov_struct=eq)

        # Call this directly instead of letting init do it to get the
        # result before reindexing.
        eq._pairs_from_labels()

        # Make sure the size is correct to hold every element.
        for g in model1.group_labels:
            p = eq.pairs[g]
            vl = [len(x[0]) for x in p.values()]
            m = sum(groups == g)
            assert_allclose(sum(vl), m * (m + 1) / 2)

        # Check for duplicates.
        ixs = set()
        for g in model1.group_labels:
            for v in eq.pairs[g].values():
                for a, b in zip(v[0], v[1]):
                    ky = (a, b)
                    assert ky not in ixs
                    ixs.add(ky)

        # Smoke test  # TODO: pytest.mark.smoke?
        eq = cov_struct.Equivalence(labels=labels, return_cov=True)
        model1 = gee.GEE(endog, exog, groups, cov_struct=eq)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model1.fit(maxiter=2)


class CheckConsistency:

    start_params = None

    def test_cov_type(self):
        mod = self.mod
        res_robust = mod.fit(start_params=self.start_params)
        res_naive = mod.fit(start_params=self.start_params,
                            cov_type='naive')
        res_robust_bc = mod.fit(start_params=self.start_params,
                                cov_type='bias_reduced')

        # call summary to make sure it does not change cov_type
        res_naive.summary()
        res_robust_bc.summary()

        # check cov_type
        assert_equal(res_robust.cov_type, 'robust')
        assert_equal(res_naive.cov_type, 'naive')
        assert_equal(res_robust_bc.cov_type, 'bias_reduced')

        # check bse and cov_params
        # we are comparing different runs of the optimization
        # bse in ordinal and multinomial have an atol around 5e-10 for two
        # consecutive calls to fit.
        rtol = 1e-8
        for (res, cov_type, cov) in [
                (res_robust, 'robust', res_robust.cov_robust),
                (res_naive, 'naive', res_robust.cov_naive),
                (res_robust_bc, 'bias_reduced', res_robust_bc.cov_robust_bc)
        ]:
            bse = np.sqrt(np.diag(cov))
            assert_allclose(res.bse, bse, rtol=rtol)
            if cov_type != 'bias_reduced':
                # cov_type=naive shortcuts calculation of bias reduced
                # covariance for efficiency
                bse = res_naive.standard_errors(cov_type=cov_type)
                assert_allclose(res.bse, bse, rtol=rtol)
            assert_allclose(res.cov_params(), cov, rtol=rtol, atol=1e-10)
            assert_allclose(res.cov_params_default, cov, rtol=rtol, atol=1e-10)

        # assert that we do not have a copy
        assert_(res_robust.cov_params_default is res_robust.cov_robust)
        assert_(res_naive.cov_params_default is res_naive.cov_naive)
        assert_(res_robust_bc.cov_params_default is
                res_robust_bc.cov_robust_bc)

        # check exception for misspelled cov_type
        assert_raises(ValueError, mod.fit, cov_type='robust_bc')


class TestGEEPoissonCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):

        endog, exog, group_n = load_data("gee_poisson_1.csv")

        family = families.Poisson()
        vi = cov_struct.Independence()

        cls.mod = gee.GEE(endog, exog, group_n, None, family, vi)

        cls.start_params = np.array([-0.03644504, -0.05432094,  0.01566427,
                                     0.57628591, -0.0046566,  -0.47709315])

    def test_wrapper(self):

        endog, exog, group_n = load_data("gee_poisson_1.csv",
                                         icept=False)
        endog = pd.Series(endog)
        exog = pd.DataFrame(exog)
        group_n = pd.Series(group_n)

        family = families.Poisson()
        vi = cov_struct.Independence()

        mod = gee.GEE(endog, exog, group_n, None, family, vi)
        rslt2 = mod.fit()

        check_wrapper(rslt2)


class TestGEEPoissonFormulaCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):

        endog, exog, group_n = load_data("gee_poisson_1.csv")

        family = families.Poisson()
        vi = cov_struct.Independence()
        # Test with formulas
        D = np.concatenate((endog[:, None], group_n[:, None],
                            exog[:, 1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ["Y", "Id", ] + ["X%d" % (k + 1)
                                     for k in range(exog.shape[1] - 1)]

        cls.mod = gee.GEE.from_formula("Y ~ X1 + X2 + X3 + X4 + X5", "Id",
                                       D, family=family, cov_struct=vi)

        cls.start_params = np.array([-0.03644504, -0.05432094,  0.01566427,
                                     0.57628591, -0.0046566,  -0.47709315])


class TestGEEOrdinalCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):

        family = families.Binomial()

        endog, exog, groups = load_data("gee_ordinal_1.csv",
                                        icept=False)

        va = cov_struct.GlobalOddsRatio("ordinal")

        cls.mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        cls.start_params = np.array([1.09250002, 0.0217443, -0.39851092,
                                     -0.01812116, 0.03023969, 1.18258516,
                                     0.01803453, -1.10203381])

    def test_wrapper(self):

        endog, exog, groups = load_data("gee_ordinal_1.csv",
                                        icept=False)

        endog = pd.Series(endog, name='yendog')
        exog = pd.DataFrame(exog)
        groups = pd.Series(groups, name='the_group')

        family = families.Binomial()
        va = cov_struct.GlobalOddsRatio("ordinal")
        mod = gee.OrdinalGEE(endog, exog, groups, None, family, va)
        rslt2 = mod.fit()

        check_wrapper(rslt2)


class TestGEEMultinomialCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):

        endog, exog, groups = load_data("gee_nominal_1.csv",
                                        icept=False)

        # Test with independence correlation
        va = cov_struct.Independence()
        cls.mod = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        cls.start_params = np.array([0.44944752,  0.45569985, -0.92007064,
                                     -0.46766728])

    def test_wrapper(self):

        endog, exog, groups = load_data("gee_nominal_1.csv",
                                        icept=False)
        endog = pd.Series(endog, name='yendog')
        exog = pd.DataFrame(exog)
        groups = pd.Series(groups, name='the_group')

        va = cov_struct.Independence()
        mod = gee.NominalGEE(endog, exog, groups, cov_struct=va)
        rslt2 = mod.fit()

        check_wrapper(rslt2)


def test_regularized_poisson():

    np.random.seed(8735)

    ng, gs, p = 1000, 5, 5

    x = np.random.normal(size=(ng*gs, p))
    r = 0.5
    x[:, 2] = r*x[:, 1] + np.sqrt(1-r**2)*x[:, 2]
    lpr = 0.7*(x[:, 1] - x[:, 3])
    mean = np.exp(lpr)
    y = np.random.poisson(mean)

    groups = np.kron(np.arange(ng), np.ones(gs))

    model = gee.GEE(y, x, groups=groups, family=families.Poisson())
    result = model.fit_regularized(0.0000001)

    assert_allclose(result.params, 0.7 * np.r_[0, 1, 0, -1, 0],
                    rtol=0.01, atol=0.12)


def test_regularized_gaussian():

    # Example 1 from Wang et al.

    np.random.seed(8735)

    ng, gs, p = 200, 4, 200

    groups = np.kron(np.arange(ng), np.ones(gs))

    x = np.zeros((ng*gs, p))
    x[:, 0] = 1 * (np.random.uniform(size=ng*gs) < 0.5)
    x[:, 1] = np.random.normal(size=ng*gs)
    r = 0.5
    for j in range(2, p):
        eps = np.random.normal(size=ng*gs)
        x[:, j] = r * x[:, j-1] + np.sqrt(1 - r**2) * eps
    lpr = np.dot(x[:, 0:4], np.r_[2, 3, 1.5, 2])
    s = 0.4
    e = np.sqrt(s) * np.kron(np.random.normal(size=ng), np.ones(gs))
    e += np.sqrt(1 - s) * np.random.normal(size=ng*gs)

    y = lpr + e

    model = gee.GEE(y, x, cov_struct=cov_struct.Exchangeable(), groups=groups)
    result = model.fit_regularized(0.01, maxiter=100)

    ex = np.zeros(200)
    ex[0:4] = np.r_[2, 3, 1.5, 2]
    assert_allclose(result.params, ex, rtol=0.01, atol=0.2)

    assert_allclose(model.cov_struct.dep_params, np.r_[s],
                    rtol=0.01, atol=0.05)


@pytest.mark.smoke
@pytest.mark.matplotlib
def test_plots(close_figures):

    np.random.seed(378)
    exog = np.random.normal(size=100)
    endog = np.random.normal(size=(100, 2))
    groups = np.kron(np.arange(50), np.r_[1, 1])

    model = gee.GEE(exog, endog, groups)
    result = model.fit()
    fig = result.plot_added_variable(1)
    assert_equal(isinstance(fig, plt.Figure), True)
    fig = result.plot_partial_residuals(1)
    assert_equal(isinstance(fig, plt.Figure), True)
    fig = result.plot_ceres_residuals(1)
    assert_equal(isinstance(fig, plt.Figure), True)
    fig = result.plot_isotropic_dependence()
    assert_equal(isinstance(fig, plt.Figure), True)


def test_missing():
    # gh-1877
    data = [['id', 'al', 'status', 'fake', 'grps'],
            ['4A', 'A', 1, 1, 0],
            ['5A', 'A', 1, 2.0, 1],
            ['6A', 'A', 1, 3, 2],
            ['7A', 'A', 1, 2.0, 3],
            ['8A', 'A', 1, 1, 4],
            ['9A', 'A', 1, 2.0, 5],
            ['11A', 'A', 1, 1, 6],
            ['12A', 'A', 1, 2.0, 7],
            ['13A', 'A', 1, 1, 8],
            ['14A', 'A', 1, 1, 9],
            ['15A', 'A', 1, 1, 10],
            ['16A', 'A', 1, 2.0, 11],
            ['17A', 'A', 1, 3.0, 12],
            ['18A', 'A', 1, 3.0, 13],
            ['19A', 'A', 1, 2.0, 14],
            ['20A', 'A', 1, 2.0, 15],
            ['2C', 'C', 0, 3.0, 0],
            ['3C', 'C', 0, 1, 1],
            ['4C', 'C', 0, 1, 2],
            ['5C', 'C', 0, 2.0, 3],
            ['6C', 'C', 0, 1, 4],
            ['9C', 'C', 0, 1, 5],
            ['10C', 'C', 0, 3, 6],
            ['12C', 'C', 0, 3, 7],
            ['14C', 'C', 0, 2.5, 8],
            ['15C', 'C', 0, 1, 9],
            ['17C', 'C', 0, 1, 10],
            ['22C', 'C', 0, 1, 11],
            ['23C', 'C', 0, 1, 12],
            ['24C', 'C', 0, 1, 13],
            ['32C', 'C', 0, 2.0, 14],
            ['35C', 'C', 0, 1, 15]]

    df = pd.DataFrame(data[1:], columns=data[0])
    df.loc[df.fake == 1, 'fake'] = np.nan
    mod = gee.GEE.from_formula('status ~ fake', data=df, groups='grps',
                               cov_struct=cov_struct.Independence(),
                               family=families.Binomial())

    df = df.dropna().copy()
    df['constant'] = 1

    mod2 = gee.GEE(df.status, df[['constant', 'fake']], groups=df.grps,
                   cov_struct=cov_struct.Independence(),
                   family=families.Binomial())

    assert_equal(mod.endog, mod2.endog)
    assert_equal(mod.exog, mod2.exog)
    assert_equal(mod.groups, mod2.groups)

    res = mod.fit()
    res2 = mod2.fit()

    assert_almost_equal(res.params.values, res2.params.values)


def simple_qic_data(fam):

    y = np.r_[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    x1 = np.r_[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    x2 = np.r_[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    x1 = x1[:, None]
    x2 = x2[:, None]

    return y, x1, x2, g


# Test quasi-likelihood by numerical integration in two settings
# where there is a closed form expression.
@pytest.mark.parametrize("family", [families.Gaussian, families.Poisson])
def test_ql_known(family):

    fam = family()

    y, x1, x2, g = simple_qic_data(family)

    model1 = gee.GEE(y, x1, family=fam, groups=g)
    result1 = model1.fit(ddof_scale=0)
    mean1 = result1.fittedvalues

    model2 = gee.GEE(y, x2, family=fam, groups=g)
    result2 = model2.fit(ddof_scale=0)
    mean2 = result2.fittedvalues

    if family is families.Gaussian:
        ql1 = -len(y) / 2.
        ql2 = -len(y) / 2.
    elif family is families.Poisson:
        c = np.zeros_like(y)
        ii = y > 0
        c[ii] = y[ii] * np.log(y[ii]) - y[ii]
        ql1 = np.sum(y * np.log(mean1) - mean1 - c)
        ql2 = np.sum(y * np.log(mean2) - mean2 - c)
    else:
        raise ValueError("Unknown family")

    qle1 = model1.qic(result1.params, result1.scale, result1.cov_params())
    qle2 = model2.qic(result2.params, result2.scale, result2.cov_params())

    assert_allclose(ql1, qle1[0], rtol=1e-4)
    assert_allclose(ql2, qle2[0], rtol=1e-4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qler1 = result1.qic()
        qler2 = result2.qic()
    assert_allclose(qler1, qle1[1:], rtol=1e-5)
    assert_allclose(qler2, qle2[1:], rtol=1e-5)


# Compare differences of QL values computed by numerical integration.
#  Use difference here so that constants that are inconvenient to compute
#  cancel out.
@pytest.mark.parametrize("family", [families.Gaussian,
                                    families.Binomial,
                                    families.Poisson])
def test_ql_diff(family):

    fam = family()

    y, x1, x2, g = simple_qic_data(family)

    model1 = gee.GEE(y, x1, family=fam, groups=g)
    result1 = model1.fit(ddof_scale=0)
    mean1 = result1.fittedvalues

    model2 = gee.GEE(y, x2, family=fam, groups=g)
    result2 = model2.fit(ddof_scale=0)
    mean2 = result2.fittedvalues

    if family is families.Gaussian:
        qldiff = 0
    elif family is families.Binomial:
        qldiff = np.sum(y * np.log(mean1 / (1 - mean1)) + np.log(1 - mean1))
        qldiff -= np.sum(y * np.log(mean2 / (1 - mean2)) + np.log(1 - mean2))
    elif family is families.Poisson:
        qldiff = (np.sum(y * np.log(mean1) - mean1)
                  - np.sum(y * np.log(mean2) - mean2))
    else:
        raise ValueError("unknown family")

    qle1, _, _ = model1.qic(result1.params, result1.scale,
                            result1.cov_params())
    qle2, _, _ = model2.qic(result2.params, result2.scale,
                            result2.cov_params())

    assert_allclose(qle1 - qle2, qldiff, rtol=1e-5, atol=1e-5)


def test_qic_warnings():
    with pytest.warns(UserWarning):
        fam = families.Gaussian()
        y, x1, _, g = simple_qic_data(fam)
        model = gee.GEE(y, x1, family=fam, groups=g)
        result = model.fit()
        result.qic()


@pytest.mark.parametrize("reg", [False, True])
def test_quasipoisson(reg):

    np.random.seed(343)

    n = 1000
    x = np.random.normal(size=(n, 3))
    g = np.random.gamma(1, 1, size=n)
    y = np.random.poisson(g)
    grp = np.kron(np.arange(100), np.ones(n // 100))

    model1 = gee.GEE(y, x, family=families.Poisson(), groups=grp,
                     )
    model2 = gee.GEE(y, x, family=families.Poisson(), groups=grp,
                     )

    if reg:
        result1 = model1.fit_regularized(pen_wt=0.1)
        result2 = model2.fit_regularized(pen_wt=0.1, scale="X2")
    else:
        result1 = model1.fit(cov_type="naive")
        result2 = model2.fit(scale="X2", cov_type="naive")

    # The parameter estimates are the same regardless of how
    # the scale parameter is handled
    assert_allclose(result1.params, result2.params)

    if not reg:
        # The robust covariance does not depend on the scale parameter,
        # but the naive covariance does.
        assert_allclose(result2.cov_naive / result1.cov_naive,
                        result2.scale * np.ones_like(result2.cov_naive))


def test_grid_ar():

    np.random.seed(243)

    r = 0.5
    m = 10
    ng = 100
    ii = np.arange(m)
    cov = r**np.abs(np.subtract.outer(ii, ii))
    covr = np.linalg.cholesky(cov)

    e = [np.dot(covr, np.random.normal(size=m)) for k in range(ng)]
    e = 2 * np.concatenate(e)

    grps = [[k]*m for k in range(ng)]
    grps = np.concatenate(grps)

    x = np.random.normal(size=(ng*m, 3))
    y = np.dot(x, np.r_[1, -1, 0]) + e

    model1 = gee.GEE(y, x, groups=grps,
                     cov_struct=cov_struct.Autoregressive(grid=False))
    result1 = model1.fit()

    model2 = gee.GEE(y, x, groups=grps,
                     cov_struct=cov_struct.Autoregressive(grid=True))
    result2 = model2.fit()

    model3 = gee.GEE(y, x, groups=grps,
                     cov_struct=cov_struct.Stationary(max_lag=1, grid=False))
    result3 = model3.fit()

    assert_allclose(result1.cov_struct.dep_params,
                    result2.cov_struct.dep_params,
                    rtol=0.05)
    assert_allclose(result1.cov_struct.dep_params,
                    result3.cov_struct.dep_params[1], rtol=0.05)


def test_unstructured_complete():

    np.random.seed(43)
    ngrp = 400
    cov = np.asarray([[1, 0.7, 0.2], [0.7, 1, 0.5], [0.2, 0.5, 1]])
    covr = np.linalg.cholesky(cov)
    e = np.random.normal(size=(ngrp, 3))
    e = np.dot(e, covr.T)
    xmat = np.random.normal(size=(3*ngrp, 3))
    par = np.r_[1, -2, 0.1]
    ey = np.dot(xmat, par)
    y = ey + e.ravel()
    g = np.kron(np.arange(ngrp), np.ones(3))
    t = np.kron(np.ones(ngrp), np.r_[0, 1, 2]).astype(int)

    m = gee.GEE(y, xmat, time=t, cov_struct=cov_struct.Unstructured(),
                groups=g)
    r = m.fit()

    assert_allclose(r.params, par, 0.05, 0.5)
    assert_allclose(m.cov_struct.dep_params, cov, 0.05, 0.5)


def test_unstructured_incomplete():

    np.random.seed(43)
    ngrp = 400
    cov = np.asarray([[1, 0.7, 0.2], [0.7, 1, 0.5], [0.2, 0.5, 1]])
    covr = np.linalg.cholesky(cov)
    e = np.random.normal(size=(ngrp, 3))
    e = np.dot(e, covr.T)
    xmat = np.random.normal(size=(3*ngrp, 3))
    par = np.r_[1, -2, 0.1]
    ey = np.dot(xmat, par)

    yl, xl, tl, gl = [], [], [], []
    for i in range(ngrp):

        # Omit one observation from each group of 3
        ix = [0, 1, 2]
        ix.pop(i % 3)
        ix = np.asarray(ix)
        tl.append(ix)

        yl.append(ey[3*i + ix] + e[i, ix])
        x = xmat[3*i + ix, :]
        xl.append(x)
        gl.append(i * np.ones(2))

    y = np.concatenate(yl)
    x = np.concatenate(xl, axis=0)
    t = np.concatenate(tl)
    t = np.asarray(t, dtype=int)
    g = np.concatenate(gl)

    m = gee.GEE(y, x, time=t[:, None], cov_struct=cov_struct.Unstructured(),
                groups=g)
    r = m.fit()

    assert_allclose(r.params, par, 0.05, 0.5)
    assert_allclose(m.cov_struct.dep_params, cov, 0.05, 0.5)


def test_ar_covsolve():

    np.random.seed(123)

    c = cov_struct.Autoregressive(grid=True)
    c.dep_params = 0.4

    for d in 1, 2, 4:
        for q in 1, 4:

            ii = np.arange(d)
            mat = 0.4 ** np.abs(np.subtract.outer(ii, ii))
            sd = np.random.uniform(size=d)

            if q == 1:
                z = np.random.normal(size=d)
            else:
                z = np.random.normal(size=(d, q))

            sm = np.diag(sd)
            z1 = np.linalg.solve(sm,
                                 np.linalg.solve(mat, np.linalg.solve(sm, z)))
            z2 = c.covariance_matrix_solve(np.zeros_like(sd),
                                           np.zeros_like(sd),
                                           sd, [z])

            assert_allclose(z1, z2[0], rtol=1e-5, atol=1e-5)


def test_ex_covsolve():

    np.random.seed(123)

    c = cov_struct.Exchangeable()
    c.dep_params = 0.4

    for d in 1, 2, 4:
        for q in 1, 4:

            mat = 0.4 * np.ones((d, d)) + 0.6 * np.eye(d)
            sd = np.random.uniform(size=d)

            if q == 1:
                z = np.random.normal(size=d)
            else:
                z = np.random.normal(size=(d, q))

            sm = np.diag(sd)
            z1 = np.linalg.solve(sm,
                                 np.linalg.solve(mat, np.linalg.solve(sm, z)))
            z2 = c.covariance_matrix_solve(np.zeros_like(sd),
                                           np.arange(d, dtype=int),
                                           sd, [z])

            assert_allclose(z1, z2[0], rtol=1e-5, atol=1e-5)


def test_stationary_covsolve():

    np.random.seed(123)

    c = cov_struct.Stationary(grid=True)
    c.time = np.arange(10, dtype=int)

    for d in 1, 2, 4:
        for q in 1, 4:

            c.dep_params = (2.0 ** (-np.arange(d)))
            c.max_lag = d - 1
            mat, _ = c.covariance_matrix(np.zeros(d),
                                         np.arange(d, dtype=int))
            sd = np.random.uniform(size=d)

            if q == 1:
                z = np.random.normal(size=d)
            else:
                z = np.random.normal(size=(d, q))

            sm = np.diag(sd)
            z1 = np.linalg.solve(sm,
                                 np.linalg.solve(mat, np.linalg.solve(sm, z)))
            z2 = c.covariance_matrix_solve(np.zeros_like(sd),
                                           np.arange(d, dtype=int),
                                           sd, [z])

            assert_allclose(z1, z2[0], rtol=1e-5, atol=1e-5)
