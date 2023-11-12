from statsmodels.compat.platform import PLATFORM_OSX

import os
import csv
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import pytest

from statsmodels.regression.mixed_linear_model import (
    MixedLM, MixedLMParams, _smw_solver, _smw_logdet)
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_)

from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd

from .results import lme_r_results

# TODO: add tests with unequal group sizes


class R_Results:
    """
    A class for holding various results obtained from fitting one data
    set using lmer in R.

    Parameters
    ----------
    meth : str
        Either "ml" or "reml".
    irfs : str
        Either "irf", for independent random effects, or "drf" for
        dependent random effects.
    ds_ix : int
        The number of the data set
    """

    def __init__(self, meth, irfs, ds_ix):

        bname = "_%s_%s_%d" % (meth, irfs, ds_ix)

        self.coef = getattr(lme_r_results, "coef" + bname)
        self.vcov_r = getattr(lme_r_results, "vcov" + bname)
        self.cov_re_r = getattr(lme_r_results, "cov_re" + bname)
        self.scale_r = getattr(lme_r_results, "scale" + bname)
        self.loglike = getattr(lme_r_results, "loglike" + bname)

        if hasattr(lme_r_results, "ranef_mean" + bname):
            self.ranef_postmean = getattr(lme_r_results, "ranef_mean" + bname)
            self.ranef_condvar = getattr(lme_r_results,
                                         "ranef_condvar" + bname)
            self.ranef_condvar = np.atleast_2d(self.ranef_condvar)

        # Load the data file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, "lme%02d.csv" % ds_ix)
        with open(fname, encoding="utf-8") as fid:
            rdr = csv.reader(fid)
            header = next(rdr)
            data = [[float(x) for x in line] for line in rdr]
        data = np.asarray(data)

        # Split into exog, endog, etc.
        self.endog = data[:, header.index("endog")]
        self.groups = data[:, header.index("groups")]
        ii = [i for i, x in enumerate(header) if x.startswith("exog_fe")]
        self.exog_fe = data[:, ii]
        ii = [i for i, x in enumerate(header) if x.startswith("exog_re")]
        self.exog_re = data[:, ii]


def loglike_function(model, profile_fe, has_fe):
    # Returns a function that evaluates the negative log-likelihood for
    # the given model.

    def f(x):
        params = MixedLMParams.from_packed(
            x, model.k_fe, model.k_re, model.use_sqrt, has_fe=has_fe)
        return -model.loglike(params, profile_fe=profile_fe)

    return f


class TestMixedLM:

    # Test analytic scores and Hessian using numeric differentiation
    @pytest.mark.slow
    @pytest.mark.parametrize("use_sqrt", [False, True])
    @pytest.mark.parametrize("reml", [False, True])
    @pytest.mark.parametrize("profile_fe", [False, True])
    def test_compare_numdiff(self, use_sqrt, reml, profile_fe):

        n_grp = 200
        grpsize = 5
        k_fe = 3
        k_re = 2

        np.random.seed(3558)
        exog_fe = np.random.normal(size=(n_grp * grpsize, k_fe))
        exog_re = np.random.normal(size=(n_grp * grpsize, k_re))
        exog_re[:, 0] = 1
        exog_vc = np.random.normal(size=(n_grp * grpsize, 3))
        slopes = np.random.normal(size=(n_grp, k_re))
        slopes[:, -1] *= 2
        slopes = np.kron(slopes, np.ones((grpsize, 1)))
        slopes_vc = np.random.normal(size=(n_grp, 3))
        slopes_vc = np.kron(slopes_vc, np.ones((grpsize, 1)))
        slopes_vc[:, -1] *= 2
        re_values = (slopes * exog_re).sum(1)
        vc_values = (slopes_vc * exog_vc).sum(1)
        err = np.random.normal(size=n_grp * grpsize)
        endog = exog_fe.sum(1) + re_values + vc_values + err
        groups = np.kron(range(n_grp), np.ones(grpsize))

        vc = {"a": {}, "b": {}}
        for i in range(n_grp):
            ix = np.flatnonzero(groups == i)
            vc["a"][i] = exog_vc[ix, 0:2]
            vc["b"][i] = exog_vc[ix, 2:3]

        with pytest.warns(UserWarning, match="Using deprecated variance"):
            model = MixedLM(
                endog,
                exog_fe,
                groups,
                exog_re,
                exog_vc=vc,
                use_sqrt=use_sqrt)
        rslt = model.fit(reml=reml)

        loglike = loglike_function(
            model, profile_fe=profile_fe, has_fe=not profile_fe)

        try:
            # Test the score at several points.
            for kr in range(5):
                fe_params = np.random.normal(size=k_fe)
                cov_re = np.random.normal(size=(k_re, k_re))
                cov_re = np.dot(cov_re.T, cov_re)
                vcomp = np.random.normal(size=2)**2
                params = MixedLMParams.from_components(
                    fe_params, cov_re=cov_re, vcomp=vcomp)
                params_vec = params.get_packed(
                    has_fe=not profile_fe, use_sqrt=use_sqrt)

                # Check scores
                gr = -model.score(params, profile_fe=profile_fe)
                ngr = nd.approx_fprime(params_vec, loglike)
                assert_allclose(gr, ngr, rtol=1e-3)

            # Check Hessian matrices at the MLE (we do not have
            # the profile Hessian matrix and we do not care
            # about the Hessian for the square root
            # transformed parameter).
            if (profile_fe is False) and (use_sqrt is False):
                hess, sing = model.hessian(rslt.params_object)
                if sing:
                    pytest.fail("hessian should not be singular")
                hess *= -1
                params_vec = rslt.params_object.get_packed(
                    use_sqrt=False, has_fe=True)
                loglike_h = loglike_function(
                    model, profile_fe=False, has_fe=True)
                nhess = nd.approx_hess(params_vec, loglike_h)
                assert_allclose(hess, nhess, rtol=1e-3)
        except AssertionError:
            # See GH#5628; because this test fails unpredictably but only on
            #  OSX, we only xfail it there.
            if PLATFORM_OSX:
                pytest.xfail("fails on OSX due to unresolved "
                             "numerical differences")
            else:
                raise

    def test_default_re(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit()
        mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
        assert_almost_equal(mdf1.params, mdf2.params, decimal=8)

    def test_history(self):

        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mod = MixedLM(endog, exog, groups)
        rslt = mod.fit(full_output=True)
        assert_equal(hasattr(rslt, "hist"), True)

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_profile_inference(self):
        np.random.seed(9814)
        k_fe = 2
        gsize = 3
        n_grp = 100
        exog = np.random.normal(size=(n_grp * gsize, k_fe))
        exog_re = np.ones((n_grp * gsize, 1))
        groups = np.kron(np.arange(n_grp), np.ones(gsize))
        vca = np.random.normal(size=n_grp * gsize)
        vcb = np.random.normal(size=n_grp * gsize)
        errors = 0
        g_errors = np.kron(np.random.normal(size=100), np.ones(gsize))
        errors += g_errors + exog_re[:, 0]
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vca
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vcb
        errors += np.random.normal(size=n_grp * gsize)

        endog = exog.sum(1) + errors
        vc = {"a": {}, "b": {}}
        for k in range(n_grp):
            ii = np.flatnonzero(groups == k)
            vc["a"][k] = vca[ii][:, None]
            vc["b"][k] = vcb[ii][:, None]
        with pytest.warns(UserWarning, match="Using deprecated variance"):
            rslt = MixedLM(endog, exog, groups=groups,
                           exog_re=exog_re, exog_vc=vc).fit()
        rslt.profile_re(0, vtype='re', dist_low=1, num_low=3,
                        dist_high=1, num_high=3)
        rslt.profile_re('b', vtype='vc', dist_low=0.5, num_low=3,
                        dist_high=0.5, num_high=3)

    def test_vcomp_1(self):
        # Fit the same model using constrained random effects and
        # variance components.

        np.random.seed(4279)
        exog = np.random.normal(size=(400, 1))
        exog_re = np.random.normal(size=(400, 2))
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=(100, 2))
        slopes[:, 1] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors = slopes.sum(1) + np.random.normal(size=400)
        endog = exog.sum(1) + errors

        free = MixedLMParams(1, 2, 0)
        free.fe_params = np.ones(1)
        free.cov_re = np.eye(2)
        free.vcomp = np.zeros(0)

        model1 = MixedLM(endog, exog, groups, exog_re=exog_re)
        result1 = model1.fit(free=free)

        exog_vc = {"a": {}, "b": {}}
        for k, group in enumerate(model1.group_labels):
            ix = model1.row_indices[group]
            exog_vc["a"][group] = exog_re[ix, 0:1]
            exog_vc["b"][group] = exog_re[ix, 1:2]
        with pytest.warns(UserWarning, match="Using deprecated variance"):
            model2 = MixedLM(endog, exog, groups, exog_vc=exog_vc)
        result2 = model2.fit()
        result2.summary()

        assert_allclose(result1.fe_params, result2.fe_params, atol=1e-4)
        assert_allclose(
            np.diag(result1.cov_re), result2.vcomp, atol=1e-2, rtol=1e-4)
        assert_allclose(
            result1.bse[[0, 1, 3]], result2.bse, atol=1e-2, rtol=1e-2)

    def test_vcomp_2(self):
        # Simulated data comparison to R

        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))

        # Build up the random error vector
        errors = 0

        # The random effects
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)

        # First variance component
        subgroups1 = np.kron(np.arange(n / 4), np.ones(4))
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))

        # Second variance component
        subgroups2 = np.kron(np.arange(n / 2), np.ones(2))
        errors += np.kron(2 * np.random.normal(size=n // 2), np.ones(2))

        # iid errors
        errors += np.random.normal(size=n)

        endog = exog.sum(1) + errors

        df = pd.DataFrame(index=range(n))
        df["y"] = endog
        df["groups"] = groups
        df["x1"] = exog[:, 0]
        df["x2"] = exog[:, 1]
        df["z1"] = exog_re[:, 0]
        df["z2"] = exog_re[:, 1]
        df["v1"] = subgroups1
        df["v2"] = subgroups2

        # Equivalent model in R:
        # df.to_csv("tst.csv")
        # model = lmer(y ~ x1 + x2 + (0 + z1 + z2 | groups) + (1 | v1) + (1 |
        # v2), df)

        vcf = {"a": "0 + C(v1)", "b": "0 + C(v2)"}
        model1 = MixedLM.from_formula(
            "y ~ x1 + x2",
            groups=groups,
            re_formula="0+z1+z2",
            vc_formula=vcf,
            data=df)
        result1 = model1.fit()

        # Compare to R
        assert_allclose(
            result1.fe_params, [0.16527, 0.99911, 0.96217], rtol=1e-4)
        assert_allclose(
            result1.cov_re, [[1.244, 0.146], [0.146, 1.371]], rtol=1e-3)
        assert_allclose(result1.vcomp, [4.024, 3.997], rtol=1e-3)
        assert_allclose(
            result1.bse.iloc[0:3], [0.12610, 0.03938, 0.03848], rtol=1e-3)

    def test_vcomp_3(self):
        # Test a model with vcomp but no other random effects, using formulas.

        np.random.seed(4279)
        x1 = np.random.normal(size=400)
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=100)
        slopes = np.kron(slopes, np.ones(4)) * x1
        y = slopes + np.random.normal(size=400)
        vc_fml = {"a": "0 + x1"}
        df = pd.DataFrame({"y": y, "x1": x1, "groups": groups})

        model = MixedLM.from_formula(
            "y ~ 1", groups="groups", vc_formula=vc_fml, data=df)
        result = model.fit()
        result.summary()

        assert_allclose(
            result.resid.iloc[0:4],
            np.r_[-1.180753, 0.279966, 0.578576, -0.667916],
            rtol=1e-3)
        assert_allclose(
            result.fittedvalues.iloc[0:4],
            np.r_[-0.101549, 0.028613, -0.224621, -0.126295],
            rtol=1e-3)

    def test_sparse(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')

        # Dense
        data = pd.read_csv(fname)
        vcf = {"cask": "0 + cask"}
        model = MixedLM.from_formula(
            "strength ~ 1",
            groups="batch",
            re_formula="1",
            vc_formula=vcf,
            data=data)
        result = model.fit()

        # Sparse
        model2 = MixedLM.from_formula(
            "strength ~ 1",
            groups="batch",
            re_formula="1",
            vc_formula=vcf,
            use_sparse=True,
            data=data)
        result2 = model2.fit()

        assert_allclose(result.params, result2.params)
        assert_allclose(result.bse, result2.bse)

    def test_dietox(self):
        # dietox data from geepack using random intercepts
        #
        # Fit in R using
        #
        # library(geepack)
        # rm = lmer(Weight ~ Time + (1 | Pig), data=dietox)
        # rm = lmer(Weight ~ Time + (1 | Pig), REML=FALSE, data=dietox)
        #
        # Comments below are R code used to extract the numbers used
        # for comparison.

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'dietox.csv')

        # REML
        data = pd.read_csv(fname)
        model = MixedLM.from_formula("Weight ~ Time", groups="Pig", data=data)
        result = model.fit()

        # fixef(rm)
        assert_allclose(
            result.fe_params, np.r_[15.723523, 6.942505], rtol=1e-5)

        # sqrt(diag(vcov(rm)))
        assert_allclose(
            result.bse[0:2], np.r_[0.78805374, 0.03338727], rtol=1e-5)

        # attr(VarCorr(rm), "sc")^2
        assert_allclose(result.scale, 11.36692, rtol=1e-5)

        # VarCorr(rm)[[1]][[1]]
        assert_allclose(result.cov_re, 40.39395, rtol=1e-5)

        # logLik(rm)
        assert_allclose(
            model.loglike(result.params_object), -2404.775, rtol=1e-5)

        # ML
        data = pd.read_csv(fname)
        model = MixedLM.from_formula("Weight ~ Time", groups="Pig", data=data)
        result = model.fit(reml=False)

        # fixef(rm)
        assert_allclose(
            result.fe_params, np.r_[15.723517, 6.942506], rtol=1e-5)

        # sqrt(diag(vcov(rm)))
        assert_allclose(
            result.bse[0:2], np.r_[0.7829397, 0.0333661], rtol=1e-5)

        # attr(VarCorr(rm), "sc")^2
        assert_allclose(result.scale, 11.35251, rtol=1e-5)

        # VarCorr(rm)[[1]][[1]]
        assert_allclose(result.cov_re, 39.82097, rtol=1e-5)

        # logLik(rm)
        assert_allclose(
            model.loglike(result.params_object), -2402.932, rtol=1e-5)

    def test_dietox_slopes(self):
        # dietox data from geepack using random intercepts
        #
        # Fit in R using
        #
        # library(geepack)
        # r = lmer(Weight ~ Time + (1 + Time | Pig), data=dietox)
        # r = lmer(Weight ~ Time + (1 + Time | Pig), REML=FALSE, data=dietox)
        #
        # Comments below are the R code used to extract the constants
        # for comparison.

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'dietox.csv')

        # REML
        data = pd.read_csv(fname)
        model = MixedLM.from_formula(
            "Weight ~ Time", groups="Pig", re_formula="1 + Time", data=data)
        result = model.fit(method="cg")

        # fixef(r)
        assert_allclose(
            result.fe_params, np.r_[15.738650, 6.939014], rtol=1e-5)

        # sqrt(diag(vcov(r)))
        assert_allclose(
            result.bse[0:2], np.r_[0.5501253, 0.0798254], rtol=1e-3)

        # attr(VarCorr(r), "sc")^2
        assert_allclose(result.scale, 6.03745, rtol=1e-3)

        # as.numeric(VarCorr(r)[[1]])
        assert_allclose(
            result.cov_re.values.ravel(),
            np.r_[19.4934552, 0.2938323, 0.2938323, 0.4160620],
            rtol=1e-1)

        # logLik(r)
        assert_allclose(
            model.loglike(result.params_object), -2217.047, rtol=1e-5)

        # ML
        data = pd.read_csv(fname)
        model = MixedLM.from_formula(
            "Weight ~ Time", groups="Pig", re_formula="1 + Time", data=data)
        result = model.fit(method='cg', reml=False)

        # fixef(r)
        assert_allclose(result.fe_params, np.r_[15.73863, 6.93902], rtol=1e-5)

        # sqrt(diag(vcov(r)))
        assert_allclose(
            result.bse[0:2], np.r_[0.54629282, 0.07926954], rtol=1e-3)

        # attr(VarCorr(r), "sc")^2
        assert_allclose(result.scale, 6.037441, rtol=1e-3)

        #  as.numeric(VarCorr(r)[[1]])
        assert_allclose(
            result.cov_re.values.ravel(),
            np.r_[19.190922, 0.293568, 0.293568, 0.409695],
            rtol=1e-2)

        # logLik(r)
        assert_allclose(
            model.loglike(result.params_object), -2215.753, rtol=1e-5)

    def test_pastes_vcomp(self):
        # pastes data from lme4
        #
        # Fit in R using:
        #
        # r = lmer(strength ~ (1|batch) + (1|batch:cask), data=data)
        # r = lmer(strength ~ (1|batch) + (1|batch:cask), data=data,
        #          reml=FALSE)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')
        data = pd.read_csv(fname)
        vcf = {"cask": "0 + cask"}

        # REML
        model = MixedLM.from_formula(
            "strength ~ 1",
            groups="batch",
            re_formula="1",
            vc_formula=vcf,
            data=data)
        result = model.fit()

        # fixef(r)
        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=1e-3)

        # sqrt(diag(vcov(r)))
        assert_allclose(result.bse.iloc[0], 0.6769, rtol=1e-3)

        # VarCorr(r)$batch[[1]]
        assert_allclose(result.cov_re.iloc[0, 0], 1.657, rtol=1e-3)

        # attr(VarCorr(r), "sc")^2
        assert_allclose(result.scale, 0.678, rtol=1e-3)

        # logLik(r)
        assert_allclose(result.llf, -123.49, rtol=1e-1)

        # do not provide aic/bic with REML
        assert_equal(result.aic, np.nan)
        assert_equal(result.bic, np.nan)

        # resid(r)[1:5]
        resid = np.r_[0.17133538, -0.02866462, -1.08662875, 1.11337125,
                      -0.12093607]
        assert_allclose(result.resid[0:5], resid, rtol=1e-3)

        # predict(r)[1:5]
        fit = np.r_[62.62866, 62.62866, 61.18663, 61.18663, 62.82094]
        assert_allclose(result.fittedvalues[0:5], fit, rtol=1e-4)

        # ML
        model = MixedLM.from_formula(
            "strength ~ 1",
            groups="batch",
            re_formula="1",
            vc_formula=vcf,
            data=data)
        result = model.fit(reml=False)

        # fixef(r)
        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=1e-3)

        # sqrt(diag(vcov(r)))
        assert_allclose(result.bse.iloc[0], 0.642, rtol=1e-3)

        # VarCorr(r)$batch[[1]]
        assert_allclose(result.cov_re.iloc[0, 0], 1.199, rtol=1e-3)

        # attr(VarCorr(r), "sc")^2
        assert_allclose(result.scale, 0.67799, rtol=1e-3)

        # logLik(r)
        assert_allclose(result.llf, -123.997, rtol=1e-1)

        # AIC(r)
        assert_allclose(result.aic, 255.9944, rtol=1e-3)

        # BIC(r)
        assert_allclose(result.bic, 264.3718, rtol=1e-3)

    @pytest.mark.slow
    def test_vcomp_formula(self):

        np.random.seed(6241)
        n = 800
        exog = np.random.normal(size=(n, 2))
        exog[:, 0] = 1
        ex_vc = []
        groups = np.kron(np.arange(n / 4), np.ones(4))
        errors = 0
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 4, 2))
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors += slopes.sum(1)
        ex_vc = np.random.normal(size=(n, 4))
        slopes = np.random.normal(size=(n // 4, 4))
        slopes[:, 2:] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * ex_vc
        errors += slopes.sum(1)
        errors += np.random.normal(size=n)
        endog = exog.sum(1) + errors

        exog_vc = {"a": {}, "b": {}}
        for k, group in enumerate(range(int(n / 4))):
            ix = np.flatnonzero(groups == group)
            exog_vc["a"][group] = ex_vc[ix, 0:2]
            exog_vc["b"][group] = ex_vc[ix, 2:]
        with pytest.warns(UserWarning, match="Using deprecated variance"):
            model1 = MixedLM(endog, exog, groups, exog_re=exog_re,
                             exog_vc=exog_vc)
        result1 = model1.fit()

        df = pd.DataFrame(exog[:, 1:], columns=["x1"])
        df["y"] = endog
        df["re1"] = exog_re[:, 0]
        df["re2"] = exog_re[:, 1]
        df["vc1"] = ex_vc[:, 0]
        df["vc2"] = ex_vc[:, 1]
        df["vc3"] = ex_vc[:, 2]
        df["vc4"] = ex_vc[:, 3]
        vc_formula = {"a": "0 + vc1 + vc2", "b": "0 + vc3 + vc4"}
        model2 = MixedLM.from_formula(
            "y ~ x1",
            groups=groups,
            re_formula="0 + re1 + re2",
            vc_formula=vc_formula,
            data=df)
        result2 = model2.fit()

        assert_allclose(result1.fe_params, result2.fe_params, rtol=1e-8)
        assert_allclose(result1.cov_re, result2.cov_re, rtol=1e-8)
        assert_allclose(result1.vcomp, result2.vcomp, rtol=1e-8)
        assert_allclose(result1.params, result2.params, rtol=1e-8)
        assert_allclose(result1.bse, result2.bse, rtol=1e-8)

    def test_formulas(self):
        np.random.seed(2410)
        exog = np.random.normal(size=(300, 4))
        exog_re = np.random.normal(size=300)
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = exog_re * np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)

        mod1 = MixedLM(endog, exog, groups, exog_re)
        # test the names
        assert_(mod1.data.xnames == ["x1", "x2", "x3", "x4"])
        assert_(mod1.data.exog_re_names == ["x_re1"])
        assert_(mod1.data.exog_re_names_full == ["x_re1 Var"])
        rslt1 = mod1.fit()

        # Fit with a formula, passing groups as the actual values.
        df = pd.DataFrame({"endog": endog})
        for k in range(exog.shape[1]):
            df["exog%d" % k] = exog[:, k]
        df["exog_re"] = exog_re
        fml = "endog ~ 0 + exog0 + exog1 + exog2 + exog3"
        re_fml = "0 + exog_re"
        mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml, groups=groups)

        assert_(mod2.data.xnames == ["exog0", "exog1", "exog2", "exog3"])
        assert_(mod2.data.exog_re_names == ["exog_re"])
        assert_(mod2.data.exog_re_names_full == ["exog_re Var"])

        rslt2 = mod2.fit()
        assert_almost_equal(rslt1.params, rslt2.params)

        # Fit with a formula, passing groups as the variable name.
        df["groups"] = groups
        mod3 = MixedLM.from_formula(
            fml, df, re_formula=re_fml, groups="groups")
        assert_(mod3.data.xnames == ["exog0", "exog1", "exog2", "exog3"])
        assert_(mod3.data.exog_re_names == ["exog_re"])
        assert_(mod3.data.exog_re_names_full == ["exog_re Var"])

        rslt3 = mod3.fit(start_params=rslt2.params)
        assert_allclose(rslt1.params, rslt3.params, rtol=1e-4)

        # Check default variance structure with non-formula model
        # creation, also use different exog_re that produces a zero
        # estimated variance parameter.
        exog_re = np.ones(len(endog), dtype=np.float64)
        mod4 = MixedLM(endog, exog, groups, exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rslt4 = mod4.fit()
        from statsmodels.formula.api import mixedlm
        mod5 = mixedlm(fml, df, groups="groups")
        assert_(mod5.data.exog_re_names == ["groups"])
        assert_(mod5.data.exog_re_names_full == ["groups Var"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rslt5 = mod5.fit()
        assert_almost_equal(rslt4.params, rslt5.params)

    @pytest.mark.slow
    def test_regularized(self):

        np.random.seed(3453)
        exog = np.random.normal(size=(400, 5))
        groups = np.kron(np.arange(100), np.ones(4))
        expected_endog = exog[:, 0] - exog[:, 2]
        endog = expected_endog +\
            np.kron(np.random.normal(size=100), np.ones(4)) +\
            np.random.normal(size=400)

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf1 = md.fit_regularized(alpha=1.)
        mdf1.summary()

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf2 = md.fit_regularized(alpha=10 * np.ones(5))
        mdf2.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf3 = md.fit_regularized(method=pen, alpha=0.)
        mdf3.summary()

        # L2 regularization
        pen = penalties.L2()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdf4 = md.fit_regularized(method=pen, alpha=10.)
        mdf4.summary()

        # Pseudo-Huber regularization
        pen = penalties.PseudoHuber(0.3)
        mdf5 = md.fit_regularized(method=pen, alpha=1.)
        mdf5.summary()


# ------------------------------------------------------------------


class TestMixedLMSummary:
    # Test various aspects of the MixedLM summary
    @classmethod
    def setup_class(cls):
        # Setup the model and estimate it.
        pid = np.repeat([0, 1], 5)
        x0 = np.repeat([1], 10)
        x1 = [1, 5, 7, 3, 5, 1, 2, 6, 9, 8]
        x2 = [6, 2, 1, 0, 1, 4, 3, 8, 2, 1]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        df = pd.DataFrame({"y": y, "pid": pid, "x0": x0, "x1": x1, "x2": x2})
        endog = df["y"].values
        exog = df[["x0", "x1", "x2"]].values
        groups = df["pid"].values
        cls.res = MixedLM(endog, exog, groups=groups).fit()

    def test_summary(self):
        # Test that the summary correctly includes all variables.
        summ = self.res.summary()
        desired = ["const", "x1", "x2", "Group Var"]
        # Second table is summary of params
        actual = summ.tables[1].index.values
        assert_equal(actual, desired)

    def test_summary_xname_fe(self):
        # Test that the `xname_fe` argument is reflected in the summary table.
        summ = self.res.summary(xname_fe=["Constant", "Age", "Weight"])
        desired = ["Constant", "Age", "Weight", "Group Var"]
        actual = summ.tables[
            1].index.values  # Second table is summary of params
        assert_equal(actual, desired)

    def test_summary_xname_re(self):
        # Test that the `xname_re` argument is reflected in the summary table.
        summ = self.res.summary(xname_re=["Random Effects"])
        desired = ["const", "x1", "x2", "Random Effects"]
        actual = summ.tables[
            1].index.values  # Second table is summary of params
        assert_equal(actual, desired)


# ------------------------------------------------------------------


class TestMixedLMSummaryRegularized(TestMixedLMSummary):
    # Test various aspects of the MixedLM summary
    # after fitting model with fit_regularized function
    @classmethod
    def setup_class(cls):
        # Setup the model and estimate it.
        pid = np.repeat([0, 1], 5)
        x0 = np.repeat([1], 10)
        x1 = [1, 5, 7, 3, 5, 1, 2, 6, 9, 8]
        x2 = [6, 2, 1, 0, 1, 4, 3, 8, 2, 1]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        df = pd.DataFrame({"y": y, "pid": pid, "x0": x0, "x1": x1, "x2": x2})
        endog = df["y"].values
        exog = df[["x0", "x1", "x2"]].values
        groups = df["pid"].values
        cls.res = MixedLM(endog, exog, groups=groups).fit_regularized()


# ------------------------------------------------------------------


# TODO: better name
def do1(reml, irf, ds_ix):

    # No need to check independent random effects when there is
    # only one of them.
    if irf and ds_ix < 6:
        return

    irfs = "irf" if irf else "drf"
    meth = "reml" if reml else "ml"

    rslt = R_Results(meth, irfs, ds_ix)

    # Fit the model
    md = MixedLM(rslt.endog, rslt.exog_fe, rslt.groups, rslt.exog_re)
    if not irf:  # Free random effects covariance
        if np.any(np.diag(rslt.cov_re_r) < 1e-5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdf = md.fit(gtol=1e-7, reml=reml)
        else:
            mdf = md.fit(gtol=1e-7, reml=reml)

    else:  # Independent random effects
        k_fe = rslt.exog_fe.shape[1]
        k_re = rslt.exog_re.shape[1]
        free = MixedLMParams(k_fe, k_re, 0)
        free.fe_params = np.ones(k_fe)
        free.cov_re = np.eye(k_re)
        free.vcomp = np.array([])
        if np.any(np.diag(rslt.cov_re_r) < 1e-5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdf = md.fit(reml=reml, gtol=1e-7, free=free)
        else:
            mdf = md.fit(reml=reml, gtol=1e-7, free=free)

    assert_almost_equal(mdf.fe_params, rslt.coef, decimal=4)
    assert_almost_equal(mdf.cov_re, rslt.cov_re_r, decimal=4)
    assert_almost_equal(mdf.scale, rslt.scale_r, decimal=4)

    k_fe = md.k_fe
    assert_almost_equal(
        rslt.vcov_r, mdf.cov_params()[0:k_fe, 0:k_fe], decimal=3)

    assert_almost_equal(mdf.llf, rslt.loglike[0], decimal=2)

    # Not supported in R except for independent random effects
    if not irf:
        assert_almost_equal(
            mdf.random_effects[0], rslt.ranef_postmean, decimal=3)
        assert_almost_equal(
            mdf.random_effects_cov[0], rslt.ranef_condvar, decimal=3)


# ------------------------------------------------------------------

# Run all the tests against R
cur_dir = os.path.dirname(os.path.abspath(__file__))
rdir = os.path.join(cur_dir, 'results')
fnames = os.listdir(rdir)
fnames = [x for x in fnames if x.startswith("lme") and x.endswith(".csv")]


# Copied from #3847
@pytest.mark.parametrize('fname', fnames)
@pytest.mark.parametrize('reml', [False, True])
@pytest.mark.parametrize('irf', [False, True])
def test_r(fname, reml, irf):
    ds_ix = int(fname[3:5])
    do1(reml, irf, ds_ix)


# ------------------------------------------------------------------


def test_mixed_lm_wrapper():
    # a bit more complicated model to test
    np.random.seed(2410)
    exog = np.random.normal(size=(300, 4))
    exog_re = np.random.normal(size=300)
    groups = np.kron(np.arange(100), [1, 1, 1])
    g_errors = exog_re * np.kron(np.random.normal(size=100), [1, 1, 1])
    endog = exog.sum(1) + g_errors + np.random.normal(size=300)

    # Fit with a formula, passing groups as the actual values.
    df = pd.DataFrame({"endog": endog})
    for k in range(exog.shape[1]):
        df["exog%d" % k] = exog[:, k]
    df["exog_re"] = exog_re
    fml = "endog ~ 0 + exog0 + exog1 + exog2 + exog3"
    re_fml = "~ exog_re"
    mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml, groups=groups)
    result = mod2.fit()
    result.summary()

    xnames = ["exog0", "exog1", "exog2", "exog3"]
    re_names = ["Group", "exog_re"]
    re_names_full = ["Group Var", "Group x exog_re Cov", "exog_re Var"]

    assert_(mod2.data.xnames == xnames)
    assert_(mod2.data.exog_re_names == re_names)
    assert_(mod2.data.exog_re_names_full == re_names_full)

    params = result.params
    assert_(params.index.tolist() == xnames + re_names_full)
    bse = result.bse
    assert_(bse.index.tolist() == xnames + re_names_full)
    tvalues = result.tvalues
    assert_(tvalues.index.tolist() == xnames + re_names_full)
    cov_params = result.cov_params()
    assert_(cov_params.index.tolist() == xnames + re_names_full)
    assert_(cov_params.columns.tolist() == xnames + re_names_full)
    fe = result.fe_params
    assert_(fe.index.tolist() == xnames)
    bse_fe = result.bse_fe
    assert_(bse_fe.index.tolist() == xnames)
    cov_re = result.cov_re
    assert_(cov_re.index.tolist() == re_names)
    assert_(cov_re.columns.tolist() == re_names)
    cov_re_u = result.cov_re_unscaled
    assert_(cov_re_u.index.tolist() == re_names)
    assert_(cov_re_u.columns.tolist() == re_names)
    bse_re = result.bse_re
    assert_(bse_re.index.tolist() == re_names_full)


def test_random_effects():

    np.random.seed(23429)

    # Default model (random effects only)
    ngrp = 100
    gsize = 10
    rsd = 2
    gsd = 3
    mn = gsd * np.random.normal(size=ngrp)
    gmn = np.kron(mn, np.ones(gsize))
    y = gmn + rsd * np.random.normal(size=ngrp * gsize)
    gr = np.kron(np.arange(ngrp), np.ones(gsize))
    x = np.ones(ngrp * gsize)
    model = MixedLM(y, x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)

    # Random intercept only, set explicitly
    model = MixedLM(y, x, exog_re=x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)

    # Random intercept and slope
    xr = np.random.normal(size=(ngrp * gsize, 2))
    xr[:, 0] = 1
    qp = np.linspace(-1, 1, gsize)
    xr[:, 1] = np.kron(np.ones(ngrp), qp)
    model = MixedLM(y, x, exog_re=xr, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 2)


@pytest.mark.slow
def test_handle_missing():

    np.random.seed(23423)
    df = np.random.normal(size=(100, 6))
    df = pd.DataFrame(df)
    df.columns = ["y", "g", "x1", "z1", "c1", "c2"]
    df["g"] = np.kron(np.arange(50), np.ones(2))
    re = np.random.normal(size=(50, 4))
    re = np.kron(re, np.ones((2, 1)))
    df["y"] = re[:, 0] + re[:, 1] * df.z1 + re[:, 2] * df.c1
    df["y"] += re[:, 3] * df.c2 + np.random.normal(size=100)
    df.loc[1, "y"] = np.NaN
    df.loc[2, "g"] = np.NaN
    df.loc[3, "x1"] = np.NaN
    df.loc[4, "z1"] = np.NaN
    df.loc[5, "c1"] = np.NaN
    df.loc[6, "c2"] = np.NaN

    fml = "y ~ x1"
    re_formula = "1 + z1"
    vc_formula = {"a": "0 + c1", "b": "0 + c2"}
    for include_re in False, True:
        for include_vc in False, True:
            kwargs = {}
            dx = df.copy()
            va = ["y", "g", "x1"]
            if include_re:
                kwargs["re_formula"] = re_formula
                va.append("z1")
            if include_vc:
                kwargs["vc_formula"] = vc_formula
                va.extend(["c1", "c2"])

            dx = dx[va].dropna()

            # Some of these models are severely misspecified with
            # small n, so produce convergence warnings.  Not relevant
            # to what we are checking here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Drop missing externally
                model1 = MixedLM.from_formula(
                    fml, groups="g", data=dx, **kwargs)
                result1 = model1.fit()

                # MixeLM handles missing
                model2 = MixedLM.from_formula(
                    fml, groups="g", data=df, missing='drop', **kwargs)
                result2 = model2.fit()

                assert_allclose(result1.params, result2.params)
                assert_allclose(result1.bse, result2.bse)
                assert_equal(len(result1.fittedvalues), result1.nobs)


def test_summary_col():
    from statsmodels.iolib.summary2 import summary_col
    ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # hard coded simulated y
    # ids = np.asarray(ids)
    # np.random.seed(123987)
    # y = x + np.array([-1, 0, 1])[ids - 1] + 2 * np.random.randn(len(y))
    y = np.array([
        1.727, -1.037, 2.904, 3.569, 4.629, 5.736, 6.747, 7.020, 5.624, 10.155,
        10.400, 17.164, 17.276, 14.988, 14.453
    ])
    d = {'Y': y, 'X': x, 'IDS': ids}
    d = pd.DataFrame(d)

    # provide start_params to speed up convergence
    sp1 = np.array([-1.26722599, 1.1617587, 0.19547518])
    mod1 = MixedLM.from_formula('Y ~ X', d, groups=d['IDS'])
    results1 = mod1.fit(start_params=sp1)
    sp2 = np.array([3.48416861, 0.55287862, 1.38537901])
    mod2 = MixedLM.from_formula('X ~ Y', d, groups=d['IDS'])
    results2 = mod2.fit(start_params=sp2)

    out = summary_col([results1, results2], stars=True)
    s = ('\n=============================\n              Y         X    \n'
         '-----------------------------\nGroup Var 0.1955    1.3854   \n'
         '          (0.6032)  (2.7377) \nIntercept -1.2672   3.4842*  \n'
         '          (1.6546)  (1.8882) \nX         1.1618***          \n'
         '          (0.1959)           \nY                   0.5529***\n'
         '                    (0.2080) \n=============================\n'
         'Standard errors in\nparentheses.\n* p<.1, ** p<.05, ***p<.01')
    assert_equal(str(out), s)


@pytest.mark.slow
def test_random_effects_getters():
    # Simulation-based test to make sure that the BLUPs and actual
    # random effects line up.

    np.random.seed(34234)
    ng = 500  # number of groups
    m = 10  # group size

    y, x, z, v0, v1, g, b, c0, c1 = [], [], [], [], [], [], [], [], []

    for i in range(ng):

        # Fixed effects
        xx = np.random.normal(size=(m, 2))
        yy = xx[:, 0] + 0.5 * np.random.normal(size=m)

        # Random effects (re_formula)
        zz = np.random.normal(size=(m, 2))
        bb = np.random.normal(size=2)
        bb[0] *= 3
        bb[1] *= 1
        yy += np.dot(zz, bb).flat
        b.append(bb)

        # First variance component
        vv0 = np.kron(np.r_[0, 1], np.ones(m // 2)).astype(int)
        cc0 = np.random.normal(size=2)
        yy += cc0[vv0]
        v0.append(vv0)
        c0.append(cc0)

        # Second variance component
        vv1 = np.kron(np.ones(m // 2), np.r_[0, 1]).astype(int)
        cc1 = np.random.normal(size=2)
        yy += cc1[vv1]
        v1.append(vv1)
        c1.append(cc1)

        y.append(yy)
        x.append(xx)
        z.append(zz)
        g.append(["g%d" % i] * m)

    y = np.concatenate(y)
    x = np.concatenate(x)
    z = np.concatenate(z)
    v0 = np.concatenate(v0)
    v1 = np.concatenate(v1)
    g = np.concatenate(g)
    df = pd.DataFrame({
        "y": y,
        "x0": x[:, 0],
        "x1": x[:, 1],
        "z0": z[:, 0],
        "z1": z[:, 1],
        "v0": v0,
        "v1": v1,
        "g": g
    })

    b = np.asarray(b)
    c0 = np.asarray(c0)
    c1 = np.asarray(c1)
    cc = np.concatenate((c0, c1), axis=1)

    model = MixedLM.from_formula(
        "y ~ x0 + x1",
        re_formula="~0 + z0 + z1",
        vc_formula={
            "v0": "~0+C(v0)",
            "v1": "0+C(v1)"
        },
        groups="g",
        data=df)
    result = model.fit()

    ref = result.random_effects
    b0 = [ref["g%d" % k][0:2] for k in range(ng)]
    b0 = np.asarray(b0)
    assert (np.corrcoef(b0[:, 0], b[:, 0])[0, 1] > 0.8)
    assert (np.corrcoef(b0[:, 1], b[:, 1])[0, 1] > 0.8)

    cf0 = [ref["g%d" % k][2:6] for k in range(ng)]
    cf0 = np.asarray(cf0)
    for k in range(4):
        assert (np.corrcoef(cf0[:, k], cc[:, k])[0, 1] > 0.8)

    # Smoke test for predictive covariances
    refc = result.random_effects_cov
    for g in refc.keys():
        p = ref[g].size
        assert (refc[g].shape == (p, p))


def check_smw_solver(p, q, r, s):
    # Helper to check that _smw_solver results do in fact solve the desired
    # SMW equation
    d = q - r

    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    B[0:r, 0:r] = np.random.normal(size=(r, r))
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = 0.5

    x = np.random.normal(size=p)
    y2 = np.linalg.solve(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)), x)

    f = _smw_solver(s, A, AtA, Qi, di)
    y1 = f(x)
    assert_allclose(y1, y2)

    f = _smw_solver(s, sparse.csr_matrix(A), sparse.csr_matrix(AtA), Qi,
                    di)
    y1 = f(x)
    assert_allclose(y1, y2)


class TestSMWSolver:
    @classmethod
    def setup_class(cls):
        np.random.seed(23)

    @pytest.mark.parametrize("p", [5, 10])
    @pytest.mark.parametrize("q", [4, 8])
    @pytest.mark.parametrize("r", [2, 3])
    @pytest.mark.parametrize("s", [0, 0.5])
    def test_smw_solver(self, p, q, r, s):
        check_smw_solver(p, q, r, s)


def check_smw_logdet(p, q, r, s):
    # Helper to check that _smw_logdet results match non-optimized equivalent
    d = q - r
    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)

    B = np.zeros((q, q))
    c = np.random.normal(size=(r, r))
    B[0:r, 0:r] = np.dot(c.T, c)
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = 0.5

    _, d2 = np.linalg.slogdet(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)))

    _, bd = np.linalg.slogdet(B)
    d1 = _smw_logdet(s, A, AtA, Qi, di, bd)
    # GH 5642, OSX OpenBlas tolerance increase
    rtol = 1e-6 if PLATFORM_OSX else 1e-7
    assert_allclose(d1, d2, rtol=rtol)


class TestSMWLogdet:
    @classmethod
    def setup_class(cls):
        np.random.seed(23)

    @pytest.mark.parametrize("p", [5, 10])
    @pytest.mark.parametrize("q", [4, 8])
    @pytest.mark.parametrize("r", [2, 3])
    @pytest.mark.parametrize("s", [0, 0.5])
    def test_smw_logdet(self, p, q, r, s):
        check_smw_logdet(p, q, r, s)


def test_singular():
    # Issue #7051

    np.random.seed(3423)
    n = 100

    data = np.random.randn(n, 2)
    df = pd.DataFrame(data, columns=['Y', 'X'])
    df['class'] = pd.Series([i % 3 for i in df.index], index=df.index)

    with pytest.warns(Warning) as wrn:
        md = MixedLM.from_formula("Y ~ X", df, groups=df['class'])
        mdf = md.fit()
        mdf.summary()
        if not wrn:
            pytest.fail("warning expected")


def test_get_distribution():

    np.random.seed(234)

    n = 100
    n_groups = 10
    fe_params = np.r_[1, -2]
    cov_re = np.asarray([[1, 0.5], [0.5, 2]])
    vcomp = np.r_[0.5**2, 1.5**2]
    scale = 1.5

    exog_fe = np.random.normal(size=(n, 2))
    exog_re = np.random.normal(size=(n, 2))
    exog_vca = np.random.normal(size=(n, 2))
    exog_vcb = np.random.normal(size=(n, 2))

    groups = np.repeat(np.arange(n_groups, dtype=int),
                       n / n_groups)

    ey = np.dot(exog_fe, fe_params)

    u = np.random.normal(size=(n_groups, 2))
    u = np.dot(u, np.linalg.cholesky(cov_re).T)

    u1 = np.sqrt(vcomp[0]) * np.random.normal(size=(n_groups, 2))
    u2 = np.sqrt(vcomp[1]) * np.random.normal(size=(n_groups, 2))

    y = ey + (u[groups, :] * exog_re).sum(1)
    y += (u1[groups, :] * exog_vca).sum(1)
    y += (u2[groups, :] * exog_vcb).sum(1)
    y += np.sqrt(scale) * np.random.normal(size=n)

    df = pd.DataFrame({"y": y, "x1": exog_fe[:, 0], "x2": exog_fe[:, 1],
                       "z0": exog_re[:, 0], "z1": exog_re[:, 1],
                       "grp": groups})
    df["z2"] = exog_vca[:, 0]
    df["z3"] = exog_vca[:, 1]
    df["z4"] = exog_vcb[:, 0]
    df["z5"] = exog_vcb[:, 1]

    vcf = {"a": "0 + z2 + z3", "b": "0 + z4 + z5"}
    m = MixedLM.from_formula("y ~ 0 + x1 + x2", groups="grp",
                             re_formula="0 + z0 + z1",
                             vc_formula=vcf, data=df)

    # Build a params vector that is comparable to
    # MixedLMResults.params
    import statsmodels
    mp = statsmodels.regression.mixed_linear_model.MixedLMParams
    po = mp.from_components(fe_params=fe_params, cov_re=cov_re,
                            vcomp=vcomp)
    pa = po.get_packed(has_fe=True, use_sqrt=False)
    pa[len(fe_params):] /= scale

    # Get a realization
    dist = m.get_distribution(pa, scale, None)
    yr = dist.rvs(0)

    # Check the overall variance
    v = (np.dot(exog_re, cov_re) * exog_re).sum(1).mean()
    v += vcomp[0] * (exog_vca**2).sum(1).mean()
    v += vcomp[1] * (exog_vcb**2).sum(1).mean()
    v += scale
    assert_allclose(np.var(yr - ey), v, rtol=1e-2, atol=1e-4)
