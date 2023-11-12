import itertools
import os

import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
                           assert_equal, assert_)
import pandas as pd
import pytest

# TODO: Include some corner cases: data sets with empty strata, strata
#      with no events, entry times after censoring times, etc.

# All the R results
from .results import survival_r_results
from .results import survival_enet_r_results

"""
Tests of PHReg against R coxph.

Tests include entry times and stratification.

phreg_gentests.py generates the test data sets and puts them into the
results folder.

survival.R runs R on all the test data sets and constructs the
survival_r_results module.
"""

# Arguments passed to the PHReg fit method.
args = {"method": "bfgs", "disp": 0}


def get_results(n, p, ext, ties):
    if ext is None:
        coef_name = "coef_%d_%d_%s" % (n, p, ties)
        se_name = "se_%d_%d_%s" % (n, p, ties)
        time_name = "time_%d_%d_%s" % (n, p, ties)
        hazard_name = "hazard_%d_%d_%s" % (n, p, ties)
    else:
        coef_name = "coef_%d_%d_%s_%s" % (n, p, ext, ties)
        se_name = "se_%d_%d_%s_%s" % (n, p, ext, ties)
        time_name = "time_%d_%d_%s_%s" % (n, p, ext, ties)
        hazard_name = "hazard_%d_%d_%s_%s" % (n, p, ext, ties)
    coef = getattr(survival_r_results, coef_name)
    se = getattr(survival_r_results, se_name)
    time = getattr(survival_r_results, time_name)
    hazard = getattr(survival_r_results, hazard_name)
    return coef, se, time, hazard

class TestPHReg:

    # Load a data file from the results directory
    @staticmethod
    def load_file(fname):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data = np.genfromtxt(os.path.join(cur_dir, 'results', fname),
                             delimiter=" ")
        time = data[:,0]
        status = data[:,1]
        entry = data[:,2]
        exog = data[:,3:]

        return time, status, entry, exog

    # Run a single test against R output
    @staticmethod
    def do1(fname, ties, entry_f, strata_f):

        # Read the test data.
        time, status, entry, exog = TestPHReg.load_file(fname)
        n = len(time)

        vs = fname.split("_")
        n = int(vs[2])
        p = int(vs[3].split(".")[0])
        ties1 = ties[0:3]

        # Needs to match the kronecker statement in survival.R
        strata = np.kron(range(5), np.ones(n // 5))

        # No stratification or entry times
        mod = PHReg(time, exog, status, ties=ties)
        phrb = mod.fit(**args)
        coef_r, se_r, time_r, hazard_r = get_results(n, p, None, ties1)
        assert_allclose(phrb.params, coef_r, rtol=1e-3)
        assert_allclose(phrb.bse, se_r, rtol=1e-4)
        time_h, cumhaz, surv = phrb.baseline_cumulative_hazard[0]

        # Entry times but no stratification
        phrb = PHReg(time, exog, status, entry=entry,
                     ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "et", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-3)
        assert_allclose(phrb.bse, se, rtol=1e-3)

        # Stratification but no entry times
        phrb = PHReg(time, exog, status, strata=strata,
                      ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "st", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-4)
        assert_allclose(phrb.bse, se, rtol=1e-4)

        # Stratification and entry times
        phrb = PHReg(time, exog, status, entry=entry,
                     strata=strata, ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "et_st", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-3)
        assert_allclose(phrb.bse, se, rtol=1e-4)

        #smoke test
        time_h, cumhaz, surv = phrb.baseline_cumulative_hazard[0]

    def test_missing(self):

        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        time[0:5] = np.nan
        status[5:10] = np.nan
        exog[10:15,:] = np.nan

        md = PHReg(time, exog, status, missing='drop')
        assert_allclose(len(md.endog), 185)
        assert_allclose(len(md.status), 185)
        assert_allclose(md.exog.shape, np.r_[185,4])

    def test_formula(self):

        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))
        entry = np.zeros_like(time)
        entry[0:10] = time[0:10] / 2

        df = pd.DataFrame({"time": time, "status": status,
                           "exog1": exog[:, 0], "exog2": exog[:, 1],
                           "exog3": exog[:, 2], "exog4": exog[:, 3],
                           "entry": entry})

        mod1 = PHReg(time, exog, status, entry=entry)
        rslt1 = mod1.fit()

        # works with "0 +" on RHS but issues warning
        fml = "time ~ exog1 + exog2 + exog3 + exog4"
        mod2 = PHReg.from_formula(fml, df, status=status,
                                  entry=entry)
        rslt2 = mod2.fit()

        mod3 = PHReg.from_formula(fml, df, status="status",
                                  entry="entry")
        rslt3 = mod3.fit()

        assert_allclose(rslt1.params, rslt2.params)
        assert_allclose(rslt1.params, rslt3.params)
        assert_allclose(rslt1.bse, rslt2.bse)
        assert_allclose(rslt1.bse, rslt3.bse)

    def test_formula_cat_interactions(self):

        time = np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]
        status = np.r_[1, 1, 0, 0, 1, 0, 1, 1, 1]
        x1 = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3]
        x2 = np.r_[1, 2, 3, 1, 2, 3, 1, 2, 3]
        df = pd.DataFrame({"time": time, "status": status,
                           "x1": x1, "x2": x2})

        model1 = PHReg.from_formula("time ~ C(x1) + C(x2) + C(x1)*C(x2)", status="status",
                                    data=df)
        assert_equal(model1.exog.shape, [9, 8])

    def test_predict_formula(self):

        n = 100
        np.random.seed(34234)
        time = 50 * np.random.uniform(size=n)
        status = np.random.randint(0, 2, n).astype(np.float64)
        exog = np.random.uniform(1, 2, size=(n, 2))

        df = pd.DataFrame({"time": time, "status": status,
                           "exog1": exog[:, 0], "exog2": exog[:, 1]})

        # Works with "0 +" on RHS but issues warning
        fml = "time ~ exog1 + np.log(exog2) + exog1*exog2"
        model1 = PHReg.from_formula(fml, df, status=status)
        result1 = model1.fit()

        from patsy import dmatrix
        dfp = dmatrix(model1.data.design_info, df)

        pr1 = result1.predict()
        pr2 = result1.predict(exog=df)
        pr3 = model1.predict(result1.params, exog=dfp)  # No standard errors
        pr4 = model1.predict(result1.params,
                             cov_params=result1.cov_params(),
                             exog=dfp)

        prl = (pr1, pr2, pr3, pr4)
        for i in range(4):
            for j in range(i):
                assert_allclose(prl[i].predicted_values,
                                prl[j].predicted_values)

        prl = (pr1, pr2, pr4)
        for i in range(3):
            for j in range(i):
                assert_allclose(prl[i].standard_errors, prl[j].standard_errors)

    def test_formula_args(self):

        np.random.seed(34234)
        n = 200
        time = 50 * np.random.uniform(size=n)
        status = np.random.randint(0, 2, size=n).astype(np.float64)
        exog = np.random.normal(size=(200, 2))
        offset = np.random.uniform(size=n)
        entry = np.random.uniform(0, 1, size=n) * time

        df = pd.DataFrame({"time": time, "status": status, "x1": exog[:, 0],
                           "x2": exog[:, 1], "offset": offset, "entry": entry})
        model1 = PHReg.from_formula("time ~ x1 + x2", status="status", offset="offset",
                                    entry="entry", data=df)
        result1 = model1.fit()
        model2 = PHReg.from_formula("time ~ x1 + x2", status=df.status, offset=df.offset,
                                    entry=df.entry, data=df)
        result2 = model2.fit()
        assert_allclose(result1.params, result2.params)
        assert_allclose(result1.bse, result2.bse)

    def test_offset(self):

        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        for ties in "breslow", "efron":
            mod1 = PHReg(time, exog, status)
            rslt1 = mod1.fit()
            offset = exog[:,0] * rslt1.params[0]
            exog = exog[:, 1:]

            mod2 = PHReg(time, exog, status, offset=offset, ties=ties)
            rslt2 = mod2.fit()

            assert_allclose(rslt2.params, rslt1.params[1:])

    def test_post_estimation(self):
        # All regression tests
        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHReg(time, exog, status)
        rslt = mod.fit()
        mart_resid = rslt.martingale_residuals
        assert_allclose(np.abs(mart_resid).sum(), 120.72475743348433)

        w_avg = rslt.weighted_covariate_averages
        assert_allclose(np.abs(w_avg[0]).sum(0),
               np.r_[7.31008415, 9.77608674,10.89515885, 13.1106801])

        bc_haz = rslt.baseline_cumulative_hazard
        v = [np.mean(np.abs(x)) for x in bc_haz[0]]
        w = np.r_[23.482841556421608, 0.44149255358417017,
                  0.68660114081275281]
        assert_allclose(v, w)

        score_resid = rslt.score_residuals
        v = np.r_[ 0.50924792, 0.4533952, 0.4876718, 0.5441128]
        w = np.abs(score_resid).mean(0)
        assert_allclose(v, w)

        groups = np.random.randint(0, 3, 200)
        mod = PHReg(time, exog, status)
        rslt = mod.fit(groups=groups)
        robust_cov = rslt.cov_params()
        v = [0.00513432, 0.01278423, 0.00810427, 0.00293147]
        w = np.abs(robust_cov).mean(0)
        assert_allclose(v, w, rtol=1e-6)

        s_resid = rslt.schoenfeld_residuals
        ii = np.flatnonzero(np.isfinite(s_resid).all(1))
        s_resid = s_resid[ii, :]
        v = np.r_[0.85154336, 0.72993748, 0.73758071, 0.78599333]
        assert_allclose(np.abs(s_resid).mean(0), v)

    @pytest.mark.smoke
    def test_summary(self):
        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHReg(time, exog, status)
        rslt = mod.fit()
        smry = rslt.summary()

        strata = np.kron(np.arange(50), np.ones(4))
        mod = PHReg(time, exog, status, strata=strata)
        rslt = mod.fit()
        smry = rslt.summary()
        msg = "3 strata dropped for having no events"
        assert_(msg in str(smry))

        groups = np.kron(np.arange(25), np.ones(8))
        mod = PHReg(time, exog, status)
        rslt = mod.fit(groups=groups)
        smry = rslt.summary()

        entry = np.random.uniform(0.1, 0.8, 200) * time
        mod = PHReg(time, exog, status, entry=entry)
        rslt = mod.fit()
        smry = rslt.summary()
        msg = "200 observations have positive entry times"
        assert_(msg in str(smry))

    @pytest.mark.smoke
    def test_predict(self):
        # All smoke tests. We should be able to convert the lhr and hr
        # tests into real tests against R.  There are many options to
        # this function that may interact in complicated ways.  Only a
        # few key combinations are tested here.
        np.random.seed(34234)
        endog = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHReg(endog, exog, status)
        rslt = mod.fit()
        rslt.predict()
        for pred_type in 'lhr', 'hr', 'cumhaz', 'surv':
            rslt.predict(pred_type=pred_type)
            rslt.predict(endog=endog[0:10], pred_type=pred_type)
            rslt.predict(endog=endog[0:10], exog=exog[0:10,:],
                         pred_type=pred_type)

    @pytest.mark.smoke
    def test_get_distribution(self):
        np.random.seed(34234)
        n = 200
        exog = np.random.normal(size=(n, 2))
        lin_pred = exog.sum(1)
        elin_pred = np.exp(-lin_pred)
        time = -elin_pred * np.log(np.random.uniform(size=n))
        status = np.ones(n)
        status[0:20] = 0
        strata = np.kron(range(5), np.ones(n // 5))

        mod = PHReg(time, exog, status=status, strata=strata)
        rslt = mod.fit()

        dist = rslt.get_distribution()

        fitted_means = dist.mean()
        true_means = elin_pred
        fitted_var = dist.var()
        fitted_sd = dist.std()
        sample = dist.rvs()

    def test_fit_regularized(self):

        # Data set sizes
        for n,p in (50,2),(100,5):

            # Penalty weights
            for js,s in enumerate([0,0.1]):

                coef_name = "coef_%d_%d_%d" % (n, p, js)
                params = getattr(survival_enet_r_results, coef_name)

                fname = "survival_data_%d_%d.csv" % (n, p)
                time, status, entry, exog = self.load_file(fname)

                exog -= exog.mean(0)
                exog /= exog.std(0, ddof=1)

                model = PHReg(time, exog, status=status, ties='breslow')
                sm_result = model.fit_regularized(alpha=s)

                # The agreement is not very high, the issue may be on
                # the R side.  See below for further checks.
                assert_allclose(sm_result.params, params, rtol=0.3)

                # The penalized log-likelihood that we are maximizing.
                def plf(params):
                    llf = model.loglike(params) / len(time)
                    L1_wt = 1
                    llf = llf - s * ((1 - L1_wt)*np.sum(params**2) / 2 + L1_wt*np.sum(np.abs(params)))
                    return llf

                # Confirm that we are doing better than glmnet.
                llf_r = plf(params)
                llf_sm = plf(sm_result.params)
                assert_equal(np.sign(llf_sm - llf_r), 1)


cur_dir = os.path.dirname(os.path.abspath(__file__))
rdir = os.path.join(cur_dir, 'results')
fnames = os.listdir(rdir)
fnames = [x for x in fnames if x.startswith("survival")
          and x.endswith(".csv")]

ties = ("breslow", "efron")
entry_f = (False, True)
strata_f = (False, True)


@pytest.mark.parametrize('fname,ties,entry_f,strata_f',
                         list(itertools.product(fnames, ties, entry_f, strata_f)))
def test_r(fname, ties, entry_f, strata_f):
    TestPHReg.do1(fname, ties, entry_f, strata_f)
