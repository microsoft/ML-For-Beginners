import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal


def test_pat():

    x = np.asarray([[1, np.nan, 3], [np.nan, 2, np.nan], [3, np.nan, 0],
                    [np.nan, 1, np.nan], [3, 2, 1]])
    bm = BayesGaussMI(x)
    assert_allclose(bm.patterns[0], np.r_[0, 2])
    assert_allclose(bm.patterns[1], np.r_[1, 3])


def test_2x2():

    # Generate correlated data with mean and variance
    np.random.seed(3434)
    x = np.random.normal(size=(1000, 2))
    r = 0.5
    x[:, 1] = r*x[:, 0] + np.sqrt(1-r**2)*x[:, 1]
    x[:, 0] *= 2
    x[:, 1] *= 3
    x[:, 0] += 1
    x[:, 1] -= 2

    # Introduce some missing values
    u = np.random.normal(size=x.shape[0])
    x[u > 1, 0] = np.nan
    u = np.random.normal(size=x.shape[0])
    x[u > 1, 1] = np.nan

    bm = BayesGaussMI(x)

    # Burn-in
    for k in range(500):
        bm.update()

    # Estimate the posterior mean
    mean = 0
    cov = 0
    dmean = 0
    dcov = 0
    for k in range(500):
        bm.update()
        mean += bm.mean
        cov += bm.cov
        dmean += bm.data.mean(0)
        dcov += np.cov(bm.data.T)
    mean /= 500
    cov /= 500
    dmean /= 500
    dcov /= 500

    assert_allclose(mean, np.r_[1, -2], 0.1)
    assert_allclose(dmean, np.r_[1, -2], 0.1)
    assert_allclose(cov, np.asarray([[4, 6*r], [6*r, 9]]), 0.1)
    assert_allclose(dcov, np.asarray([[4, 6*r], [6*r, 9]]), 0.1)


def test_MI():

    np.random.seed(414)
    x = np.random.normal(size=(200, 4))
    x[[1, 3, 9], 0] = np.nan
    x[[1, 4, 3], 1] = np.nan
    x[[2, 11, 21], 2] = np.nan
    x[[11, 22, 99], 3] = np.nan

    def model_args_fn(x):
        # Return endog, exog
        # Regress x0 on x1 and x2
        if type(x) is np.ndarray:
            return (x[:, 0], x[:, 1:])
        else:
            return (x.iloc[:, 0].values, x.iloc[:, 1:].values)

    for j in (0, 1):
        np.random.seed(2342)
        imp = BayesGaussMI(x.copy())
        mi = MI(imp, sm.OLS, model_args_fn, burn=0)
        r = mi.fit()
        r.summary()  # smoke test
        # TODO: why does the test tolerance need to be so slack?
        # There is unexpected variation across versions
        assert_allclose(r.params, np.r_[
            -0.05347919, -0.02479701, 0.10075517], 0.25, 0)

        c = np.asarray([[0.00418232, 0.00029746, -0.00035057],
                        [0.00029746, 0.00407264, 0.00019496],
                        [-0.00035057, 0.00019496, 0.00509413]])
        assert_allclose(r.cov_params(), c, 0.3, 0)

        # Test with ndarray and pandas input
        x = pd.DataFrame(x)


def test_MI_stat():
    # Test for MI where we know statistically what should happen. The
    # analysis model is x0 ~ x1 with standard error 1/sqrt(n) for the
    # slope parameter.  The nominal n is 1000, but half of the cases
    # have missing x1.  Then we introduce x2 that is either
    # independent of x1, or almost perfectly correlated with x1.  In
    # the first case the SE is 1/sqrt(500), in the second case the SE
    # is 1/sqrt(1000).

    np.random.seed(414)
    z = np.random.normal(size=(1000, 3))
    z[:, 0] += 0.5*z[:, 1]

    # Control the degree to which x2 proxies for x1
    exp = [1/np.sqrt(500), 1/np.sqrt(1000)]
    fmi = [0.5, 0]
    for j, r in enumerate((0, 0.9999)):

        x = z.copy()
        x[:, 2] = r*x[:, 1] + np.sqrt(1 - r**2)*x[:, 2]
        x[0:500, 1] = np.nan

        def model_args(x):
            # Return endog, exog
            # Regress x1 on x2
            return (x[:, 0], x[:, 1])

        np.random.seed(2342)
        imp = BayesGaussMI(x.copy())
        mi = MI(imp, sm.OLS, model_args, nrep=100, skip=10)
        r = mi.fit()

        # Check the SE
        d = np.abs(r.bse[0] - exp[j]) / exp[j]
        assert d < 0.03

        # Check the FMI
        d = np.abs(r.fmi[0] - fmi[j])
        assert d < 0.05


def test_mi_formula():

    np.random.seed(414)
    x = np.random.normal(size=(200, 4))
    x[[1, 3, 9], 0] = np.nan
    x[[1, 4, 3], 1] = np.nan
    x[[2, 11, 21], 2] = np.nan
    x[[11, 22, 99], 3] = np.nan
    df = pd.DataFrame({"y": x[:, 0], "x1": x[:, 1],
                       "x2": x[:, 2], "x3": x[:, 3]})
    fml = "y ~ 0 + x1 + x2 + x3"

    def model_kwds_fn(x):
        return {"data": x}

    np.random.seed(2342)
    imp = BayesGaussMI(df.copy())
    mi = MI(imp, sm.OLS, formula=fml, burn=0,
            model_kwds_fn=model_kwds_fn)

    results_cb = lambda x: x

    r = mi.fit(results_cb=results_cb)
    r.summary()  # smoke test
    # TODO: why does the test tolerance need to be so slack?
    # There is unexpected variation across versions
    assert_allclose(r.params, np.r_[
            -0.05347919, -0.02479701, 0.10075517], 0.25, 0)

    c = np.asarray([[0.00418232, 0.00029746, -0.00035057],
                    [0.00029746, 0.00407264, 0.00019496],
                    [-0.00035057, 0.00019496, 0.00509413]])
    assert_allclose(r.cov_params(), c, 0.3, 0)

    assert_equal(len(r.results), 20)
