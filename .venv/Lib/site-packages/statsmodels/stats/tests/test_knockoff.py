import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import statsmodels.api as sm
from statsmodels.stats import knockoff_regeffects as kr
from statsmodels.stats._knockoff import (RegressionFDR,
                                         _design_knockoff_equi,
                                         _design_knockoff_sdp)

try:
    import cvxopt  # noqa:F401
    has_cvxopt = True
except ImportError:
    has_cvxopt = False


def test_equi():
    # Test the structure of the equivariant knockoff construction.

    np.random.seed(2342)
    exog = np.random.normal(size=(10, 4))

    exog1, exog2, sl = _design_knockoff_equi(exog)

    exoga = np.concatenate((exog1, exog2), axis=1)

    gmat = np.dot(exoga.T, exoga)

    cm1 = gmat[0:4, 0:4]
    cm2 = gmat[4:, 4:]
    cm3 = gmat[0:4, 4:]

    assert_allclose(cm1, cm2, rtol=1e-4, atol=1e-4)
    assert_allclose(cm1 - cm3, np.diag(sl * np.ones(4)), rtol=1e-4, atol=1e-4)


def test_sdp():
    # Test the structure of the SDP knockoff construction.

    if not has_cvxopt:
        return

    np.random.seed(2342)
    exog = np.random.normal(size=(10, 4))

    exog1, exog2, sl = _design_knockoff_sdp(exog)

    exoga = np.concatenate((exog1, exog2), axis=1)

    gmat = np.dot(exoga.T, exoga)

    cm1 = gmat[0:4, 0:4]
    cm2 = gmat[4:, 4:]
    cm3 = gmat[0:4, 4:]

    assert_allclose(cm1, cm2, rtol=1e-4, atol=1e-4)
    assert_allclose(cm1 - cm3, np.diag(sl * np.ones(4)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [49, 50])
@pytest.mark.parametrize("tester", [
                   kr.CorrelationEffects(),
                   kr.ForwardEffects(pursuit=False),
                   kr.ForwardEffects(pursuit=True),
                   kr.OLSEffects(),
                   kr.RegModelEffects(sm.OLS),
                   kr.RegModelEffects(sm.OLS, True,
                                      fit_kws={"L1_wt": 0, "alpha": 1}),
                ])
@pytest.mark.parametrize("method", ["equi", "sdp"])
def test_testers(p, tester, method):

    if method == "sdp" and not has_cvxopt:
        return

    np.random.seed(2432)
    n = 200

    y = np.random.normal(size=n)
    x = np.random.normal(size=(n, p))

    kn = RegressionFDR(y, x, tester, design_method=method)
    assert_equal(len(kn.stats), p)
    assert_equal(len(kn.fdr), p)
    kn.summary()  # smoke test


@pytest.mark.slow
@pytest.mark.parametrize("method", ["equi", "sdp"])
@pytest.mark.parametrize("tester,n,p,es", [
    [kr.CorrelationEffects(), 300, 100, 6],
    [kr.ForwardEffects(pursuit=False), 300, 100, 3.5],
    [kr.ForwardEffects(pursuit=True), 300, 100, 3.5],
    [kr.OLSEffects(), 3000, 200, 3.5],
])
def test_sim(method, tester, n, p, es):
    # This function assesses the performance of the knockoff approach
    # relative to its theoretical claims.

    if method == "sdp" and not has_cvxopt:
        return

    np.random.seed(43234)

    # Number of variables with a non-zero coefficient
    npos = 30

    # Aim to control FDR to this level
    target_fdr = 0.2

    # Number of siumulation replications
    nrep = 10

    if method == "sdp" and not has_cvxopt:
        return

    fdr, power = 0, 0
    for k in range(nrep):

        # Generate the predictors
        x = np.random.normal(size=(n, p))
        x /= np.sqrt(np.sum(x*x, 0))

        # Generate the response variable
        coeff = es * (-1)**np.arange(npos)
        y = np.dot(x[:, 0:npos], coeff) + np.random.normal(size=n)

        kn = RegressionFDR(y, x, tester)

        # Threshold to achieve the target FDR
        tr = kn.threshold(target_fdr)

        # Number of selected coefficients
        cp = np.sum(kn.stats >= tr)

        # Number of false positives
        fp = np.sum(kn.stats[npos:] >= tr)

        # Observed FDR
        fdr += fp / max(cp, 1)

        # Proportion of true positives that are detected
        power += np.mean(kn.stats[0:npos] >= tr)

        # The estimated FDR may never exceed the target FDR
        estimated_fdr = (np.sum(kn.stats <= -tr) /
                         (1 + np.sum(kn.stats >= tr)))
        assert_equal(estimated_fdr < target_fdr, True)

    power /= nrep
    fdr /= nrep

    # Check for reasonable power
    assert_array_equal(power > 0.6, True)

    # Check that we are close to the target FDR
    assert_array_equal(fdr < target_fdr + 0.1, True)
