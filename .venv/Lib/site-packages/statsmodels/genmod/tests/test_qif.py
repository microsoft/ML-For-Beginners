import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.genmod.qif import (QIF, QIFIndependence, QIFExchangeable,
                                    QIFAutoregressive)
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.genmod import families


@pytest.mark.parametrize("fam", [families.Gaussian(), families.Poisson(),
                         families.Binomial()])
@pytest.mark.parametrize("cov_struct", [QIFIndependence(), QIFExchangeable(),
                         QIFAutoregressive()])
def test_qif_numdiff(fam, cov_struct):
    # Test the analytic scores against numeric derivatives

    np.random.seed(234234)
    n = 200
    q = 4
    x = np.random.normal(size=(n, 3))
    if isinstance(fam, families.Gaussian):
        e = np.kron(np.random.normal(size=n//q), np.ones(q))
        e = np.sqrt(0.5)*e + np.sqrt(1 - 0.5**2)*np.random.normal(size=n)
        y = x.sum(1) + e
    elif isinstance(fam, families.Poisson):
        y = np.random.poisson(5, size=n)
    elif isinstance(fam, families.Binomial):
        y = np.random.randint(0, 2, size=n)
    g = np.kron(np.arange(n//q), np.ones(q)).astype(int)

    model = QIF(y, x, groups=g, family=fam, cov_struct=cov_struct)

    for _ in range(5):

        pt = np.random.normal(size=3)

        # Check the Jacobian of the vector of estimating equations.
        _, grad, _, _, gn_deriv = model.objective(pt)

        def llf_gn(params):
            return model.objective(params)[3]
        gn_numdiff = approx_fprime(pt, llf_gn, 1e-7)
        assert_allclose(gn_deriv, gn_numdiff, 1e-4)

        # Check the gradient of the QIF
        def llf(params):
            return model.objective(params)[0]
        grad_numdiff = approx_fprime(pt, llf, 1e-7)
        assert_allclose(grad, grad_numdiff, 1e-4)


@pytest.mark.parametrize("fam", [families.Gaussian(), families.Poisson(),
                         families.Binomial()])
@pytest.mark.parametrize("cov_struct", [QIFIndependence(), QIFExchangeable(),
                         QIFAutoregressive()])
def test_qif_fit(fam, cov_struct):

    np.random.seed(234234)

    n = 1000
    q = 4
    params = np.r_[1, -0.5, 0.2]
    x = np.random.normal(size=(n, len(params)))
    if isinstance(fam, families.Gaussian):
        e = np.kron(np.random.normal(size=n//q), np.ones(q))
        e = np.sqrt(0.5)*e + np.sqrt(1 - 0.5**2)*np.random.normal(size=n)
        y = np.dot(x, params) + e
    elif isinstance(fam, families.Poisson):
        lpr = np.dot(x, params)
        mean = np.exp(lpr)
        y = np.random.poisson(mean)
    elif isinstance(fam, families.Binomial):
        lpr = np.dot(x, params)
        mean = 1 / (1 + np.exp(-lpr))
        y = (np.random.uniform(0, 1, size=n) < mean).astype(int)
    g = np.kron(np.arange(n // q), np.ones(q)).astype(int)

    model = QIF(y, x, groups=g, family=fam, cov_struct=cov_struct)
    rslt = model.fit()

    # Slack comparison to population values
    assert_allclose(rslt.params, params, atol=0.05, rtol=0.05)

    # Smoke test
    _ = rslt.summary()


@pytest.mark.parametrize("cov_struct", [QIFIndependence(), QIFExchangeable(),
                         QIFAutoregressive()])
def test_formula(cov_struct):

    np.random.seed(3423)

    y = np.random.normal(size=100)
    x = np.random.normal(size=(100, 2))
    groups = np.kron(np.arange(25), np.ones(4))

    model1 = QIF(y, x, groups=groups, cov_struct=cov_struct)
    result1 = model1.fit()

    df = pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "groups": groups})

    model2 = QIF.from_formula("y ~ 0 + x1 + x2", groups="groups",
                              cov_struct=cov_struct, data=df)
    result2 = model2.fit()

    assert_allclose(result1.params, result2.params)
    assert_allclose(result1.bse, result2.bse)

    if not isinstance(cov_struct, QIFIndependence):
        _ = result2.bic
        _ = result2.aic
