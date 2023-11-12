import numpy as np
from statsmodels.discrete.conditional_models import (
      ConditionalLogit, ConditionalPoisson, ConditionalMNLogit)
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd


def test_logit_1d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalLogit(y, x, groups=g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x, ]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x)).squeeze()
        assert_allclose(grad, ngrad)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        grad = approx_fprime(np.r_[x, ], model.loglike).squeeze()
        score = model.score(np.r_[x, ])
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[0.9272407], rtol=1e-5)
    assert_allclose(result.bse, np.r_[1.295155], rtol=1e-5)


def test_logit_2d():

    y = np.r_[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    model = ConditionalLogit(y, x, groups=g)

    # Check the gradient for the denominator of the partial likelihood
    for x in -1, 0, 1, 2:
        params = np.r_[x, -1.5*x]
        _, grad = model._denom_grad(0, params)
        ngrad = approx_fprime(params, lambda x: model._denom(0, x))
        assert_allclose(grad, ngrad, rtol=1e-5)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        params = np.r_[-0.5*x, 0.5*x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[1.011074, 1.236758], rtol=1e-3)
    assert_allclose(result.bse, np.r_[1.420784, 1.361738], rtol=1e-5)

    result.summary()


def test_formula():

    for j in 0, 1:

        np.random.seed(34234)
        n = 200
        y = np.random.randint(0, 2, size=n)
        x1 = np.random.normal(size=n)
        x2 = np.random.normal(size=n)
        g = np.random.randint(0, 25, size=n)

        x = np.hstack((x1[:, None], x2[:, None]))
        if j == 0:
            model1 = ConditionalLogit(y, x, groups=g)
        else:
            model1 = ConditionalPoisson(y, x, groups=g)
        result1 = model1.fit()

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
        if j == 0:
            model2 = ConditionalLogit.from_formula(
                        "y ~ 0 + x1 + x2", groups="g", data=df)
        else:
            model2 = ConditionalPoisson.from_formula(
                        "y ~ 0 + x1 + x2", groups="g", data=df)
        result2 = model2.fit()

        assert_allclose(result1.params, result2.params, rtol=1e-5)
        assert_allclose(result1.bse, result2.bse, rtol=1e-5)
        assert_allclose(result1.cov_params(), result2.cov_params(), rtol=1e-5)
        assert_allclose(result1.tvalues, result2.tvalues, rtol=1e-5)


def test_poisson_1d():

    y = np.r_[3, 1, 1, 4, 5, 2, 0, 1, 6, 2]
    g = np.r_[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    x = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x = x[:, None]

    model = ConditionalPoisson(y, x, groups=g)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        grad = approx_fprime(np.r_[x, ], model.loglike).squeeze()
        score = model.score(np.r_[x, ])
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[0.6466272], rtol=1e-4)
    assert_allclose(result.bse, np.r_[0.4170918], rtol=1e-5)


def test_poisson_2d():

    y = np.r_[3, 1, 4, 8, 2, 5, 4, 7, 2, 6]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    x1 = np.r_[0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    x2 = np.r_[2, 1, 0, 0, 1, 2, 3, 2, 0, 1]
    x = np.empty((10, 2))
    x[:, 0] = x1
    x[:, 1] = x2

    model = ConditionalPoisson(y, x, groups=g)

    # Check the gradient for the loglikelihood
    for x in -1, 0, 1, 2:
        params = np.r_[-0.5*x, 0.5*x]
        grad = approx_fprime(params, model.loglike)
        score = model.score(params)
        assert_allclose(grad, score, rtol=1e-4)

    result = model.fit()

    # From Stata
    assert_allclose(result.params, np.r_[-.9478957, -.0134279], rtol=1e-3)
    assert_allclose(result.bse, np.r_[.3874942, .1686712], rtol=1e-5)

    result.summary()


def test_lasso_logistic():

    np.random.seed(3423948)

    n = 200
    groups = np.arange(10)
    groups = np.kron(groups, np.ones(n // 10))
    group_effects = np.random.normal(size=10)
    group_effects = np.kron(group_effects, np.ones(n // 10))

    x = np.random.normal(size=(n, 4))
    params = np.r_[0, 0, 1, 0]
    lin_pred = np.dot(x, params) + group_effects

    mean = 1 / (1 + np.exp(-lin_pred))
    y = (np.random.uniform(size=n) < mean).astype(int)

    model0 = ConditionalLogit(y, x, groups=groups)
    result0 = model0.fit()

    # Should be the same as model0
    model1 = ConditionalLogit(y, x, groups=groups)
    result1 = model1.fit_regularized(L1_wt=0, alpha=0)

    assert_allclose(result0.params, result1.params, rtol=1e-3)

    model2 = ConditionalLogit(y, x, groups=groups)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.05)

    # Rxegression test
    assert_allclose(result2.params, np.r_[0, 0, 0.55235152, 0], rtol=1e-4)

    # Test with formula
    df = pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2],
                       "x4": x[:, 3], "groups": groups})
    fml = "y ~ 0 + x1 + x2 + x3 + x4"
    model3 = ConditionalLogit.from_formula(fml, groups="groups", data=df)
    result3 = model3.fit_regularized(L1_wt=1, alpha=0.05)
    assert_allclose(result2.params, result3.params)


def test_lasso_poisson():

    np.random.seed(342394)

    n = 200
    groups = np.arange(10)
    groups = np.kron(groups, np.ones(n // 10))
    group_effects = np.random.normal(size=10)
    group_effects = np.kron(group_effects, np.ones(n // 10))

    x = np.random.normal(size=(n, 4))
    params = np.r_[0, 0, 1, 0]
    lin_pred = np.dot(x, params) + group_effects

    mean = np.exp(lin_pred)
    y = np.random.poisson(mean)

    model0 = ConditionalPoisson(y, x, groups=groups)
    result0 = model0.fit()

    # Should be the same as model0
    model1 = ConditionalPoisson(y, x, groups=groups)
    result1 = model1.fit_regularized(L1_wt=0, alpha=0)

    assert_allclose(result0.params, result1.params, rtol=1e-3)

    model2 = ConditionalPoisson(y, x, groups=groups)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.2)

    # Regression test
    assert_allclose(result2.params, np.r_[0, 0, 0.91697508, 0], rtol=1e-4)

    # Test with formula
    df = pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2],
                       "x4": x[:, 3], "groups": groups})
    fml = "y ~ 0 + x1 + x2 + x3 + x4"
    model3 = ConditionalPoisson.from_formula(fml, groups="groups", data=df)
    result3 = model3.fit_regularized(L1_wt=1, alpha=0.2)
    assert_allclose(result2.params, result3.params)


def gen_mnlogit(n):

    np.random.seed(235)

    g = np.kron(np.ones(5), np.arange(n//5))
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    xm = np.concatenate((x1[:, None], x2[:, None]), axis=1)
    pa = np.array([[0, 1, -1], [0, 2, -1]])
    lpr = np.dot(xm, pa)
    pr = np.exp(lpr)
    pr /= pr.sum(1)[:, None]
    cpr = pr.cumsum(1)
    y = 2 * np.ones(n)
    u = np.random.uniform(size=n)
    y[u < cpr[:, 2]] = 2
    y[u < cpr[:, 1]] = 1
    y[u < cpr[:, 0]] = 0

    df = pd.DataFrame({"y": y, "x1": x1,
                       "x2": x2, "g": g})
    return df


def test_conditional_mnlogit_grad():

    df = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula(
                "y ~ 0 + x1 + x2", groups="g", data=df)

    # Compare the gradients to numeric gradients
    for _ in range(5):
        za = np.random.normal(size=4)
        grad = model.score(za)
        ngrad = approx_fprime(za, model.loglike)
        assert_allclose(grad, ngrad, rtol=1e-5, atol=1e-3)


def test_conditional_mnlogit_2d():

    df = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula(
                "y ~ 0 + x1 + x2", groups="g", data=df)
    result = model.fit()

    # Regression tests
    assert_allclose(
        result.params,
        np.asarray([[0.75592035, -1.58565494],
                    [1.82919869, -1.32594231]]),
        rtol=1e-5, atol=1e-5)
    assert_allclose(
        result.bse,
        np.asarray([[0.68099698, 0.70142727],
                    [0.65190315, 0.59653771]]),
        rtol=1e-5, atol=1e-5)


def test_conditional_mnlogit_3d():

    df = gen_mnlogit(90)
    df["x3"] = np.random.normal(size=df.shape[0])
    model = ConditionalMNLogit.from_formula(
                "y ~ 0 + x1 + x2 + x3", groups="g", data=df)
    result = model.fit()

    # Regression tests
    assert_allclose(
        result.params,
        np.asarray([[ 0.729629, -1.633673],
                    [ 1.879019, -1.327163],
                    [-0.114124, -0.109378]]),
        atol=1e-5, rtol=1e-5)

    assert_allclose(
        result.bse,
        np.asarray([[0.682965, 0.60472],
                    [0.672947, 0.42401],
                    [0.722631, 0.33663]]),
        atol=1e-5, rtol=1e-5)

    # Smoke test
    result.summary()
