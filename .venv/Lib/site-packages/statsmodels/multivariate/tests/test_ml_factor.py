import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings

# A small model for basic testing
def _toy():
    uniq = np.r_[4, 9, 16]
    load = np.asarray([[3, 1, 2], [2, 5, 8]]).T
    par = np.r_[2, 3, 4, 3, 1, 2, 2, 5, 8]
    corr = np.asarray([[1, .5, .25], [.5, 1, .5], [.25, .5, 1]])
    return uniq, load, corr, par


def test_loglike():

    uniq, load, corr, par = _toy()
    fa = Factor(n_factor=2, corr=corr)

    # Two ways of passing the parameters to loglike
    ll1 = fa.loglike((load, uniq))
    ll2 = fa.loglike(par)

    assert_allclose(ll1, ll2)


def test_score():

    uniq, load, corr, par = _toy()
    fa = Factor(n_factor=2, corr=corr)

    def f(par):
        return fa.loglike(par)

    par2 = np.r_[0.1, 0.2, 0.3, 0.4, 0.3, 0.1, 0.2, -0.2, 0, 0.8, 0.5, 0]

    for pt in (par, par2):
        g1 = approx_fprime(pt, f, 1e-8)
        g2 = fa.score(pt)
        assert_allclose(g1, g2, atol=1e-3)


def test_exact():
    # Test if we can recover exact factor-structured matrices with
    # default starting values.

    np.random.seed(23324)

    # Works for larger k_var but slow for routine testing.
    for k_var in 5, 10, 25:
        for n_factor in 1, 2, 3:
            load = np.random.normal(size=(k_var, n_factor))
            uniq = np.linspace(1, 2, k_var)
            c = np.dot(load, load.T)
            c.flat[::c.shape[0]+1] += uniq
            s = np.sqrt(np.diag(c))
            c /= np.outer(s, s)
            fa = Factor(corr=c, n_factor=n_factor, method='ml')
            rslt = fa.fit()
            assert_allclose(rslt.fitted_cov, c, rtol=1e-4, atol=1e-4)
            rslt.summary()  # smoke test


def test_exact_em():
    # Test if we can recover exact factor-structured matrices with
    # default starting values using the EM algorithm.

    np.random.seed(23324)

    # Works for larger k_var but slow for routine testing.
    for k_var in 5, 10, 25:
        for n_factor in 1, 2, 3:
            load = np.random.normal(size=(k_var, n_factor))
            uniq = np.linspace(1, 2, k_var)
            c = np.dot(load, load.T)
            c.flat[::c.shape[0]+1] += uniq
            s = np.sqrt(np.diag(c))
            c /= np.outer(s, s)
            fa = Factor(corr=c, n_factor=n_factor, method='ml')
            load_e, uniq_e = fa._fit_ml_em(2000)
            c_e = np.dot(load_e, load_e.T)
            c_e.flat[::c_e.shape[0]+1] += uniq_e
            assert_allclose(c_e, c, rtol=1e-4, atol=1e-4)


def test_fit_ml_em_random_state():
    # Ensure Factor._fit_ml_em doesn't change numpy's singleton random state
    # see #7357

    T = 10
    epsilon = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=T).T
    initial = np.random.get_state()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Fitting did not converge')
        Factor(endog=epsilon, n_factor=2, method='ml').fit()
    final = np.random.get_state()

    assert initial[0] == final[0]
    assert_equal(initial[1], final[1])
    assert initial[2:] == final[2:]


def test_em():

    n_factor = 1
    cor = np.asarray([[1, 0.5, 0.3], [0.5, 1, 0], [0.3, 0, 1]])

    fa = Factor(corr=cor, n_factor=n_factor, method='ml')
    rslt = fa.fit(opt={'gtol': 1e-3})
    load_opt = rslt.loadings
    uniq_opt = rslt.uniqueness

    load_em, uniq_em = fa._fit_ml_em(1000)
    cc = np.dot(load_em, load_em.T)
    cc.flat[::cc.shape[0]+1] += uniq_em

    assert_allclose(cc, rslt.fitted_cov, rtol=1e-2, atol=1e-2)


def test_1factor():
    """
    # R code:
    r = 0.4
    p = 4
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    fa = factanal(covmat=cm, factors=1)
    print(fa, digits=10)
    """

    r = 0.4
    p = 4
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))

    fa = Factor(corr=cm, n_factor=1, method='ml')
    rslt = fa.fit()

    if rslt.loadings[0, 0] < 0:
        rslt.loadings[:, 0] *= -1

    # R solution, but our likelihood is higher
    # uniq = np.r_[0.8392472054, 0.5820958187, 0.5820958187, 0.8392472054]
    # load = np.asarray([[0.4009399224, 0.6464550935, 0.6464550935,
    #                     0.4009399224]]).T
    # l1 = fa.loglike(fa._pack(load, uniq))
    # l2 = fa.loglike(fa._pack(rslt.loadings, rslt.uniqueness))

    # So use a smoke test
    uniq = np.r_[0.85290232,  0.60916033,  0.55382266,  0.82610666]
    load = np.asarray([[0.38353316], [0.62517171], [0.66796508],
                       [0.4170052]])

    assert_allclose(load, rslt.loadings, rtol=1e-3, atol=1e-3)
    assert_allclose(uniq, rslt.uniqueness, rtol=1e-3, atol=1e-3)

    assert_equal(rslt.df, 2)


def test_2factor():
    """
    # R code:
    r = 0.4
    p = 6
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    factanal(covmat=cm, factors=2)
    """

    r = 0.4
    p = 6
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))

    fa = Factor(corr=cm, n_factor=2, nobs=100, method='ml')
    rslt = fa.fit()

    for j in 0, 1:
        if rslt.loadings[0, j] < 0:
            rslt.loadings[:, j] *= -1

    uniq = np.r_[0.782, 0.367, 0.696, 0.696, 0.367, 0.782]
    assert_allclose(uniq, rslt.uniqueness, rtol=1e-3, atol=1e-3)

    loads = [np.r_[0.323, 0.586, 0.519, 0.519, 0.586, 0.323],
             np.r_[0.337, 0.538, 0.187, -0.187, -0.538, -0.337]]
    for k in 0, 1:
        if np.dot(loads[k], rslt.loadings[:, k]) < 0:
            loads[k] *= -1
        assert_allclose(loads[k], rslt.loadings[:, k], rtol=1e-3, atol=1e-3)

    assert_equal(rslt.df, 4)

    # Smoke test for standard errors
    e = np.asarray([0.11056836, 0.05191071, 0.09836349,
                    0.09836349, 0.05191071, 0.11056836])
    assert_allclose(rslt.uniq_stderr, e, atol=1e-4)
    e = np.asarray([[0.08842151, 0.08842151], [0.06058582, 0.06058582],
                    [0.08339874, 0.08339874], [0.08339874, 0.08339874],
                    [0.06058582, 0.06058582], [0.08842151, 0.08842151]])
    assert_allclose(rslt.load_stderr, e, atol=1e-4)
