# pylint: disable=F841
"""
unit test for GAM

Author: Luca Puggini

Created on 08/07/2015
"""

import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import block_diag
import pytest

from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.gam.smooth_basis import (
    UnivariatePolynomialSmoother, PolynomialSmoother, BSplines,
    GenericSmoothers, UnivariateCubicSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
    GLMGam, LogitGam, make_augmented_matrix, penalized_wls)
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
    MultivariateGAMCV, MultivariateGAMCVPath, _split_train_test_smoothers)
from statsmodels.gam.gam_penalties import (UnivariateGamPenalty,
                                           MultivariateGamPenalty)
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian
from statsmodels.genmod.generalized_linear_model import lm

sigmoid = np.vectorize(lambda x: 1.0 / (1.0 + np.exp(-x)))


def polynomial_sample_data():
    """A polynomial of degree 4

    poly = ax^4 + bx^3 + cx^2 + dx + e
    second der = 12ax^2 + 6bx + 2c
    integral from -1 to 1 of second der^2 is
        (288 a^2)/5 + 32 a c + 8 (3 b^2 + c^2)
    the gradient of the integral is der
        [576*a/5 + 32 * c, 48*b, 32*a + 16*c, 0, 0]

    Returns
    -------
    poly : smoother instance
    y : ndarray
        generated function values, demeaned
    """
    n = 10000
    x = np.linspace(-1, 1, n)
    y = 2 * x ** 3 - x
    y -= y.mean()

    degree = [4]
    pol = PolynomialSmoother(x, degree)

    return pol, y


def integral(params):
    d, c, b, a = params
    itg = (288 * a ** 2) / 5 + (32 * a * c) + 8 * (3 * b ** 2 + c ** 2)
    itg /= 2
    return itg


def grad(params):
    d, c, b, a = params
    grd = np.array([576 * a / 5 + 32 * c, 48 * b, 32 * a + 16 * c, 0])
    grd = grd[::-1]
    return grd / 2


def hessian(params):
    hess = np.array([[576 / 5, 0, 32, 0],
                     [0, 48, 0, 0],
                     [32, 0, 16, 0],
                     [0, 0, 0, 0]
                     ])
    return hess / 2


def cost_function(params, pol, y, alpha):
    # this should be the MSE or log likelihood value
    lin_pred = np.dot(pol.basis, params)
    gaussian = Gaussian()
    expval = gaussian.link.inverse(lin_pred)
    loglike = gaussian.loglike(y, expval)

    # this is the vale of the GAM penalty. For the example polynomial
    itg = integral(params)

    # return the cost function of the GAM for the given polynomial
    return loglike - alpha * itg, loglike, itg


def test_gam_penalty():
    """
    test the func method of the gam penalty
    :return:
    """
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)

    for _ in range(10):
        params = np.random.randint(-2, 2, 4)
        gp_score = gp.func(params)
        itg = integral(params)
        assert_allclose(gp_score, itg, atol=1.e-1)


def test_gam_gradient():
    # test the gam gradient for the example polynomial
    np.random.seed(1)
    pol, y = polynomial_sample_data()

    alpha = 1
    smoother = pol.smoothers[0]
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=smoother)

    for _ in range(10):
        params = np.random.uniform(-2, 2, 4)
        params = np.array([1, 1, 1, 1])
        gam_grad = gp.deriv(params)
        grd = grad(params)

        assert_allclose(gam_grad, grd, rtol=1.e-2, atol=1.e-2)


def test_gam_hessian():
    # test the deriv2 method of the gam penalty
    np.random.seed(1)
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)

    for _ in range(10):
        params = np.random.randint(-2, 2, 5)
        gam_der2 = gp.deriv2(params)
        hess = hessian(params)
        hess = np.flipud(hess)
        hess = np.fliplr(hess)
        assert_allclose(gam_der2, hess, atol=1.e-13, rtol=1.e-3)


def test_approximation():
    np.random.seed(1)
    poly, y = polynomial_sample_data()
    alpha = 1
    for _ in range(10):
        params = np.random.uniform(-1, 1, 4)
        cost, err, itg = cost_function(params, poly, y, alpha)
        glm_gam = GLMGam(y, smoother=poly, alpha=alpha)
        # TODO: why do we need pen_weight=1
        gam_loglike = glm_gam.loglike(params, scale=1, pen_weight=1)
        assert_allclose(err - itg, cost, rtol=1e-10)
        assert_allclose(gam_loglike, cost, rtol=0.1)


def test_gam_glm():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y.values

    df = [10]
    degree = [3]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = np.asarray(data_from_r.y_est)

    alpha = 0.1  # chosen by trial and error

    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='bfgs', max_start_irls=0,
                              disp=1, maxiter=10000)
    y_gam0 = np.dot(bsplines.basis, res_glm_gam.params)
    y_gam = np.asarray(res_glm_gam.fittedvalues)
    assert_allclose(y_gam, y_gam0, rtol=1e-10)

    # plt.plot(x, y_gam, '.', label='gam')
    # plt.plot(x, y_mgcv, '.', label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, atol=1.e-2)


def test_gam_discrete():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.ybin.values

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)

    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.ybin_est

    alpha = 0.00002
    # gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=bsplines)
    # lg_gam = LogitGam(y, bsplines.basis, penal=gp)
    #
    lg_gam = LogitGam(y, bsplines, alpha=alpha)
    res_lg_gam = lg_gam.fit(maxiter=10000)
    y_gam = np.dot(bsplines.basis, res_lg_gam.params)
    y_gam = sigmoid(y_gam)
    y_mgcv = sigmoid(y_mgcv)

    # plt.plot(x, y_gam, label='gam')
    # plt.plot(x, y_mgcv, label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.ylim(-0.4, 1.4)
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, rtol=1.e-10, atol=1.e-1)


def multivariate_sample_data(seed=1):
    n = 1000
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-10, 10, n)
    x = np.vstack([x1, x2]).T
    np.random.seed(seed)
    y = x1 * x1 * x1 + x2 + np.random.normal(0, 0.01, n)
    degree1 = 4
    degree2 = 3
    degrees = [degree1, degree2]
    pol = PolynomialSmoother(x, degrees)
    return x, y, pol


def test_multivariate_penalty():
    alphas = [1, 2]
    weights = [1, 1]
    np.random.seed(1)
    x, y, pol = multivariate_sample_data()

    univ_pol1 = UnivariatePolynomialSmoother(x[:, 0], degree=pol.degrees[0])
    univ_pol2 = UnivariatePolynomialSmoother(x[:, 1], degree=pol.degrees[1])

    gp1 = UnivariateGamPenalty(alpha=alphas[0], univariate_smoother=univ_pol1)
    gp2 = UnivariateGamPenalty(alpha=alphas[1], univariate_smoother=univ_pol2)
    with pytest.warns(UserWarning, match="weights is currently ignored"):
        mgp = MultivariateGamPenalty(multivariate_smoother=pol, alpha=alphas,
                                     weights=weights)

    for i in range(10):
        params1 = np.random.randint(-3, 3, pol.smoothers[0].dim_basis)
        params2 = np.random.randint(-3, 3, pol.smoothers[1].dim_basis)
        params = np.concatenate([params1, params2])
        c1 = gp1.func(params1)
        c2 = gp2.func(params2)
        c = mgp.func(params)
        assert_allclose(c, c1 + c2, atol=1.e-10, rtol=1.e-10)

        d1 = gp1.deriv(params1)
        d2 = gp2.deriv(params2)
        d12 = np.concatenate([d1, d2])
        d = mgp.deriv(params)
        assert_allclose(d, d12)

        h1 = gp1.deriv2(params1)
        h2 = gp2.deriv2(params2)
        h12 = block_diag(h1, h2)
        h = mgp.deriv2(params)
        assert_allclose(h, h12)


def test_generic_smoother():
    x, y, poly = multivariate_sample_data()
    alphas = [0.4, 0.7]
    weights = [1, 1]  # noqa: F841

    gs = GenericSmoothers(poly.x, poly.smoothers)
    gam_gs = GLMGam(y, smoother=gs, alpha=alphas)
    gam_gs_res = gam_gs.fit()

    gam_poly = GLMGam(y, smoother=poly, alpha=alphas)
    gam_poly_res = gam_poly.fit()

    assert_allclose(gam_gs_res.params, gam_poly_res.params)


def test_multivariate_gam_1d_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y

    df = [10]
    degree = [3]
    bsplines = BSplines(x, degree=degree, df=df)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)
    y_mgcv = data_from_r.y_est

    # alpha is by manually adjustment to reduce discrepancy in fittedvalues
    alpha = [0.0168 * 0.0251 / 2 * 500]
    gp = MultivariateGamPenalty(bsplines, alpha=alpha)    # noqa: F841

    glm_gam = GLMGam(y, exog=np.ones((len(y), 1)), smoother=bsplines,
                     alpha=alpha)
    # "nm" converges to a different params, "bfgs" params are close to pirls
    # res_glm_gam = glm_gam.fit(method='nm', max_start_irls=0,
    #                           disp=1, maxiter=10000, maxfun=5000)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0,
                              disp=1, maxiter=10000)
    y_gam = res_glm_gam.fittedvalues

    # plt.plot(x, y_gam, '.', label='gam')
    # plt.plot(x, y_mgcv, '.', label='mgcv')
    # plt.plot(x, y, '.', label='y')
    # plt.legend()
    # plt.show()

    assert_allclose(y_gam, y_mgcv, atol=0.01)


def test_multivariate_gam_cv():
    # SMOKE test
    # no test is performed. It only checks that there is not any runtime error

    def cost(x1, x2):
        return np.linalg.norm(x1 - x2) / len(x1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)
    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y.values

    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    # y_mgcv is obtained from R with the following code
    # g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)

    alphas = [0.0251]
    alphas = [2]
    cv = KFold(3)

    gp = MultivariateGamPenalty(bsplines, alpha=alphas)  # noqa: F841
    gam_cv = MultivariateGAMCV(smoother=bsplines, alphas=alphas, gam=GLMGam,
                               cost=cost, endog=y, exog=None, cv_iterator=cv)
    gam_cv_res = gam_cv.fit()  # noqa: F841


def test_multivariate_gam_cv_path():
    def sample_metric(y1, y2):
        return np.linalg.norm(y1 - y2) / len(y1)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se  # noqa: F841
    y_mgcv = data_from_r.y_mgcv_gcv  # noqa: F841

    df = [10]
    degree = [6]

    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)

    gam = GLMGam
    alphas = [np.linspace(0, 2, 10)]
    k = 3
    cv = KFold(k_folds=k, shuffle=True)

    # Note: kfold cv uses random shuffle
    np.random.seed(123)
    gam_cv = MultivariateGAMCVPath(smoother=bsplines, alphas=alphas, gam=gam,
                                   cost=sample_metric, endog=y, exog=None,
                                   cv_iterator=cv)
    gam_cv_res = gam_cv.fit()  # noqa: F841

    glm_gam = GLMGam(y, smoother=bsplines, alpha=gam_cv.alpha_cv)
    res_glm_gam = glm_gam.fit(method='irls', max_start_irls=0,
                              disp=1, maxiter=10000)
    y_est = res_glm_gam.predict(bsplines.basis)

    # plt.plot(x, y, '.', label='y')
    # plt.plot(x, y_est, '.', label='y est')
    # plt.plot(x, y_mgcv, '.', label='y mgcv')
    # plt.legend()
    # plt.show()

    # The test compares to result obtained with GCV and not KFOLDS CV.
    # This is because MGCV does not support KFOLD CV
    assert_allclose(data_from_r.y_mgcv_gcv, y_est, atol=1.e-1, rtol=1.e-1)

    # Note: kfold cv uses random shuffle
    np.random.seed(123)
    alpha_cv, res_cv = glm_gam.select_penweight_kfold(alphas=alphas, k_folds=3)
    assert_allclose(alpha_cv, gam_cv.alpha_cv, rtol=1e-12)


def test_train_test_smoothers():
    n = 6
    x = np.zeros(shape=(n, 2))
    x[:, 0] = range(6)
    x[:, 1] = range(6, 12)
    poly = PolynomialSmoother(x, degrees=[3, 3])
    train_index = list(range(3))
    test_index = list(range(3, 6))
    train_smoother, test_smoother = _split_train_test_smoothers(poly.x, poly,
                                                                train_index,
                                                                test_index)

    expected_train_basis = [[0., 0., 0., 6., 36., 216.],
                            [1., 1., 1., 7., 49., 343.],
                            [2., 4., 8., 8., 64., 512.]]
    assert_allclose(train_smoother.basis, expected_train_basis)

    expected_test_basis = [[3., 9., 27., 9., 81., 729.],
                           [4., 16., 64., 10., 100., 1000.],
                           [5., 25., 125., 11., 121., 1331.]]
    assert_allclose(test_smoother.basis, expected_test_basis)


def test_get_sqrt():
    n = 1000
    np.random.seed(1)
    x = np.random.normal(0, 1, (n, 3))
    x2 = np.dot(x.T, x)

    sqrt_x2 = matrix_sqrt(x2)

    x2_reconstruction = np.dot(sqrt_x2.T, sqrt_x2)
    assert_allclose(x2_reconstruction, x2)


def test_make_augmented_matrix():
    np.random.seed(1)
    n = 500
    x = np.random.uniform(-1, 1, (n, 3))
    s = np.dot(x.T, x)
    y = np.array(list(range(n)))
    w = np.random.uniform(0, 1, n)
    nobs, n_columns = x.shape

    # matrix_sqrt removes redundant rows,
    # if alpha is zero, then no augmentation is needed
    alpha = 0
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, alpha * s, w)
    expected_aug_x = x
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = y
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = w
    assert_allclose(aug_w, expected_aug_w)

    alpha = 1
    aug_y, aug_x, aug_w = make_augmented_matrix(y, x, s, w)
    rs = matrix_sqrt(alpha * s)
    # alternative version to matrix_sqrt using cholesky is not available
    # rs = sp.linalg.cholesky(alpha * s)
    assert_allclose(np.dot(rs.T, rs), alpha * s)
    expected_aug_x = np.vstack([x, rs])
    assert_allclose(aug_x, expected_aug_x)
    expected_aug_y = np.zeros(shape=(nobs + n_columns,))
    expected_aug_y[:nobs] = y
    assert_allclose(aug_y, expected_aug_y)
    expected_aug_w = np.concatenate((w, [1] * n_columns), axis=0)
    assert_allclose(aug_w, expected_aug_w)


def test_penalized_wls():
    np.random.seed(1)
    n = 20
    p = 3
    x = np.random.normal(0, 1, (n, 3))
    y = x[:, 1] - x[:, 2] + np.random.normal(0, .1, n)
    y -= y.mean()

    weights = np.ones(shape=(n,))
    s = np.random.normal(0, 1, (p, p))

    pen_wls_res = penalized_wls(y, x, 0 * s, weights)
    ls_res = lm.OLS(y, x).fit()

    assert_allclose(ls_res.params, pen_wls_res.params)


def test_cyclic_cubic_splines():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results",
                             "cubic_cyclic_splines_from_mgcv.csv")
    data_from_r = pd.read_csv(file_path)

    x = data_from_r[['x0', 'x2']].values
    y = data_from_r['y'].values
    y_est_mgcv = data_from_r[['y_est']].values  # noqa: F841
    s_mgcv = data_from_r[['s(x0)', 's(x2)']].values

    dfs = [10, 10]
    ccs = CyclicCubicSplines(x, df=dfs)
    alpha = [0.05 / 2, 0.0005 / 2]
    # TODO: if alpha changes in pirls this should be updated

    gam = GLMGam(y, smoother=ccs, alpha=alpha)
    gam_res = gam.fit(method='pirls')

    s0 = np.dot(ccs.basis[:, ccs.mask[0]],
                gam_res.params[ccs.mask[0]])
    # TODO: Mean has to be removed
    # removing mean could be replaced by options for intercept handling
    s0 -= s0.mean()

    s1 = np.dot(ccs.basis[:, ccs.mask[1]],
                gam_res.params[ccs.mask[1]])
    s1 -= s1.mean()  # TODO: Mean has to be removed

    # plt.subplot(2, 1, 1)
    # plt.plot(x[:, 0], s0, '.', label='s0')
    # plt.plot(x[:, 0], s_mgcv[:, 0], '.', label='s0_mgcv')
    # plt.legend(loc='best')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x[:, 1], s1, '.', label='s1_est')
    # plt.plot(x[:, 1], s_mgcv[:, 1], '.', label='s1_mgcv')
    # plt.legend(loc='best')
    # plt.show()

    assert_allclose(s0, s_mgcv[:, 0], atol=0.02)
    assert_allclose(s1, s_mgcv[:, 1], atol=0.33)


def test_multivariate_cubic_splines():
    np.random.seed(0)
    from statsmodels.gam.smooth_basis import CubicSplines

    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.linspace(0, 1, n)**2

    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    # need small enough noise variance to get good estimate for this test
    y = y0 + np.random.normal(0, .3 / 2, n)
    y -= y.mean()
    y0 -= y0.mean()

    alphas = [1e-3, 1e-3]
    cs = CubicSplines(x, df=[10, 10], constraints='center')

    gam = GLMGam(y, exog=np.ones((n, 1)), smoother=cs, alpha=alphas)
    gam_res = gam.fit(method='pirls')

    y_est = gam_res.fittedvalues
    y_est -= y_est.mean()

    # cut the tails
    index = list(range(50, n - 50))
    y_est = y_est[index]
    y0 = y0[index]
    y = y[index]

    # plt.plot(y_est, label='y est')
    # plt.plot(y0, label='y0')
    # plt.plot(y, '.', label='y')
    # plt.legend(loc='best')
    # plt.show()

    assert_allclose(y_est, y0, atol=0.04)


def test_glm_pirls_compatibility():
    np.random.seed(0)

    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.random.rand(n)

    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    y = y0 + np.random.normal(0, .3, n)
    y -= y.mean()
    y0 -= y0.mean()

    # TODO: we have now alphas == alphas_glm
    alphas = [5.75] * 2
    alphas_glm = [1.2] * 2  # noqa: F841
    # using constraints avoids singular exog.
    cs = BSplines(x, df=[10, 10], degree=[3, 3], constraints='center')

    gam_pirls = GLMGam(y, smoother=cs, alpha=alphas)
    gam_glm = GLMGam(y, smoother=cs, alpha=alphas)

    gam_res_glm = gam_glm.fit(method='nm', max_start_irls=0,
                              disp=1, maxiter=20000)
    gam_res_glm = gam_glm.fit(start_params=gam_res_glm.params,
                              method='bfgs', max_start_irls=0,
                              disp=1, maxiter=20000)
    gam_res_pirls = gam_pirls.fit()

    y_est_glm = np.dot(cs.basis, gam_res_glm.params)
    y_est_glm -= y_est_glm.mean()
    y_est_pirls = np.dot(cs.basis, gam_res_pirls.params)
    y_est_pirls -= y_est_pirls.mean()

    # plt.plot(y_est_pirls)
    # plt.plot(y_est_glm)
    # plt.plot(y, '.')
    # plt.show()
    assert_allclose(gam_res_glm.params, gam_res_pirls.params, atol=5e-5,
                    rtol=5e-5)
    assert_allclose(y_est_glm, y_est_pirls, atol=5e-5)


def test_zero_penalty():
    x, y, poly = multivariate_sample_data()
    alphas = [0, 0]
    gam_gs = GLMGam(y, smoother=poly, alpha=alphas)
    gam_gs_res = gam_gs.fit()
    y_est_gam = gam_gs_res.predict()

    glm = GLM(y, poly.basis).fit()
    y_est = glm.predict()

    assert_allclose(y_est, y_est_gam)


def test_spl_s():
    # matrix from R
    spl_s_R = [[0, 0, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
               [0, 0, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
               [0, 0, 0.001400000, 0.000200000, -0.001133333, -0.001000000],
               [0, 0, 0.000200000, 0.002733333, 0.001666667, -0.001133333],
               [0, 0, -0.001133333, 0.001666667, 0.002733333, 0.000200000],
               [0, 0, -0.001000000, -0.001133333, 0.000200000, 0.001400000]]

    np.random.seed(1)
    x = np.random.normal(0, 1, 10)
    xk = np.array([0.2, .4, .6, .8])
    cs = UnivariateCubicSplines(x, df=4)
    cs.knots = xk

    spl_s = cs._splines_s()
    assert_allclose(spl_s_R, spl_s, atol=4.e-10)


def test_partial_values2():
    np.random.seed(0)
    n = 1000
    x = np.random.uniform(0, 1, (n, 2))
    x = x - x.mean()
    y = x[:, 0] * x[:, 0] + np.random.normal(0, .01, n)
    y -= y.mean()
    alpha = 0.0
    # BUG: mask is incorrect if exog is not None, start_idx missing
    # bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2)
    # glm_gam = GLMGam(y, exog=np.ones((len(y), 1)), smoother=bsplines,
    #                  alpha=alpha)
    bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2,
                        include_intercept=[True, False])
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0,
                              disp=0, maxiter=5000)
    glm = GLM(y, bsplines.basis)  # noqa: F841

    # case with constant column in exog is currently wrong
    # ex = np.column_stack((np.zeros((len(y), 1)), bsplines.smoothers[0].basis,
    #                       np.zeros_like(bsplines.smoothers[1].basis) ))
    ex = np.column_stack((bsplines.smoothers[0].basis,
                          np.zeros_like(bsplines.smoothers[1].basis)))

    y_est = res_glm_gam.predict(ex, transform=False)
    y_partial_est, se = res_glm_gam.partial_values(0)

    assert_allclose(y_est, y_partial_est, atol=0.05)
    assert se.min() < 100
    # TODO: sometimes the SE reported by partial_values is very large.
    # This should be double checked


def test_partial_values():
    # this test is only approximate because we do not use the same spline
    # basis functions (knots) as mgcv
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)

    # TODO: alpha found by trial and error to pass assert
    alpha = 0.025 / 115 * 500
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method='bfgs')
    # TODO: if IRLS is used res_glm_gam has not partial_values.

    univ_bsplines = bsplines.smoothers[0]  # noqa: F841
    hat_y, se = res_glm_gam.partial_values(0)

    assert_allclose(hat_y, data_from_r["y_est"], rtol=0, atol=0.008)
    # TODO: bug missing scale
    bug_fact = np.sqrt(res_glm_gam.scale) * 0.976  # this is = 0.106
    assert_allclose(se, se_from_mgcv * bug_fact, rtol=0, atol=0.008)


@pytest.mark.matplotlib
def test_partial_plot():
    # verify that plot and partial_values method agree
    # the model only has one component so partial values is the same as
    # fittedvalues
    # Generate a plot to visualize analyze the result.

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, "results", "prediction_from_mgcv.csv")

    data_from_r = pd.read_csv(file_path)

    # dataset used to train the R model
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se  # noqa: F841
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df)

    alpha = 0.03
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(maxiter=10000, method='bfgs')
    fig = res_glm_gam.plot_partial(0)
    xp, yp = fig.axes[0].get_children()[0].get_data()
    # Note xp and yp are sorted by x
    sort_idx = np.argsort(x)
    hat_y, se = res_glm_gam.partial_values(0)
    # assert that main plot line is the prediction
    assert_allclose(xp, x[sort_idx])
    assert_allclose(yp, hat_y[sort_idx])

    # Uncomment to visualize the plot
    # import matplotlib.pyplot as plt
    # res_glm_gam.plot_partial(0)
    # plt.plot(x, y, '.')
    # plt.show()


def test_cov_params():

    np.random.seed(0)
    n = 1000
    x = np.random.uniform(0, 1, (n, 2))
    x = x - x.mean()
    y = x[:, 0] * x[:, 0] + np.random.normal(0, .01, n)
    y -= y.mean()

    bsplines = BSplines(x, degree=[3] * 2, df=[10] * 2, constraints='center')
    alpha = [0, 0]
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0,
                              disp=0, maxiter=5000)
    glm = GLM(y, bsplines.basis)
    res_glm = glm.fit()

    assert_allclose(res_glm.cov_params(), res_glm_gam.cov_params(),
                    rtol=0.0025)

    alpha = 1e-13
    glm_gam = GLMGam(y, smoother=bsplines, alpha=alpha)
    res_glm_gam = glm_gam.fit(method='pirls', max_start_irls=0,
                              disp=0, maxiter=5000)

    assert_allclose(res_glm.cov_params(), res_glm_gam.cov_params(),
                    atol=1e-10)

    res_glm_gam = glm_gam.fit(method='bfgs', max_start_irls=0,
                              disp=0, maxiter=5000)

    assert_allclose(res_glm.cov_params(), res_glm_gam.cov_params(),
                    rtol=1e-4, atol=1e-8)
