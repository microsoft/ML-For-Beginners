"""
Tests for LinearModelLoss

Note that correctness of losses (which compose LinearModelLoss) is already well
covered in the _loss module.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize

from sklearn._loss.loss import (
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
)
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS

# We do not need to test all losses, just what LinearModelLoss does on top of the
# base losses.
LOSSES = [HalfBinomialLoss, HalfMultinomialLoss, HalfPoissonLoss]


def random_X_y_coef(
    linear_model_loss, n_samples, n_features, coef_bound=(-2, 2), seed=42
):
    """Random generate y, X and coef in valid range."""
    rng = np.random.RandomState(seed)
    n_dof = n_features + linear_model_loss.fit_intercept
    X = make_low_rank_matrix(
        n_samples=n_samples,
        n_features=n_features,
        random_state=rng,
    )
    coef = linear_model_loss.init_zero_coef(X)

    if linear_model_loss.base_loss.is_multiclass:
        n_classes = linear_model_loss.base_loss.n_classes
        coef.flat[:] = rng.uniform(
            low=coef_bound[0],
            high=coef_bound[1],
            size=n_classes * n_dof,
        )
        if linear_model_loss.fit_intercept:
            raw_prediction = X @ coef[:, :-1].T + coef[:, -1]
        else:
            raw_prediction = X @ coef.T
        proba = linear_model_loss.base_loss.link.inverse(raw_prediction)

        # y = rng.choice(np.arange(n_classes), p=proba) does not work.
        # See https://stackoverflow.com/a/34190035/16761084
        def choice_vectorized(items, p):
            s = p.cumsum(axis=1)
            r = rng.rand(p.shape[0])[:, None]
            k = (s < r).sum(axis=1)
            return items[k]

        y = choice_vectorized(np.arange(n_classes), p=proba).astype(np.float64)
    else:
        coef.flat[:] = rng.uniform(
            low=coef_bound[0],
            high=coef_bound[1],
            size=n_dof,
        )
        if linear_model_loss.fit_intercept:
            raw_prediction = X @ coef[:-1] + coef[-1]
        else:
            raw_prediction = X @ coef
        y = linear_model_loss.base_loss.link.inverse(
            raw_prediction + rng.uniform(low=-1, high=1, size=n_samples)
        )

    return X, y, coef


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("n_features", [0, 1, 10])
@pytest.mark.parametrize("dtype", [None, np.float32, np.float64, np.int64])
def test_init_zero_coef(base_loss, fit_intercept, n_features, dtype):
    """Test that init_zero_coef initializes coef correctly."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, n_features))
    coef = loss.init_zero_coef(X, dtype=dtype)
    if loss.base_loss.is_multiclass:
        n_classes = loss.base_loss.n_classes
        assert coef.shape == (n_classes, n_features + fit_intercept)
        assert coef.flags["F_CONTIGUOUS"]
    else:
        assert coef.shape == (n_features + fit_intercept,)

    if dtype is None:
        assert coef.dtype == X.dtype
    else:
        assert coef.dtype == dtype

    assert np.count_nonzero(coef) == 0


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_loss_grad_hess_are_the_same(
    base_loss, fit_intercept, sample_weight, l2_reg_strength, csr_container
):
    """Test that loss and gradient are the same across different functions."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=10, n_features=5, seed=42
    )

    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    l1 = loss.loss(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    g1 = loss.gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    l2, g2 = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    g3, h3 = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    if not base_loss.is_multiclass:
        g4, h4, _ = loss.gradient_hessian(
            coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
        )
    else:
        with pytest.raises(NotImplementedError):
            loss.gradient_hessian(
                coef,
                X,
                y,
                sample_weight=sample_weight,
                l2_reg_strength=l2_reg_strength,
            )

    assert_allclose(l1, l2)
    assert_allclose(g1, g2)
    assert_allclose(g1, g3)
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4)
        assert_allclose(h4 @ g4, h3(g3))

    # same for sparse X
    X = csr_container(X)
    l1_sp = loss.loss(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    g1_sp = loss.gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    l2_sp, g2_sp = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    g3_sp, h3_sp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    if not base_loss.is_multiclass:
        g4_sp, h4_sp, _ = loss.gradient_hessian(
            coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
        )

    assert_allclose(l1, l1_sp)
    assert_allclose(l1, l2_sp)
    assert_allclose(g1, g1_sp)
    assert_allclose(g1, g2_sp)
    assert_allclose(g1, g3_sp)
    assert_allclose(h3(g1), h3_sp(g1_sp))
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4_sp)
        assert_allclose(h4 @ g4, h4_sp @ g1_sp)


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
@pytest.mark.parametrize("X_container", CSR_CONTAINERS + [None])
def test_loss_gradients_hessp_intercept(
    base_loss, sample_weight, l2_reg_strength, X_container
):
    """Test that loss and gradient handle intercept correctly."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=False)
    loss_inter = LinearModelLoss(base_loss=base_loss(), fit_intercept=True)
    n_samples, n_features = 10, 5
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )

    X[:, -1] = 1  # make last column of 1 to mimic intercept term
    X_inter = X[
        :, :-1
    ]  # exclude intercept column as it is added automatically by loss_inter

    if X_container is not None:
        X = X_container(X)

    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    l, g = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    _, hessp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    l_inter, g_inter = loss_inter.loss_gradient(
        coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    _, hessp_inter = loss_inter.gradient_hessian_product(
        coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )

    # Note, that intercept gets no L2 penalty.
    assert l == pytest.approx(
        l_inter + 0.5 * l2_reg_strength * squared_norm(coef.T[-1])
    )

    g_inter_corrected = g_inter
    g_inter_corrected.T[-1] += l2_reg_strength * coef.T[-1]
    assert_allclose(g, g_inter_corrected)

    s = np.random.RandomState(42).randn(*coef.shape)
    h = hessp(s)
    h_inter = hessp_inter(s)
    h_inter_corrected = h_inter
    h_inter_corrected.T[-1] += l2_reg_strength * s.T[-1]
    assert_allclose(h, h_inter_corrected)


@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
def test_gradients_hessians_numerically(
    base_loss, fit_intercept, sample_weight, l2_reg_strength
):
    """Test gradients and hessians with numerical derivatives.

    Gradient should equal the numerical derivatives of the loss function.
    Hessians should equal the numerical derivatives of gradients.
    """
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    n_samples, n_features = 10, 5
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )
    coef = coef.ravel(order="F")  # this is important only for multinomial loss

    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    # 1. Check gradients numerically
    eps = 1e-6
    g, hessp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # Use a trick to get central finite difference of accuracy 4 (five-point stencil)
    # https://en.wikipedia.org/wiki/Numerical_differentiation
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # approx_g1 = (f(x + eps) - f(x - eps)) / (2*eps)
    approx_g1 = optimize.approx_fprime(
        coef,
        lambda coef: loss.loss(
            coef - eps,
            X,
            y,
            sample_weight=sample_weight,
            l2_reg_strength=l2_reg_strength,
        ),
        2 * eps,
    )
    # approx_g2 = (f(x + 2*eps) - f(x - 2*eps)) / (4*eps)
    approx_g2 = optimize.approx_fprime(
        coef,
        lambda coef: loss.loss(
            coef - 2 * eps,
            X,
            y,
            sample_weight=sample_weight,
            l2_reg_strength=l2_reg_strength,
        ),
        4 * eps,
    )
    # Five-point stencil approximation
    # See: https://en.wikipedia.org/wiki/Five-point_stencil#1D_first_derivative
    approx_g = (4 * approx_g1 - approx_g2) / 3
    assert_allclose(g, approx_g, rtol=1e-2, atol=1e-8)

    # 2. Check hessp numerically along the second direction of the gradient
    vector = np.zeros_like(g)
    vector[1] = 1
    hess_col = hessp(vector)
    # Computation of the Hessian is particularly fragile to numerical errors when doing
    # simple finite differences. Here we compute the grad along a path in the direction
    # of the vector and then use a least-square regression to estimate the slope
    eps = 1e-3
    d_x = np.linspace(-eps, eps, 30)
    d_grad = np.array(
        [
            loss.gradient(
                coef + t * vector,
                X,
                y,
                sample_weight=sample_weight,
                l2_reg_strength=l2_reg_strength,
            )
            for t in d_x
        ]
    )
    d_grad -= d_grad.mean(axis=0)
    approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()
    assert_allclose(approx_hess_col, hess_col, rtol=1e-3)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_multinomial_coef_shape(fit_intercept):
    """Test that multinomial LinearModelLoss respects shape of coef."""
    loss = LinearModelLoss(base_loss=HalfMultinomialLoss(), fit_intercept=fit_intercept)
    n_samples, n_features = 10, 5
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )
    s = np.random.RandomState(42).randn(*coef.shape)

    l, g = loss.loss_gradient(coef, X, y)
    g1 = loss.gradient(coef, X, y)
    g2, hessp = loss.gradient_hessian_product(coef, X, y)
    h = hessp(s)
    assert g.shape == coef.shape
    assert h.shape == coef.shape
    assert_allclose(g, g1)
    assert_allclose(g, g2)

    coef_r = coef.ravel(order="F")
    s_r = s.ravel(order="F")
    l_r, g_r = loss.loss_gradient(coef_r, X, y)
    g1_r = loss.gradient(coef_r, X, y)
    g2_r, hessp_r = loss.gradient_hessian_product(coef_r, X, y)
    h_r = hessp_r(s_r)
    assert g_r.shape == coef_r.shape
    assert h_r.shape == coef_r.shape
    assert_allclose(g_r, g1_r)
    assert_allclose(g_r, g2_r)

    assert_allclose(g, g_r.reshape(loss.base_loss.n_classes, -1, order="F"))
    assert_allclose(h, h_r.reshape(loss.base_loss.n_classes, -1, order="F"))
