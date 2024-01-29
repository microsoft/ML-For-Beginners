# License: BSD 3 clause

import inspect

import numpy as np
import pytest

from sklearn.base import is_classifier
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PoissonRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TweedieRegressor,
)


# Note: GammaRegressor() and TweedieRegressor(power != 1) have a non-canonical link.
@pytest.mark.parametrize(
    "model",
    [
        ARDRegression(),
        BayesianRidge(),
        ElasticNet(),
        ElasticNetCV(),
        Lars(),
        LarsCV(),
        Lasso(),
        LassoCV(),
        LassoLarsCV(),
        LassoLarsIC(),
        LinearRegression(),
        # TODO: FIx SAGA which fails badly with sample_weights.
        # This is a known limitation, see:
        # https://github.com/scikit-learn/scikit-learn/issues/21305
        pytest.param(
            LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, tol=1e-15
            ),
            marks=pytest.mark.xfail(reason="Missing importance sampling scheme"),
        ),
        LogisticRegressionCV(tol=1e-6),
        MultiTaskElasticNet(),
        MultiTaskElasticNetCV(),
        MultiTaskLasso(),
        MultiTaskLassoCV(),
        OrthogonalMatchingPursuit(),
        OrthogonalMatchingPursuitCV(),
        PoissonRegressor(),
        Ridge(),
        RidgeCV(),
        pytest.param(
            SGDRegressor(tol=1e-15),
            marks=pytest.mark.xfail(reason="Insufficient precision."),
        ),
        SGDRegressor(penalty="elasticnet", max_iter=10_000),
        TweedieRegressor(power=0),  # same as Ridge
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("with_sample_weight", [False, True])
def test_balance_property(model, with_sample_weight, global_random_seed):
    # Test that sum(y_predicted) == sum(y_observed) on the training set.
    # This must hold for all linear models with deviance of an exponential disperson
    # family as loss and the corresponding canonical link if fit_intercept=True.
    # Examples:
    #     - squared error and identity link (most linear models)
    #     - Poisson deviance with log link
    #     - log loss with logit link
    # This is known as balance property or unconditional calibration/unbiasedness.
    # For reference, see Corollary 3.18, 3.20 and Chapter 5.1.5 of
    # M.V. Wuthrich and M. Merz, "Statistical Foundations of Actuarial Learning and its
    # Applications" (June 3, 2022). http://doi.org/10.2139/ssrn.3822407

    if (
        with_sample_weight
        and "sample_weight" not in inspect.signature(model.fit).parameters.keys()
    ):
        pytest.skip("Estimator does not support sample_weight.")

    rel = 2e-4  # test precision
    if isinstance(model, SGDRegressor):
        rel = 1e-1
    elif hasattr(model, "solver") and model.solver == "saga":
        rel = 1e-2

    rng = np.random.RandomState(global_random_seed)
    n_train, n_features, n_targets = 100, 10, None
    if isinstance(
        model,
        (MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV),
    ):
        n_targets = 3
    X = make_low_rank_matrix(n_samples=n_train, n_features=n_features, random_state=rng)
    if n_targets:
        coef = (
            rng.uniform(low=-2, high=2, size=(n_features, n_targets))
            / np.max(X, axis=0)[:, None]
        )
    else:
        coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)

    expectation = np.exp(X @ coef + 0.5)
    y = rng.poisson(lam=expectation) + 1  # strict positive, i.e. y > 0
    if is_classifier(model):
        y = (y > expectation + 1).astype(np.float64)

    if with_sample_weight:
        sw = rng.uniform(low=1, high=10, size=y.shape[0])
    else:
        sw = None

    model.set_params(fit_intercept=True)  # to be sure
    if with_sample_weight:
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)

    # Assert balance property.
    if is_classifier(model):
        assert np.average(model.predict_proba(X)[:, 1], weights=sw) == pytest.approx(
            np.average(y, weights=sw), rel=rel
        )
    else:
        assert np.average(model.predict(X), weights=sw, axis=0) == pytest.approx(
            np.average(y, weights=sw, axis=0), rel=rel
        )
