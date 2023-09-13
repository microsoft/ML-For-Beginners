import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
    LinearConstraint,
    minimize,
    minimize_scalar,
    newton,
)
from scipy.special import logsumexp

from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
    _LOSSES,
    AbsoluteError,
    BaseLoss,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    HuberLoss,
    PinballLoss,
)
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit

ALL_LOSSES = list(_LOSSES.values())

LOSS_INSTANCES = [loss() for loss in ALL_LOSSES]
# HalfTweedieLoss(power=1.5) is already there as default
LOSS_INSTANCES += [
    PinballLoss(quantile=0.25),
    HuberLoss(quantile=0.75),
    HalfTweedieLoss(power=-1.5),
    HalfTweedieLoss(power=0),
    HalfTweedieLoss(power=1),
    HalfTweedieLoss(power=2),
    HalfTweedieLoss(power=3.0),
    HalfTweedieLossIdentity(power=0),
    HalfTweedieLossIdentity(power=1),
    HalfTweedieLossIdentity(power=2),
    HalfTweedieLossIdentity(power=3.0),
]


def loss_instance_name(param):
    if isinstance(param, BaseLoss):
        loss = param
        name = loss.__class__.__name__
        if isinstance(loss, PinballLoss):
            name += f"(quantile={loss.closs.quantile})"
        elif isinstance(loss, HuberLoss):
            name += f"(quantile={loss.quantile}"
        elif hasattr(loss, "closs") and hasattr(loss.closs, "power"):
            name += f"(power={loss.closs.power})"
        return name
    else:
        return str(param)


def random_y_true_raw_prediction(
    loss, n_samples, y_bound=(-100, 100), raw_bound=(-5, 5), seed=42
):
    """Random generate y_true and raw_prediction in valid range."""
    rng = np.random.RandomState(seed)
    if loss.is_multiclass:
        raw_prediction = np.empty((n_samples, loss.n_classes))
        raw_prediction.flat[:] = rng.uniform(
            low=raw_bound[0],
            high=raw_bound[1],
            size=n_samples * loss.n_classes,
        )
        y_true = np.arange(n_samples).astype(float) % loss.n_classes
    else:
        # If link is identity, we must respect the interval of y_pred:
        if isinstance(loss.link, IdentityLink):
            low, high = _inclusive_low_high(loss.interval_y_pred)
            low = np.amax([low, raw_bound[0]])
            high = np.amin([high, raw_bound[1]])
            raw_bound = (low, high)
        raw_prediction = rng.uniform(
            low=raw_bound[0], high=raw_bound[1], size=n_samples
        )
        # generate a y_true in valid range
        low, high = _inclusive_low_high(loss.interval_y_true)
        low = max(low, y_bound[0])
        high = min(high, y_bound[1])
        y_true = rng.uniform(low, high, size=n_samples)
        # set some values at special boundaries
        if loss.interval_y_true.low == 0 and loss.interval_y_true.low_inclusive:
            y_true[:: (n_samples // 3)] = 0
        if loss.interval_y_true.high == 1 and loss.interval_y_true.high_inclusive:
            y_true[1 :: (n_samples // 3)] = 1

    return y_true, raw_prediction


def numerical_derivative(func, x, eps):
    """Helper function for numerical (first) derivatives."""
    # For numerical derivatives, see
    # https://en.wikipedia.org/wiki/Numerical_differentiation
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # We use central finite differences of accuracy 4.
    h = np.full_like(x, fill_value=eps)
    f_minus_2h = func(x - 2 * h)
    f_minus_1h = func(x - h)
    f_plus_1h = func(x + h)
    f_plus_2h = func(x + 2 * h)
    return (-f_plus_2h + 8 * f_plus_1h - 8 * f_minus_1h + f_minus_2h) / (12.0 * eps)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_loss_boundary(loss):
    """Test interval ranges of y_true and y_pred in losses."""
    # make sure low and high are always within the interval, used for linspace
    if loss.is_multiclass:
        y_true = np.linspace(0, 9, num=10)
    else:
        low, high = _inclusive_low_high(loss.interval_y_true)
        y_true = np.linspace(low, high, num=10)

    # add boundaries if they are included
    if loss.interval_y_true.low_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.low]
    if loss.interval_y_true.high_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.high]

    assert loss.in_y_true_range(y_true)

    n = y_true.shape[0]
    low, high = _inclusive_low_high(loss.interval_y_pred)
    if loss.is_multiclass:
        y_pred = np.empty((n, 3))
        y_pred[:, 0] = np.linspace(low, high, num=n)
        y_pred[:, 1] = 0.5 * (1 - y_pred[:, 0])
        y_pred[:, 2] = 0.5 * (1 - y_pred[:, 0])
    else:
        y_pred = np.linspace(low, high, num=n)

    assert loss.in_y_pred_range(y_pred)

    # calculating losses should not fail
    raw_prediction = loss.link.link(y_pred)
    loss.loss(y_true=y_true, raw_prediction=raw_prediction)


# Fixture to test valid value ranges.
Y_COMMON_PARAMS = [
    # (loss, [y success], [y fail])
    (HalfSquaredError(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (AbsoluteError(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (PinballLoss(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (HuberLoss(), [-100, 0, 0.1, 100], [-np.inf, np.inf]),
    (HalfPoissonLoss(), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    (HalfGammaLoss(), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLoss(power=-3), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLoss(power=0), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLoss(power=1.5), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    (HalfTweedieLoss(power=2), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLoss(power=3), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLossIdentity(power=-3), [0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLossIdentity(power=0), [-3, -0.1, 0, 0.1, 100], [-np.inf, np.inf]),
    (HalfTweedieLossIdentity(power=1.5), [0.1, 100], [-np.inf, -3, -0.1, np.inf]),
    (HalfTweedieLossIdentity(power=2), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfTweedieLossIdentity(power=3), [0.1, 100], [-np.inf, -3, -0.1, 0, np.inf]),
    (HalfBinomialLoss(), [0.1, 0.5, 0.9], [-np.inf, -1, 2, np.inf]),
    (HalfMultinomialLoss(), [], [-np.inf, -1, 1.1, np.inf]),
]
# y_pred and y_true do not always have the same domain (valid value range).
# Hence, we define extra sets of parameters for each of them.
Y_TRUE_PARAMS = [  # type: ignore
    # (loss, [y success], [y fail])
    (HalfPoissonLoss(), [0], []),
    (HuberLoss(), [0], []),
    (HalfTweedieLoss(power=-3), [-100, -0.1, 0], []),
    (HalfTweedieLoss(power=0), [-100, 0], []),
    (HalfTweedieLoss(power=1.5), [0], []),
    (HalfTweedieLossIdentity(power=-3), [-100, -0.1, 0], []),
    (HalfTweedieLossIdentity(power=0), [-100, 0], []),
    (HalfTweedieLossIdentity(power=1.5), [0], []),
    (HalfBinomialLoss(), [0, 1], []),
    (HalfMultinomialLoss(), [0.0, 1.0, 2], []),
]
Y_PRED_PARAMS = [
    # (loss, [y success], [y fail])
    (HalfPoissonLoss(), [], [0]),
    (HalfTweedieLoss(power=-3), [], [-3, -0.1, 0]),
    (HalfTweedieLoss(power=0), [], [-3, -0.1, 0]),
    (HalfTweedieLoss(power=1.5), [], [0]),
    (HalfTweedieLossIdentity(power=-3), [], [-3, -0.1, 0]),
    (HalfTweedieLossIdentity(power=0), [-3, -0.1, 0], []),
    (HalfTweedieLossIdentity(power=1.5), [], [0]),
    (HalfBinomialLoss(), [], [0, 1]),
    (HalfMultinomialLoss(), [0.1, 0.5], [0, 1]),
]


@pytest.mark.parametrize(
    "loss, y_true_success, y_true_fail", Y_COMMON_PARAMS + Y_TRUE_PARAMS
)
def test_loss_boundary_y_true(loss, y_true_success, y_true_fail):
    """Test boundaries of y_true for loss functions."""
    for y in y_true_success:
        assert loss.in_y_true_range(np.array([y]))
    for y in y_true_fail:
        assert not loss.in_y_true_range(np.array([y]))


@pytest.mark.parametrize(
    "loss, y_pred_success, y_pred_fail", Y_COMMON_PARAMS + Y_PRED_PARAMS  # type: ignore
)
def test_loss_boundary_y_pred(loss, y_pred_success, y_pred_fail):
    """Test boundaries of y_pred for loss functions."""
    for y in y_pred_success:
        assert loss.in_y_pred_range(np.array([y]))
    for y in y_pred_fail:
        assert not loss.in_y_pred_range(np.array([y]))


@pytest.mark.parametrize(
    "loss, y_true, raw_prediction, loss_true",
    [
        (HalfSquaredError(), 1.0, 5.0, 8),
        (AbsoluteError(), 1.0, 5.0, 4),
        (PinballLoss(quantile=0.5), 1.0, 5.0, 2),
        (PinballLoss(quantile=0.25), 1.0, 5.0, 4 * (1 - 0.25)),
        (PinballLoss(quantile=0.25), 5.0, 1.0, 4 * 0.25),
        (HuberLoss(quantile=0.5, delta=3), 1.0, 5.0, 3 * (4 - 3 / 2)),
        (HuberLoss(quantile=0.5, delta=3), 1.0, 3.0, 0.5 * 2**2),
        (HalfPoissonLoss(), 2.0, np.log(4), 4 - 2 * np.log(4)),
        (HalfGammaLoss(), 2.0, np.log(4), np.log(4) + 2 / 4),
        (HalfTweedieLoss(power=3), 2.0, np.log(4), -1 / 4 + 1 / 4**2),
        (HalfTweedieLossIdentity(power=1), 2.0, 4.0, 2 - 2 * np.log(2)),
        (HalfTweedieLossIdentity(power=2), 2.0, 4.0, np.log(2) - 1 / 2),
        (HalfTweedieLossIdentity(power=3), 2.0, 4.0, -1 / 4 + 1 / 4**2 + 1 / 2 / 2),
        (HalfBinomialLoss(), 0.25, np.log(4), np.log(5) - 0.25 * np.log(4)),
        (
            HalfMultinomialLoss(n_classes=3),
            0.0,
            [0.2, 0.5, 0.3],
            logsumexp([0.2, 0.5, 0.3]) - 0.2,
        ),
        (
            HalfMultinomialLoss(n_classes=3),
            1.0,
            [0.2, 0.5, 0.3],
            logsumexp([0.2, 0.5, 0.3]) - 0.5,
        ),
        (
            HalfMultinomialLoss(n_classes=3),
            2.0,
            [0.2, 0.5, 0.3],
            logsumexp([0.2, 0.5, 0.3]) - 0.3,
        ),
    ],
    ids=loss_instance_name,
)
def test_loss_on_specific_values(loss, y_true, raw_prediction, loss_true):
    """Test losses at specific values."""
    assert loss(
        y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction])
    ) == approx(loss_true, rel=1e-11, abs=1e-12)


@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize("readonly_memmap", [False, True])
@pytest.mark.parametrize("dtype_in", [np.float32, np.float64])
@pytest.mark.parametrize("dtype_out", [np.float32, np.float64])
@pytest.mark.parametrize("sample_weight", [None, 1])
@pytest.mark.parametrize("out1", [None, 1])
@pytest.mark.parametrize("out2", [None, 1])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_loss_dtype(
    loss, readonly_memmap, dtype_in, dtype_out, sample_weight, out1, out2, n_threads
):
    """Test acceptance of dtypes, readonly and writeable arrays in loss functions.

    Check that loss accepts if all input arrays are either all float32 or all
    float64, and all output arrays are either all float32 or all float64.

    Also check that input arrays can be readonly, e.g. memory mapped.
    """
    loss = loss()
    # generate a y_true and raw_prediction in valid range
    n_samples = 5
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )
    y_true = y_true.astype(dtype_in)
    raw_prediction = raw_prediction.astype(dtype_in)

    if sample_weight is not None:
        sample_weight = np.array([2.0] * n_samples, dtype=dtype_in)
    if out1 is not None:
        out1 = np.empty_like(y_true, dtype=dtype_out)
    if out2 is not None:
        out2 = np.empty_like(raw_prediction, dtype=dtype_out)

    if readonly_memmap:
        y_true = create_memmap_backed_data(y_true, aligned=True)
        raw_prediction = create_memmap_backed_data(raw_prediction, aligned=True)
        if sample_weight is not None:
            sample_weight = create_memmap_backed_data(sample_weight, aligned=True)

    loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out1,
        n_threads=n_threads,
    )
    loss.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out2,
        n_threads=n_threads,
    )
    loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out1,
        gradient_out=out2,
        n_threads=n_threads,
    )
    if out1 is not None and loss.is_multiclass:
        out1 = np.empty_like(raw_prediction, dtype=dtype_out)
    loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out1,
        hessian_out=out2,
        n_threads=n_threads,
    )
    loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)
    loss.constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
    if hasattr(loss, "predict_proba"):
        loss.predict_proba(raw_prediction=raw_prediction)
    if hasattr(loss, "gradient_proba"):
        loss.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out1,
            proba_out=out2,
            n_threads=n_threads,
        )


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_loss_same_as_C_functions(loss, sample_weight):
    """Test that Python and Cython functions return same results."""
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )
    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    out_l1 = np.empty_like(y_true)
    out_l2 = np.empty_like(y_true)
    out_g1 = np.empty_like(raw_prediction)
    out_g2 = np.empty_like(raw_prediction)
    out_h1 = np.empty_like(raw_prediction)
    out_h2 = np.empty_like(raw_prediction)
    assert_allclose(
        loss.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=out_l1,
        ),
        loss.closs.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=out_l2,
        ),
    )
    assert_allclose(
        loss.gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out_g1,
        ),
        loss.closs.gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out_g2,
        ),
    )
    loss.closs.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l1,
        gradient_out=out_g1,
    )
    loss.closs.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l2,
        gradient_out=out_g2,
    )
    assert_allclose(out_l1, out_l2)
    assert_allclose(out_g1, out_g2)
    loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g1,
        hessian_out=out_h1,
    )
    loss.closs.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g2,
        hessian_out=out_h2,
    )
    assert_allclose(out_g1, out_g2)
    assert_allclose(out_h1, out_h2)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_loss_gradients_are_the_same(loss, sample_weight, global_random_seed):
    """Test that loss and gradient are the same across different functions.

    Also test that output arguments contain correct results.
    """
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=global_random_seed,
    )
    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    out_l1 = np.empty_like(y_true)
    out_l2 = np.empty_like(y_true)
    out_g1 = np.empty_like(raw_prediction)
    out_g2 = np.empty_like(raw_prediction)
    out_g3 = np.empty_like(raw_prediction)
    out_h3 = np.empty_like(raw_prediction)

    l1 = loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l1,
    )
    g1 = loss.gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g1,
    )
    l2, g2 = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        loss_out=out_l2,
        gradient_out=out_g2,
    )
    g3, h3 = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
        gradient_out=out_g3,
        hessian_out=out_h3,
    )
    assert_allclose(l1, l2)
    assert_array_equal(l1, out_l1)
    assert np.shares_memory(l1, out_l1)
    assert_array_equal(l2, out_l2)
    assert np.shares_memory(l2, out_l2)
    assert_allclose(g1, g2)
    assert_allclose(g1, g3)
    assert_array_equal(g1, out_g1)
    assert np.shares_memory(g1, out_g1)
    assert_array_equal(g2, out_g2)
    assert np.shares_memory(g2, out_g2)
    assert_array_equal(g3, out_g3)
    assert np.shares_memory(g3, out_g3)

    if hasattr(loss, "gradient_proba"):
        assert loss.is_multiclass  # only for HalfMultinomialLoss
        out_g4 = np.empty_like(raw_prediction)
        out_proba = np.empty_like(raw_prediction)
        g4, proba = loss.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=out_g4,
            proba_out=out_proba,
        )
        assert_allclose(g1, out_g4)
        assert_allclose(g1, g4)
        assert_allclose(proba, out_proba)
        assert_allclose(np.sum(proba, axis=1), 1, rtol=1e-11)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", ["ones", "random"])
def test_sample_weight_multiplies(loss, sample_weight, global_random_seed):
    """Test sample weights in loss, gradients and hessians.

    Make sure that passing sample weights to loss, gradient and hessian
    computation methods is equivalent to multiplying by the weights.
    """
    n_samples = 100
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    if sample_weight == "ones":
        sample_weight = np.ones(shape=n_samples, dtype=np.float64)
    else:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = rng.normal(size=n_samples).astype(np.float64)

    assert_allclose(
        loss.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        ),
        sample_weight
        * loss.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=None,
        ),
    )

    losses, gradient = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=None,
    )
    losses_sw, gradient_sw = loss.loss_gradient(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    assert_allclose(losses * sample_weight, losses_sw)
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)

    gradient, hessian = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=None,
    )
    gradient_sw, hessian_sw = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    if not loss.is_multiclass:
        assert_allclose(gradient * sample_weight, gradient_sw)
        assert_allclose(hessian * sample_weight, hessian_sw)
    else:
        assert_allclose(gradient * sample_weight[:, None], gradient_sw)
        assert_allclose(hessian * sample_weight[:, None], hessian_sw)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_graceful_squeezing(loss):
    """Test that reshaped raw_prediction gives same results."""
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=20,
        y_bound=(-100, 100),
        raw_bound=(-10, 10),
        seed=42,
    )

    if raw_prediction.ndim == 1:
        raw_prediction_2d = raw_prediction[:, None]
        assert_allclose(
            loss.loss(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.loss(y_true=y_true, raw_prediction=raw_prediction),
        )
        assert_allclose(
            loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction),
        )
        assert_allclose(
            loss.gradient(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.gradient(y_true=y_true, raw_prediction=raw_prediction),
        )
        assert_allclose(
            loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction_2d),
            loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction),
        )


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_loss_of_perfect_prediction(loss, sample_weight):
    """Test value of perfect predictions.

    Loss of y_pred = y_true plus constant_to_optimal_zero should sums up to
    zero.
    """
    if not loss.is_multiclass:
        # Use small values such that exp(value) is not nan.
        raw_prediction = np.array([-10, -0.1, 0, 0.1, 3, 10])
        # If link is identity, we must respect the interval of y_pred:
        if isinstance(loss.link, IdentityLink):
            eps = 1e-10
            low = loss.interval_y_pred.low
            if not loss.interval_y_pred.low_inclusive:
                low = low + eps
            high = loss.interval_y_pred.high
            if not loss.interval_y_pred.high_inclusive:
                high = high - eps
            raw_prediction = np.clip(raw_prediction, low, high)
        y_true = loss.link.inverse(raw_prediction)
    else:
        # HalfMultinomialLoss
        y_true = np.arange(loss.n_classes).astype(float)
        # raw_prediction with entries -exp(10), but +exp(10) on the diagonal
        # this is close enough to np.inf which would produce nan
        raw_prediction = np.full(
            shape=(loss.n_classes, loss.n_classes),
            fill_value=-np.exp(10),
            dtype=float,
        )
        raw_prediction.flat[:: loss.n_classes + 1] = np.exp(10)

    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    loss_value = loss.loss(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )
    constant_term = loss.constant_to_optimal_zero(
        y_true=y_true, sample_weight=sample_weight
    )
    # Comparing loss_value + constant_term to zero would result in large
    # round-off errors.
    assert_allclose(loss_value, -constant_term, atol=1e-14, rtol=1e-15)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_gradients_hessians_numerically(loss, sample_weight, global_random_seed):
    """Test gradients and hessians with numerical derivatives.

    Gradient should equal the numerical derivatives of the loss function.
    Hessians should equal the numerical derivatives of gradients.
    """
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    if sample_weight == "range":
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])

    g, h = loss.gradient_hessian(
        y_true=y_true,
        raw_prediction=raw_prediction,
        sample_weight=sample_weight,
    )

    assert g.shape == raw_prediction.shape
    assert h.shape == raw_prediction.shape

    if not loss.is_multiclass:

        def loss_func(x):
            return loss.loss(
                y_true=y_true,
                raw_prediction=x,
                sample_weight=sample_weight,
            )

        g_numeric = numerical_derivative(loss_func, raw_prediction, eps=1e-6)
        assert_allclose(g, g_numeric, rtol=5e-6, atol=1e-10)

        def grad_func(x):
            return loss.gradient(
                y_true=y_true,
                raw_prediction=x,
                sample_weight=sample_weight,
            )

        h_numeric = numerical_derivative(grad_func, raw_prediction, eps=1e-6)
        if loss.approx_hessian:
            # TODO: What could we test if loss.approx_hessian?
            pass
        else:
            assert_allclose(h, h_numeric, rtol=5e-6, atol=1e-10)
    else:
        # For multiclass loss, we should only change the predictions of the
        # class for which the derivative is taken for, e.g. offset[:, k] = eps
        # for class k.
        # As a softmax is computed, offsetting the whole array by a constant
        # would have no effect on the probabilities, and thus on the loss.
        for k in range(loss.n_classes):

            def loss_func(x):
                raw = raw_prediction.copy()
                raw[:, k] = x
                return loss.loss(
                    y_true=y_true,
                    raw_prediction=raw,
                    sample_weight=sample_weight,
                )

            g_numeric = numerical_derivative(loss_func, raw_prediction[:, k], eps=1e-5)
            assert_allclose(g[:, k], g_numeric, rtol=5e-6, atol=1e-10)

            def grad_func(x):
                raw = raw_prediction.copy()
                raw[:, k] = x
                return loss.gradient(
                    y_true=y_true,
                    raw_prediction=raw,
                    sample_weight=sample_weight,
                )[:, k]

            h_numeric = numerical_derivative(grad_func, raw_prediction[:, k], eps=1e-6)
            if loss.approx_hessian:
                # TODO: What could we test if loss.approx_hessian?
                pass
            else:
                assert_allclose(h[:, k], h_numeric, rtol=5e-6, atol=1e-10)


@pytest.mark.parametrize(
    "loss, x0, y_true",
    [
        ("squared_error", -2.0, 42),
        ("squared_error", 117.0, 1.05),
        ("squared_error", 0.0, 0.0),
        # The argmin of binomial_loss for y_true=0 and y_true=1 is resp.
        # -inf and +inf due to logit, cf. "complete separation". Therefore, we
        # use 0 < y_true < 1.
        ("binomial_loss", 0.3, 0.1),
        ("binomial_loss", -12, 0.2),
        ("binomial_loss", 30, 0.9),
        ("poisson_loss", 12.0, 1.0),
        ("poisson_loss", 0.0, 2.0),
        ("poisson_loss", -22.0, 10.0),
    ],
)
@skip_if_32bit
def test_derivatives(loss, x0, y_true):
    """Test that gradients are zero at the minimum of the loss.

    We check this on a single value/sample using Halley's method with the
    first and second order derivatives computed by the Loss instance.
    Note that methods of Loss instances operate on arrays while the newton
    root finder expects a scalar or a one-element array for this purpose.
    """
    loss = _LOSSES[loss](sample_weight=None)
    y_true = np.array([y_true], dtype=np.float64)
    x0 = np.array([x0], dtype=np.float64)

    def func(x: np.ndarray) -> np.ndarray:
        """Compute loss plus constant term.

        The constant term is such that the minimum function value is zero,
        which is required by the Newton method.
        """
        return loss.loss(
            y_true=y_true, raw_prediction=x
        ) + loss.constant_to_optimal_zero(y_true=y_true)

    def fprime(x: np.ndarray) -> np.ndarray:
        return loss.gradient(y_true=y_true, raw_prediction=x)

    def fprime2(x: np.ndarray) -> np.ndarray:
        return loss.gradient_hessian(y_true=y_true, raw_prediction=x)[1]

    optimum = newton(
        func,
        x0=x0,
        fprime=fprime,
        fprime2=fprime2,
        maxiter=100,
        tol=5e-8,
    )

    # Need to ravel arrays because assert_allclose requires matching
    # dimensions.
    y_true = y_true.ravel()
    optimum = optimum.ravel()
    assert_allclose(loss.link.inverse(optimum), y_true)
    assert_allclose(func(optimum), 0, atol=1e-14)
    assert_allclose(loss.gradient(y_true=y_true, raw_prediction=optimum), 0, atol=5e-7)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
@pytest.mark.parametrize("sample_weight", [None, "range"])
def test_loss_intercept_only(loss, sample_weight):
    """Test that fit_intercept_only returns the argmin of the loss.

    Also test that the gradient is zero at the minimum.
    """
    n_samples = 50
    if not loss.is_multiclass:
        y_true = loss.link.inverse(np.linspace(-4, 4, num=n_samples))
    else:
        y_true = np.arange(n_samples).astype(np.float64) % loss.n_classes
        y_true[::5] = 0  # exceedance of class 0

    if sample_weight == "range":
        sample_weight = np.linspace(0.1, 2, num=n_samples)

    a = loss.fit_intercept_only(y_true=y_true, sample_weight=sample_weight)

    # find minimum by optimization
    def fun(x):
        if not loss.is_multiclass:
            raw_prediction = np.full(shape=(n_samples), fill_value=x)
        else:
            raw_prediction = np.ascontiguousarray(
                np.broadcast_to(x, shape=(n_samples, loss.n_classes))
            )
        return loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
        )

    if not loss.is_multiclass:
        opt = minimize_scalar(fun, tol=1e-7, options={"maxiter": 100})
        grad = loss.gradient(
            y_true=y_true,
            raw_prediction=np.full_like(y_true, a),
            sample_weight=sample_weight,
        )
        assert a.shape == tuple()  # scalar
        assert a.dtype == y_true.dtype
        assert_all_finite(a)
        a == approx(opt.x, rel=1e-7)
        grad.sum() == approx(0, abs=1e-12)
    else:
        # The constraint corresponds to sum(raw_prediction) = 0. Without it, we would
        # need to apply loss.symmetrize_raw_prediction to opt.x before comparing.
        opt = minimize(
            fun,
            np.zeros((loss.n_classes)),
            tol=1e-13,
            options={"maxiter": 100},
            method="SLSQP",
            constraints=LinearConstraint(np.ones((1, loss.n_classes)), 0, 0),
        )
        grad = loss.gradient(
            y_true=y_true,
            raw_prediction=np.tile(a, (n_samples, 1)),
            sample_weight=sample_weight,
        )
        assert a.dtype == y_true.dtype
        assert_all_finite(a)
        assert_allclose(a, opt.x, rtol=5e-6, atol=1e-12)
        assert_allclose(grad.sum(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize(
    "loss, func, random_dist",
    [
        (HalfSquaredError(), np.mean, "normal"),
        (AbsoluteError(), np.median, "normal"),
        (PinballLoss(quantile=0.25), lambda x: np.percentile(x, q=25), "normal"),
        (HalfPoissonLoss(), np.mean, "poisson"),
        (HalfGammaLoss(), np.mean, "exponential"),
        (HalfTweedieLoss(), np.mean, "exponential"),
        (HalfBinomialLoss(), np.mean, "binomial"),
    ],
)
def test_specific_fit_intercept_only(loss, func, random_dist, global_random_seed):
    """Test that fit_intercept_only returns the correct functional.

    We test the functional for specific, meaningful distributions, e.g.
    squared error estimates the expectation of a probability distribution.
    """
    rng = np.random.RandomState(global_random_seed)
    if random_dist == "binomial":
        y_train = rng.binomial(1, 0.5, size=100)
    else:
        y_train = getattr(rng, random_dist)(size=100)
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)
    # Make sure baseline prediction is the expected functional=func, e.g. mean
    # or median.
    assert_all_finite(baseline_prediction)
    assert baseline_prediction == approx(loss.link.link(func(y_train)))
    assert loss.link.inverse(baseline_prediction) == approx(func(y_train))
    if isinstance(loss, IdentityLink):
        assert_allclose(loss.link.inverse(baseline_prediction), baseline_prediction)

    # Test baseline at boundary
    if loss.interval_y_true.low_inclusive:
        y_train.fill(loss.interval_y_true.low)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert_all_finite(baseline_prediction)
    if loss.interval_y_true.high_inclusive:
        y_train.fill(loss.interval_y_true.high)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert_all_finite(baseline_prediction)


def test_multinomial_loss_fit_intercept_only():
    """Test that fit_intercept_only returns the mean functional for CCE."""
    rng = np.random.RandomState(0)
    n_classes = 4
    loss = HalfMultinomialLoss(n_classes=n_classes)
    # Same logic as test_specific_fit_intercept_only. Here inverse link
    # function = softmax and link function = log - symmetry term.
    y_train = rng.randint(0, n_classes + 1, size=100).astype(np.float64)
    baseline_prediction = loss.fit_intercept_only(y_true=y_train)
    assert baseline_prediction.shape == (n_classes,)
    p = np.zeros(n_classes, dtype=y_train.dtype)
    for k in range(n_classes):
        p[k] = (y_train == k).mean()
    assert_allclose(baseline_prediction, np.log(p) - np.mean(np.log(p)))
    assert_allclose(baseline_prediction[None, :], loss.link.link(p[None, :]))

    for y_train in (np.zeros(shape=10), np.ones(shape=10)):
        y_train = y_train.astype(np.float64)
        baseline_prediction = loss.fit_intercept_only(y_true=y_train)
        assert baseline_prediction.dtype == y_train.dtype
        assert_all_finite(baseline_prediction)


def test_binomial_and_multinomial_loss(global_random_seed):
    """Test that multinomial loss with n_classes = 2 is the same as binomial loss."""
    rng = np.random.RandomState(global_random_seed)
    n_samples = 20
    binom = HalfBinomialLoss()
    multinom = HalfMultinomialLoss(n_classes=2)
    y_train = rng.randint(0, 2, size=n_samples).astype(np.float64)
    raw_prediction = rng.normal(size=n_samples)
    raw_multinom = np.empty((n_samples, 2))
    raw_multinom[:, 0] = -0.5 * raw_prediction
    raw_multinom[:, 1] = 0.5 * raw_prediction
    assert_allclose(
        binom.loss(y_true=y_train, raw_prediction=raw_prediction),
        multinom.loss(y_true=y_train, raw_prediction=raw_multinom),
    )


@pytest.mark.parametrize("y_true", (np.array([0.0, 0, 0]), np.array([1.0, 1, 1])))
@pytest.mark.parametrize("y_pred", (np.array([-5.0, -5, -5]), np.array([3.0, 3, 3])))
def test_binomial_vs_alternative_formulation(y_true, y_pred, global_dtype):
    """Test that both formulations of the binomial deviance agree.

    Often, the binomial deviance or log loss is written in terms of a variable
    z in {-1, +1}, but we use y in {0, 1}, hence z = 2 * y - 1.
    ESL II Eq. (10.18):

        -loglike(z, f) = log(1 + exp(-2 * z * f))

    Note:
        - ESL 2*f = raw_prediction, hence the factor 2 of ESL disappears.
        - Deviance = -2*loglike + .., but HalfBinomialLoss is half of the
          deviance, hence the factor of 2 cancels in the comparison.
    """

    def alt_loss(y, raw_pred):
        z = 2 * y - 1
        return np.mean(np.log(1 + np.exp(-z * raw_pred)))

    def alt_gradient(y, raw_pred):
        # alternative gradient formula according to ESL
        z = 2 * y - 1
        return -z / (1 + np.exp(z * raw_pred))

    bin_loss = HalfBinomialLoss()

    y_true = y_true.astype(global_dtype)
    y_pred = y_pred.astype(global_dtype)
    datum = (y_true, y_pred)

    assert bin_loss(*datum) == approx(alt_loss(*datum))
    assert_allclose(bin_loss.gradient(*datum), alt_gradient(*datum))


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_predict_proba(loss, global_random_seed):
    """Test that predict_proba and gradient_proba work as expected."""
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=global_random_seed,
    )

    if hasattr(loss, "predict_proba"):
        proba = loss.predict_proba(raw_prediction)
        assert proba.shape == (n_samples, loss.n_classes)
        assert np.sum(proba, axis=1) == approx(1, rel=1e-11)

    if hasattr(loss, "gradient_proba"):
        for grad, proba in (
            (None, None),
            (None, np.empty_like(raw_prediction)),
            (np.empty_like(raw_prediction), None),
            (np.empty_like(raw_prediction), np.empty_like(raw_prediction)),
        ):
            grad, proba = loss.gradient_proba(
                y_true=y_true,
                raw_prediction=raw_prediction,
                sample_weight=None,
                gradient_out=grad,
                proba_out=proba,
            )
            assert proba.shape == (n_samples, loss.n_classes)
            assert np.sum(proba, axis=1) == approx(1, rel=1e-11)
            assert_allclose(
                grad,
                loss.gradient(
                    y_true=y_true,
                    raw_prediction=raw_prediction,
                    sample_weight=None,
                    gradient_out=None,
                ),
            )


@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("order", ("C", "F"))
def test_init_gradient_and_hessians(loss, sample_weight, dtype, order):
    """Test that init_gradient_and_hessian works as expected.

    passing sample_weight to a loss correctly influences the constant_hessian
    attribute, and consequently the shape of the hessian array.
    """
    n_samples = 5
    if sample_weight == "range":
        sample_weight = np.ones(n_samples)
    loss = loss(sample_weight=sample_weight)
    gradient, hessian = loss.init_gradient_and_hessian(
        n_samples=n_samples,
        dtype=dtype,
        order=order,
    )
    if loss.constant_hessian:
        assert gradient.shape == (n_samples,)
        assert hessian.shape == (1,)
    elif loss.is_multiclass:
        assert gradient.shape == (n_samples, loss.n_classes)
        assert hessian.shape == (n_samples, loss.n_classes)
    else:
        assert hessian.shape == (n_samples,)
        assert hessian.shape == (n_samples,)

    assert gradient.dtype == dtype
    assert hessian.dtype == dtype

    if order == "C":
        assert gradient.flags.c_contiguous
        assert hessian.flags.c_contiguous
    else:
        assert gradient.flags.f_contiguous
        assert hessian.flags.f_contiguous


@pytest.mark.parametrize("loss", ALL_LOSSES)
@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"dtype": np.int64},
            f"Valid options for 'dtype' are .* Got dtype={np.int64} instead.",
        ),
    ],
)
def test_init_gradient_and_hessian_raises(loss, params, err_msg):
    """Test that init_gradient_and_hessian raises errors for invalid input."""
    loss = loss()
    with pytest.raises((ValueError, TypeError), match=err_msg):
        gradient, hessian = loss.init_gradient_and_hessian(n_samples=5, **params)


@pytest.mark.parametrize(
    "loss, params, err_type, err_msg",
    [
        (
            PinballLoss,
            {"quantile": None},
            TypeError,
            "quantile must be an instance of float, not NoneType.",
        ),
        (
            PinballLoss,
            {"quantile": 0},
            ValueError,
            "quantile == 0, must be > 0.",
        ),
        (PinballLoss, {"quantile": 1.1}, ValueError, "quantile == 1.1, must be < 1."),
        (
            HuberLoss,
            {"quantile": None},
            TypeError,
            "quantile must be an instance of float, not NoneType.",
        ),
        (
            HuberLoss,
            {"quantile": 0},
            ValueError,
            "quantile == 0, must be > 0.",
        ),
        (HuberLoss, {"quantile": 1.1}, ValueError, "quantile == 1.1, must be < 1."),
    ],
)
def test_loss_init_parameter_validation(loss, params, err_type, err_msg):
    """Test that loss raises errors for invalid input."""
    with pytest.raises(err_type, match=err_msg):
        loss(**params)


@pytest.mark.parametrize("loss", LOSS_INSTANCES, ids=loss_instance_name)
def test_loss_pickle(loss):
    """Test that losses can be pickled."""
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=loss,
        n_samples=n_samples,
        y_bound=(-100, 100),
        raw_bound=(-5, 5),
        seed=42,
    )
    pickled_loss = pickle.dumps(loss)
    unpickled_loss = pickle.loads(pickled_loss)
    assert loss(y_true=y_true, raw_prediction=raw_prediction) == approx(
        unpickled_loss(y_true=y_true, raw_prediction=raw_prediction)
    )


@pytest.mark.parametrize("p", [-1.5, 0, 1, 1.5, 2, 3])
def test_tweedie_log_identity_consistency(p):
    """Test for identical losses when only the link function is different."""
    half_tweedie_log = HalfTweedieLoss(power=p)
    half_tweedie_identity = HalfTweedieLossIdentity(power=p)
    n_samples = 10
    y_true, raw_prediction = random_y_true_raw_prediction(
        loss=half_tweedie_log, n_samples=n_samples, seed=42
    )
    y_pred = half_tweedie_log.link.inverse(raw_prediction)  # exp(raw_prediction)

    # Let's compare the loss values, up to some constant term that is dropped
    # in HalfTweedieLoss but not in HalfTweedieLossIdentity.
    loss_log = half_tweedie_log.loss(
        y_true=y_true, raw_prediction=raw_prediction
    ) + half_tweedie_log.constant_to_optimal_zero(y_true)
    loss_identity = half_tweedie_identity.loss(
        y_true=y_true, raw_prediction=y_pred
    ) + half_tweedie_identity.constant_to_optimal_zero(y_true)
    # Note that HalfTweedieLoss ignores different constant terms than
    # HalfTweedieLossIdentity. Constant terms means terms not depending on
    # raw_prediction. By adding these terms, `constant_to_optimal_zero`, both losses
    # give the same values.
    assert_allclose(loss_log, loss_identity)

    # For gradients and hessians, the constant terms do not matter. We have, however,
    # to account for the chain rule, i.e. with x=raw_prediction
    #     gradient_log(x) = d/dx loss_log(x)
    #                     = d/dx loss_identity(exp(x))
    #                     = exp(x) * gradient_identity(exp(x))
    # Similarly,
    #     hessian_log(x) = exp(x) * gradient_identity(exp(x))
    #                    + exp(x)**2 * hessian_identity(x)
    gradient_log, hessian_log = half_tweedie_log.gradient_hessian(
        y_true=y_true, raw_prediction=raw_prediction
    )
    gradient_identity, hessian_identity = half_tweedie_identity.gradient_hessian(
        y_true=y_true, raw_prediction=y_pred
    )
    assert_allclose(gradient_log, y_pred * gradient_identity)
    assert_allclose(
        hessian_log, y_pred * gradient_identity + y_pred**2 * hessian_identity
    )
