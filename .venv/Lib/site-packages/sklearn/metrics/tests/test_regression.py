from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy

from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)


def test_regression_metrics(n_samples=50):
    y_true = np.arange(n_samples)
    y_pred = y_true + 1
    y_pred_2 = y_true - 1

    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.0)
    assert_almost_equal(
        mean_squared_log_error(y_true, y_pred),
        mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)),
    )
    assert_almost_equal(mean_absolute_error(y_true, y_pred), 1.0)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred), 0.5)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2), 0.5)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred, alpha=0.4), 0.6)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2, alpha=0.4), 0.4)
    assert_almost_equal(median_absolute_error(y_true, y_pred), 1.0)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(mape)
    assert mape > 1e6
    assert_almost_equal(max_error(y_true, y_pred), 1.0)
    assert_almost_equal(r2_score(y_true, y_pred), 0.995, 2)
    assert_almost_equal(r2_score(y_true, y_pred, force_finite=False), 0.995, 2)
    assert_almost_equal(explained_variance_score(y_true, y_pred), 1.0)
    assert_almost_equal(
        explained_variance_score(y_true, y_pred, force_finite=False), 1.0
    )
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=0),
        mean_squared_error(y_true, y_pred),
    )
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=0), r2_score(y_true, y_pred)
    )
    dev_median = np.abs(y_true - np.median(y_true)).sum()
    assert_array_almost_equal(
        d2_absolute_error_score(y_true, y_pred),
        1 - np.abs(y_true - y_pred).sum() / dev_median,
    )
    alpha = 0.2
    pinball_loss = lambda y_true, y_pred, alpha: alpha * np.maximum(
        y_true - y_pred, 0
    ) + (1 - alpha) * np.maximum(y_pred - y_true, 0)
    y_quantile = np.percentile(y_true, q=alpha * 100)
    assert_almost_equal(
        d2_pinball_score(y_true, y_pred, alpha=alpha),
        1
        - pinball_loss(y_true, y_pred, alpha).sum()
        / pinball_loss(y_true, y_quantile, alpha).sum(),
    )
    assert_almost_equal(
        d2_absolute_error_score(y_true, y_pred),
        d2_pinball_score(y_true, y_pred, alpha=0.5),
    )

    # Tweedie deviance needs positive y_pred, except for p=0,
    # p>=2 needs positive y_true
    # results evaluated by sympy
    y_true = np.arange(1, 1 + n_samples)
    y_pred = 2 * y_true
    n = n_samples
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=-1),
        5 / 12 * n * (n**2 + 2 * n + 1),
    )
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=1), (n + 1) * (1 - np.log(2))
    )
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=2), 2 * np.log(2) - 1
    )
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=3 / 2),
        ((6 * np.sqrt(2) - 8) / n) * np.sqrt(y_true).sum(),
    )
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=3), np.sum(1 / y_true) / (4 * n)
    )

    dev_mean = 2 * np.mean(xlogy(y_true, 2 * y_true / (n + 1)))
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=1),
        1 - (n + 1) * (1 - np.log(2)) / dev_mean,
    )

    dev_mean = 2 * np.log((n + 1) / 2) - 2 / n * np.log(factorial(n))
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=2), 1 - (2 * np.log(2) - 1) / dev_mean
    )


def test_root_mean_squared_error_multioutput_raw_value():
    # non-regression test for
    # https://github.com/scikit-learn/scikit-learn/pull/16323
    mse = mean_squared_error([[1]], [[10]], multioutput="raw_values")
    rmse = root_mean_squared_error([[1]], [[10]], multioutput="raw_values")
    assert np.sqrt(mse) == pytest.approx(rmse)


def test_multioutput_regression():
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    error = mean_squared_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 / 3 + 2.0 / 3 + 2.0 / 3) / 4.0)

    error = root_mean_squared_error(y_true, y_pred)
    assert_almost_equal(error, 0.454, decimal=2)

    error = mean_squared_log_error(y_true, y_pred)
    assert_almost_equal(error, 0.200, decimal=2)

    error = root_mean_squared_log_error(y_true, y_pred)
    assert_almost_equal(error, 0.315, decimal=2)

    # mean_absolute_error and mean_squared_error are equal because
    # it is a binary problem.
    error = mean_absolute_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 2.0 / 3) / 4.0)

    error = mean_pinball_loss(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 2.0 / 3) / 8.0)

    error = np.around(mean_absolute_percentage_error(y_true, y_pred), decimals=2)
    assert np.isfinite(error)
    assert error > 1e6
    error = median_absolute_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 1.0) / 4.0)

    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    assert_almost_equal(error, 1.0 - 5.0 / 2)
    error = r2_score(y_true, y_pred, multioutput="uniform_average")
    assert_almost_equal(error, -0.875)

    score = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")
    raw_expected_score = [
        1
        - np.abs(y_true[:, i] - y_pred[:, i]).sum()
        / np.abs(y_true[:, i] - np.median(y_true[:, i])).sum()
        for i in range(y_true.shape[1])
    ]
    # in the last case, the denominator vanishes and hence we get nan,
    # but since the numerator vanishes as well the expected score is 1.0
    raw_expected_score = np.where(np.isnan(raw_expected_score), 1, raw_expected_score)
    assert_array_almost_equal(score, raw_expected_score)

    score = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="uniform_average")
    assert_almost_equal(score, raw_expected_score.mean())
    # constant `y_true` with force_finite=True leads to 1. or 0.
    yc = [5.0, 5.0]
    error = r2_score(yc, [5.0, 5.0], multioutput="variance_weighted")
    assert_almost_equal(error, 1.0)
    error = r2_score(yc, [5.0, 5.1], multioutput="variance_weighted")
    assert_almost_equal(error, 0.0)

    # Setting force_finite=False results in the nan for 4th output propagating
    error = r2_score(
        y_true, y_pred, multioutput="variance_weighted", force_finite=False
    )
    assert_almost_equal(error, np.nan)
    error = r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False)
    assert_almost_equal(error, np.nan)

    # Dropping the 4th output to check `force_finite=False` for nominal
    y_true = y_true[:, :-1]
    y_pred = y_pred[:, :-1]
    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    error2 = r2_score(
        y_true, y_pred, multioutput="variance_weighted", force_finite=False
    )
    assert_almost_equal(error, error2)
    error = r2_score(y_true, y_pred, multioutput="uniform_average")
    error2 = r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False)
    assert_almost_equal(error, error2)

    # constant `y_true` with force_finite=False leads to NaN or -Inf.
    error = r2_score(
        yc, [5.0, 5.0], multioutput="variance_weighted", force_finite=False
    )
    assert_almost_equal(error, np.nan)
    error = r2_score(
        yc, [5.0, 6.0], multioutput="variance_weighted", force_finite=False
    )
    assert_almost_equal(error, -np.inf)


def test_regression_metrics_at_limits():
    # Single-sample case
    # Note: for r2 and d2_tweedie see also test_regression_single_sample
    assert_almost_equal(mean_squared_error([0.0], [0.0]), 0.0)
    assert_almost_equal(root_mean_squared_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_squared_log_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_absolute_error([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_pinball_loss([0.0], [0.0]), 0.0)
    assert_almost_equal(mean_absolute_percentage_error([0.0], [0.0]), 0.0)
    assert_almost_equal(median_absolute_error([0.0], [0.0]), 0.0)
    assert_almost_equal(max_error([0.0], [0.0]), 0.0)
    assert_almost_equal(explained_variance_score([0.0], [0.0]), 1.0)

    # Perfect cases
    assert_almost_equal(r2_score([0.0, 1], [0.0, 1]), 1.0)
    assert_almost_equal(d2_pinball_score([0.0, 1], [0.0, 1]), 1.0)

    # Non-finite cases
    # R² and explained variance have a fix by default for non-finite cases
    for s in (r2_score, explained_variance_score):
        assert_almost_equal(s([0, 0], [1, -1]), 0.0)
        assert_almost_equal(s([0, 0], [1, -1], force_finite=False), -np.inf)
        assert_almost_equal(s([1, 1], [1, 1]), 1.0)
        assert_almost_equal(s([1, 1], [1, 1], force_finite=False), np.nan)
    msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([-1.0], [-1.0])
    msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])
    msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])
    msg = (
        "Root Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        root_mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])

    # Tweedie deviance error
    power = -1.2
    assert_allclose(
        mean_tweedie_deviance([0], [1.0], power=power), 2 / (2 - power), rtol=1e-3
    )
    msg = "can only be used on strictly positive y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)

    assert_almost_equal(mean_tweedie_deviance([0.0], [0.0], power=0), 0.0, 2)

    power = 1.0
    msg = "only be used on non-negative y and strictly positive y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)

    power = 1.5
    assert_allclose(mean_tweedie_deviance([0.0], [1.0], power=power), 2 / (2 - power))
    msg = "only be used on non-negative y and strictly positive y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)

    power = 2.0
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.00, atol=1e-8)
    msg = "can only be used on strictly positive y and y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)

    power = 3.0
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.00, atol=1e-8)
    msg = "can only be used on strictly positive y and y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)


def test__check_reg_targets():
    # All of length 3
    EXAMPLES = [
        ("continuous", [1, 2, 3], 1),
        ("continuous", [[1], [2], [3]], 1),
        ("continuous-multioutput", [[1, 1], [2, 2], [3, 1]], 2),
        ("continuous-multioutput", [[5, 1], [4, 2], [3, 1]], 2),
        ("continuous-multioutput", [[1, 3, 4], [2, 2, 2], [3, 1, 1]], 3),
    ]

    for (type1, y1, n_out1), (type2, y2, n_out2) in product(EXAMPLES, repeat=2):
        if type1 == type2 and n_out1 == n_out2:
            y_type, y_check1, y_check2, multioutput = _check_reg_targets(y1, y2, None)
            assert type1 == y_type
            if type1 == "continuous":
                assert_array_equal(y_check1, np.reshape(y1, (-1, 1)))
                assert_array_equal(y_check2, np.reshape(y2, (-1, 1)))
            else:
                assert_array_equal(y_check1, y1)
                assert_array_equal(y_check2, y2)
        else:
            with pytest.raises(ValueError):
                _check_reg_targets(y1, y2, None)


def test__check_reg_targets_exception():
    invalid_multioutput = "this_value_is_not_valid"
    expected_message = (
        "Allowed 'multioutput' string values are.+You provided multioutput={!r}".format(
            invalid_multioutput
        )
    )
    with pytest.raises(ValueError, match=expected_message):
        _check_reg_targets([1, 2, 3], [[1], [2], [3]], invalid_multioutput)


def test_regression_multioutput_array():
    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")

    pbl = mean_pinball_loss(y_true, y_pred, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    r = r2_score(y_true, y_pred, multioutput="raw_values")
    evs = explained_variance_score(y_true, y_pred, multioutput="raw_values")
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")
    evs2 = explained_variance_score(
        y_true, y_pred, multioutput="raw_values", force_finite=False
    )

    assert_array_almost_equal(mse, [0.125, 0.5625], decimal=2)
    assert_array_almost_equal(mae, [0.25, 0.625], decimal=2)
    assert_array_almost_equal(pbl, [0.25 / 2, 0.625 / 2], decimal=2)
    assert_array_almost_equal(mape, [0.0778, 0.2262], decimal=2)
    assert_array_almost_equal(r, [0.95, 0.93], decimal=2)
    assert_array_almost_equal(evs, [0.95, 0.93], decimal=2)
    assert_array_almost_equal(d2ps, [0.833, 0.722], decimal=2)
    assert_array_almost_equal(evs2, [0.95, 0.93], decimal=2)

    # mean_absolute_error and mean_squared_error are equal because
    # it is a binary problem.
    y_true = [[0, 0]] * 4
    y_pred = [[1, 1]] * 4
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    pbl = mean_pinball_loss(y_true, y_pred, multioutput="raw_values")
    r = r2_score(y_true, y_pred, multioutput="raw_values")
    d2ps = d2_pinball_score(y_true, y_pred, multioutput="raw_values")
    assert_array_almost_equal(mse, [1.0, 1.0], decimal=2)
    assert_array_almost_equal(mae, [1.0, 1.0], decimal=2)
    assert_array_almost_equal(pbl, [0.5, 0.5], decimal=2)
    assert_array_almost_equal(r, [0.0, 0.0], decimal=2)
    assert_array_almost_equal(d2ps, [0.0, 0.0], decimal=2)

    r = r2_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="raw_values")
    assert_array_almost_equal(r, [0, -3.5], decimal=2)
    assert np.mean(r) == r2_score(
        [[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="uniform_average"
    )
    evs = explained_variance_score(
        [[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="raw_values"
    )
    assert_array_almost_equal(evs, [0, -1.25], decimal=2)
    evs2 = explained_variance_score(
        [[0, -1], [0, 1]],
        [[2, 2], [1, 1]],
        multioutput="raw_values",
        force_finite=False,
    )
    assert_array_almost_equal(evs2, [-np.inf, -1.25], decimal=2)

    # Checking for the condition in which both numerator and denominator is
    # zero.
    y_true = [[1, 3], [1, 2]]
    y_pred = [[1, 4], [1, 1]]
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    assert_array_almost_equal(r2, [1.0, -3.0], decimal=2)
    assert np.mean(r2) == r2_score(y_true, y_pred, multioutput="uniform_average")
    r22 = r2_score(y_true, y_pred, multioutput="raw_values", force_finite=False)
    assert_array_almost_equal(r22, [np.nan, -3.0], decimal=2)
    assert_almost_equal(
        np.mean(r22),
        r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False),
    )

    evs = explained_variance_score(y_true, y_pred, multioutput="raw_values")
    assert_array_almost_equal(evs, [1.0, -3.0], decimal=2)
    assert np.mean(evs) == explained_variance_score(y_true, y_pred)
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")
    assert_array_almost_equal(d2ps, [1.0, -1.0], decimal=2)
    evs2 = explained_variance_score(
        y_true, y_pred, multioutput="raw_values", force_finite=False
    )
    assert_array_almost_equal(evs2, [np.nan, -3.0], decimal=2)
    assert_almost_equal(
        np.mean(evs2), explained_variance_score(y_true, y_pred, force_finite=False)
    )

    # Handling msle separately as it does not accept negative inputs.
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    msle = mean_squared_log_error(y_true, y_pred, multioutput="raw_values")
    msle2 = mean_squared_error(
        np.log(1 + y_true), np.log(1 + y_pred), multioutput="raw_values"
    )
    assert_array_almost_equal(msle, msle2, decimal=2)


def test_regression_custom_weights():
    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    msew = mean_squared_error(y_true, y_pred, multioutput=[0.4, 0.6])
    rmsew = root_mean_squared_error(y_true, y_pred, multioutput=[0.4, 0.6])
    maew = mean_absolute_error(y_true, y_pred, multioutput=[0.4, 0.6])
    mapew = mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.4, 0.6])
    rw = r2_score(y_true, y_pred, multioutput=[0.4, 0.6])
    evsw = explained_variance_score(y_true, y_pred, multioutput=[0.4, 0.6])
    d2psw = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput=[0.4, 0.6])
    evsw2 = explained_variance_score(
        y_true, y_pred, multioutput=[0.4, 0.6], force_finite=False
    )

    assert_almost_equal(msew, 0.39, decimal=2)
    assert_almost_equal(rmsew, 0.59, decimal=2)
    assert_almost_equal(maew, 0.475, decimal=3)
    assert_almost_equal(mapew, 0.1668, decimal=2)
    assert_almost_equal(rw, 0.94, decimal=2)
    assert_almost_equal(evsw, 0.94, decimal=2)
    assert_almost_equal(d2psw, 0.766, decimal=2)
    assert_almost_equal(evsw2, 0.94, decimal=2)

    # Handling msle separately as it does not accept negative inputs.
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    msle = mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
    msle2 = mean_squared_error(
        np.log(1 + y_true), np.log(1 + y_pred), multioutput=[0.3, 0.7]
    )
    assert_almost_equal(msle, msle2, decimal=2)


@pytest.mark.parametrize("metric", [r2_score, d2_tweedie_score, d2_pinball_score])
def test_regression_single_sample(metric):
    y_true = [0]
    y_pred = [1]
    warning_msg = "not well-defined with less than two samples."

    # Trigger the warning
    with pytest.warns(UndefinedMetricWarning, match=warning_msg):
        score = metric(y_true, y_pred)
        assert np.isnan(score)


def test_tweedie_deviance_continuity():
    n_samples = 100

    y_true = np.random.RandomState(0).rand(n_samples) + 0.1
    y_pred = np.random.RandomState(1).rand(n_samples) + 0.1

    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=0 - 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=0),
    )

    # Ws we get closer to the limit, with 1e-12 difference the absolute
    # tolerance to pass the below check increases. There are likely
    # numerical precision issues on the edges of different definition
    # regions.
    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=1 + 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=1),
        atol=1e-6,
    )

    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=2 - 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=2),
        atol=1e-6,
    )

    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=2 + 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=2),
        atol=1e-6,
    )


def test_mean_absolute_percentage_error():
    random_number_generator = np.random.RandomState(42)
    y_true = random_number_generator.exponential(size=100)
    y_pred = 1.2 * y_true
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "distribution", ["normal", "lognormal", "exponential", "uniform"]
)
@pytest.mark.parametrize("target_quantile", [0.05, 0.5, 0.75])
def test_mean_pinball_loss_on_constant_predictions(distribution, target_quantile):
    if not hasattr(np, "quantile"):
        pytest.skip(
            "This test requires a more recent version of numpy "
            "with support for np.quantile."
        )

    # Check that the pinball loss is minimized by the empirical quantile.
    n_samples = 3000
    rng = np.random.RandomState(42)
    data = getattr(rng, distribution)(size=n_samples)

    # Compute the best possible pinball loss for any constant predictor:
    best_pred = np.quantile(data, target_quantile)
    best_constant_pred = np.full(n_samples, fill_value=best_pred)
    best_pbl = mean_pinball_loss(data, best_constant_pred, alpha=target_quantile)

    # Evaluate the loss on a grid of quantiles
    candidate_predictions = np.quantile(data, np.linspace(0, 1, 100))
    for pred in candidate_predictions:
        # Compute the pinball loss of a constant predictor:
        constant_pred = np.full(n_samples, fill_value=pred)
        pbl = mean_pinball_loss(data, constant_pred, alpha=target_quantile)

        # Check that the loss of this constant predictor is greater or equal
        # than the loss of using the optimal quantile (up to machine
        # precision):
        assert pbl >= best_pbl - np.finfo(best_pbl.dtype).eps

        # Check that the value of the pinball loss matches the analytical
        # formula.
        expected_pbl = (pred - data[data < pred]).sum() * (1 - target_quantile) + (
            data[data >= pred] - pred
        ).sum() * target_quantile
        expected_pbl /= n_samples
        assert_almost_equal(expected_pbl, pbl)

    # Check that we can actually recover the target_quantile by minimizing the
    # pinball loss w.r.t. the constant prediction quantile.
    def objective_func(x):
        constant_pred = np.full(n_samples, fill_value=x)
        return mean_pinball_loss(data, constant_pred, alpha=target_quantile)

    result = optimize.minimize(objective_func, data.mean(), method="Nelder-Mead")
    assert result.success
    # The minimum is not unique with limited data, hence the large tolerance.
    assert result.x == pytest.approx(best_pred, rel=1e-2)
    assert result.fun == pytest.approx(best_pbl)


def test_dummy_quantile_parameter_tuning():
    # Integration test to check that it is possible to use the pinball loss to
    # tune the hyperparameter of a quantile regressor. This is conceptually
    # similar to the previous test but using the scikit-learn estimator and
    # scoring API instead.
    n_samples = 1000
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, 5))  # Ignored
    y = rng.exponential(size=n_samples)

    all_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for alpha in all_quantiles:
        neg_mean_pinball_loss = make_scorer(
            mean_pinball_loss,
            alpha=alpha,
            greater_is_better=False,
        )
        regressor = DummyRegressor(strategy="quantile", quantile=0.25)
        grid_search = GridSearchCV(
            regressor,
            param_grid=dict(quantile=all_quantiles),
            scoring=neg_mean_pinball_loss,
        ).fit(X, y)

        assert grid_search.best_params_["quantile"] == pytest.approx(alpha)


def test_pinball_loss_relation_with_mae():
    # Test that mean_pinball loss with alpha=0.5 if half of mean absolute error
    rng = np.random.RandomState(714)
    n = 100
    y_true = rng.normal(size=n)
    y_pred = y_true.copy() + rng.uniform(n)
    assert (
        mean_absolute_error(y_true, y_pred)
        == mean_pinball_loss(y_true, y_pred, alpha=0.5) * 2
    )


# TODO(1.6): remove this test
@pytest.mark.parametrize("metric", [mean_squared_error, mean_squared_log_error])
def test_mean_squared_deprecation_squared(metric):
    """Check the deprecation warning of the squared parameter"""
    depr_msg = "'squared' is deprecated in version 1.4 and will be removed in 1.6."
    y_true, y_pred = np.arange(10), np.arange(1, 11)
    with pytest.warns(FutureWarning, match=depr_msg):
        metric(y_true, y_pred, squared=False)


# TODO(1.6): remove this test
@pytest.mark.filterwarnings("ignore:'squared' is deprecated")
@pytest.mark.parametrize(
    "old_func, new_func",
    [
        (mean_squared_error, root_mean_squared_error),
        (mean_squared_log_error, root_mean_squared_log_error),
    ],
)
def test_rmse_rmsle_parameter(old_func, new_func):
    # Check that the new rmse/rmsle function is equivalent to
    # the old mse/msle + squared=False function.
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    sw = np.arange(len(y_true))

    expected = old_func(y_true, y_pred, squared=False)
    actual = new_func(y_true, y_pred)
    assert_allclose(expected, actual)

    expected = old_func(y_true, y_pred, sample_weight=sw, squared=False)
    actual = new_func(y_true, y_pred, sample_weight=sw)
    assert_allclose(expected, actual)

    expected = old_func(y_true, y_pred, multioutput="raw_values", squared=False)
    actual = new_func(y_true, y_pred, multioutput="raw_values")
    assert_allclose(expected, actual)

    expected = old_func(
        y_true, y_pred, sample_weight=sw, multioutput="raw_values", squared=False
    )
    actual = new_func(y_true, y_pred, sample_weight=sw, multioutput="raw_values")
    assert_allclose(expected, actual)
