import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises

from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate as constrain,
    unconstrain_stationary_univariate as unconstrain)

from statsmodels.tsa.arima import specification


def check_attributes(spec, order, seasonal_order, enforce_stationarity,
                     enforce_invertibility, concentrate_scale):
    p, d, q = order
    P, D, Q, s = seasonal_order

    assert_equal(spec.order, (p, d, q))
    assert_equal(spec.seasonal_order, (P, D, Q, s))

    assert_equal(spec.ar_order, p)
    assert_equal(spec.diff, d)
    assert_equal(spec.ma_order, q)

    assert_equal(spec.seasonal_ar_order, P)
    assert_equal(spec.seasonal_diff, D)
    assert_equal(spec.seasonal_ma_order, Q)
    assert_equal(spec.seasonal_periods, s)

    assert_equal(spec.ar_lags,
                 (p if isinstance(p, list) else np.arange(1, p + 1)))
    assert_equal(spec.ma_lags,
                 (q if isinstance(q, list) else np.arange(1, q + 1)))

    assert_equal(spec.seasonal_ar_lags,
                 (P if isinstance(P, list) else np.arange(1, P + 1)))
    assert_equal(spec.seasonal_ma_lags,
                 (Q if isinstance(Q, list) else np.arange(1, Q + 1)))

    max_ar_order = p[-1] if isinstance(p, list) else p
    max_ma_order = q[-1] if isinstance(q, list) else q
    max_seasonal_ar_order = P[-1] if isinstance(P, list) else P
    max_seasonal_ma_order = Q[-1] if isinstance(Q, list) else Q
    assert_equal(spec.max_ar_order, max_ar_order)
    assert_equal(spec.max_ma_order, max_ma_order)
    assert_equal(spec.max_seasonal_ar_order, max_seasonal_ar_order)
    assert_equal(spec.max_seasonal_ma_order, max_seasonal_ma_order)
    assert_equal(spec.max_reduced_ar_order,
                 max_ar_order + max_seasonal_ar_order * s)
    assert_equal(spec.max_reduced_ma_order,
                 max_ma_order + max_seasonal_ma_order * s)

    assert_equal(spec.enforce_stationarity, enforce_stationarity)
    assert_equal(spec.enforce_invertibility, enforce_invertibility)
    assert_equal(spec.concentrate_scale, concentrate_scale)


def check_properties(spec, order, seasonal_order, enforce_stationarity,
                     enforce_invertibility, concentrate_scale,
                     is_ar_consecutive, is_ma_consecutive, exog_names,
                     ar_names, ma_names, seasonal_ar_names, seasonal_ma_names):
    p, d, q = order
    P, D, Q, s = seasonal_order

    k_exog_params = len(exog_names)
    k_ar_params = len(p) if isinstance(p, list) else p
    k_ma_params = len(q) if isinstance(q, list) else q
    k_seasonal_ar_params = len(P) if isinstance(P, list) else P
    k_seasonal_ma_params = len(Q) if isinstance(Q, list) else Q
    k_variance_params = int(not concentrate_scale)

    param_names = (exog_names + ar_names + ma_names + seasonal_ar_names +
                   seasonal_ma_names)
    if not concentrate_scale:
        param_names.append('sigma2')

    assert_equal(spec.is_ar_consecutive, is_ar_consecutive)
    assert_equal(spec.is_ma_consecutive, is_ma_consecutive)
    assert_equal(spec.is_integrated, d + D > 0)
    assert_equal(spec.is_seasonal, s > 0)

    assert_equal(spec.k_exog_params, k_exog_params)
    assert_equal(spec.k_ar_params, k_ar_params)
    assert_equal(spec.k_ma_params, k_ma_params)
    assert_equal(spec.k_seasonal_ar_params, k_seasonal_ar_params)
    assert_equal(spec.k_seasonal_ma_params, k_seasonal_ma_params)
    assert_equal(spec.k_params,
                 k_exog_params + k_ar_params + k_ma_params +
                 k_seasonal_ar_params + k_seasonal_ma_params +
                 k_variance_params)

    assert_equal(spec.exog_names, exog_names)
    assert_equal(spec.ar_names, ar_names)
    assert_equal(spec.ma_names, ma_names)
    assert_equal(spec.seasonal_ar_names, seasonal_ar_names)
    assert_equal(spec.seasonal_ma_names, seasonal_ma_names)
    assert_equal(spec.param_names, param_names)


def check_methods(spec, order, seasonal_order, enforce_stationarity,
                  enforce_invertibility, concentrate_scale,
                  exog_params, ar_params, ma_params, seasonal_ar_params,
                  seasonal_ma_params, sigma2):
    params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params,
                   seasonal_ma_params, sigma2]

    # Test methods
    desired = {
        'exog_params': exog_params,
        'ar_params': ar_params,
        'ma_params': ma_params,
        'seasonal_ar_params': seasonal_ar_params,
        'seasonal_ma_params': seasonal_ma_params}
    if not concentrate_scale:
        desired['sigma2'] = sigma2
    assert_equal(spec.split_params(params), desired)

    assert_equal(spec.join_params(**desired), params)

    assert_equal(spec.validate_params(params), None)

    # Wrong shape
    assert_raises(ValueError, spec.validate_params, [])

    # Wrong dtype
    assert_raises(ValueError, spec.validate_params,
                  ['a'] + params[1:].tolist())

    # NaN / Infinity
    assert_raises(ValueError, spec.validate_params,
                  np.r_[np.inf, params[1:]])
    assert_raises(ValueError, spec.validate_params,
                  np.r_[np.nan, params[1:]])

    # Non-stationary / non-invertible
    if spec.max_ar_order > 0:
        params = np.r_[exog_params, np.ones_like(ar_params), ma_params,
                       np.zeros_like(seasonal_ar_params),
                       seasonal_ma_params, sigma2]
        if enforce_stationarity:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_ma_order > 0:
        params = np.r_[exog_params, ar_params, np.ones_like(ma_params),
                       seasonal_ar_params, np.zeros_like(seasonal_ma_params),
                       sigma2]
        if enforce_invertibility:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_seasonal_ar_order > 0:
        params = np.r_[exog_params, np.zeros_like(ar_params), ma_params,
                       np.ones_like(seasonal_ar_params), seasonal_ma_params,
                       sigma2]
        if enforce_stationarity:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)
    if spec.max_seasonal_ma_order > 0:
        params = np.r_[exog_params, ar_params, np.zeros_like(ma_params),
                       seasonal_ar_params, np.ones_like(seasonal_ma_params),
                       sigma2]
        if enforce_invertibility:
            assert_raises(ValueError, spec.validate_params, params)
        else:
            assert_equal(spec.validate_params(params), None)

    # Invalid variances
    if not concentrate_scale:
        params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params,
                       seasonal_ma_params, 0.]
        assert_raises(ValueError, spec.validate_params, params)
        params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params,
                       seasonal_ma_params, -1]
        assert_raises(ValueError, spec.validate_params, params)

    # Constrain / unconstrain
    unconstrained_ar_params = ar_params
    unconstrained_ma_params = ma_params
    unconstrained_seasonal_ar_params = seasonal_ar_params
    unconstrained_seasonal_ma_params = seasonal_ma_params
    unconstrained_sigma2 = sigma2

    if spec.max_ar_order > 0 and enforce_stationarity:
        unconstrained_ar_params = unconstrain(np.array(ar_params))
    if spec.max_ma_order > 0 and enforce_invertibility:
        unconstrained_ma_params = unconstrain(-np.array(ma_params))
    if spec.max_seasonal_ar_order > 0 and enforce_stationarity:
        unconstrained_seasonal_ar_params = (
            unconstrain(np.array(seasonal_ar_params)))
    if spec.max_seasonal_ma_order > 0 and enforce_invertibility:
        unconstrained_seasonal_ma_params = (
            unconstrain(-np.array(unconstrained_seasonal_ma_params)))
    if not concentrate_scale:
        unconstrained_sigma2 = unconstrained_sigma2**0.5

    unconstrained_params = np.r_[
        exog_params, unconstrained_ar_params, unconstrained_ma_params,
        unconstrained_seasonal_ar_params, unconstrained_seasonal_ma_params,
        unconstrained_sigma2]
    params = np.r_[exog_params, ar_params, ma_params, seasonal_ar_params,
                   seasonal_ma_params, sigma2]

    assert_allclose(spec.unconstrain_params(params), unconstrained_params)

    assert_allclose(spec.constrain_params(unconstrained_params), params)

    assert_allclose(
        spec.constrain_params(spec.unconstrain_params(params)), params)


@pytest.mark.parametrize("n,d,D,s,params,which", [
    # AR models
    (0, 0, 0, 0, np.array([1.]), 'p'),
    (1, 0, 0, 0, np.array([0.5, 1.]), 'p'),
    (1, 0, 0, 0, np.array([-0.2, 100.]), 'p'),
    (2, 0, 0, 0, np.array([-0.2, 0.5, 100.]), 'p'),
    (20, 0, 0, 0, np.array([0.0] * 20 + [100.]), 'p'),
    # ARI models
    (0, 1, 0, 0, np.array([1.]), 'p'),
    (0, 1, 1, 4, np.array([1.]), 'p'),
    (1, 1, 0, 0, np.array([0.5, 1.]), 'p'),
    (1, 1, 1, 4, np.array([0.5, 1.]), 'p'),
    # MA models
    (0, 0, 0, 0, np.array([1.]), 'q'),
    (1, 0, 0, 0, np.array([0.5, 1.]), 'q'),
    (1, 0, 0, 0, np.array([-0.2, 100.]), 'q'),
    (2, 0, 0, 0, np.array([-0.2, 0.5, 100.]), 'q'),
    (20, 0, 0, 0, np.array([0.0] * 20 + [100.]), 'q'),
    # IMA models
    (0, 1, 0, 0, np.array([1.]), 'q'),
    (0, 1, 1, 4, np.array([1.]), 'q'),
    (1, 1, 0, 0, np.array([0.5, 1.]), 'q'),
    (1, 1, 1, 4, np.array([0.5, 1.]), 'q'),
])
def test_specification_ar_or_ma(n, d, D, s, params, which):
    if which == 'p':
        p, d, q = n, d, 0
        ar_names = ['ar.L%d' % i for i in range(1, p + 1)]
        ma_names = []
    else:
        p, d, q = 0, d, n
        ar_names = []
        ma_names = ['ma.L%d' % i for i in range(1, q + 1)]
    ar_params = params[:p]
    ma_params = params[p:-1]
    sigma2 = params[-1]
    P, D, Q, s = 0, D, 0, s

    args = ((p, d, q), (P, D, Q, s))
    kwargs = {
        'enforce_stationarity': None,
        'enforce_invertibility': None,
        'concentrate_scale': None
    }

    properties_kwargs = kwargs.copy()
    properties_kwargs.update({
        'is_ar_consecutive': True,
        'is_ma_consecutive': True,
        'exog_names': [],
        'ar_names': ar_names,
        'ma_names': ma_names,
        'seasonal_ar_names': [],
        'seasonal_ma_names': []})

    methods_kwargs = kwargs.copy()
    methods_kwargs.update({
        'exog_params': [],
        'ar_params': ar_params,
        'ma_params': ma_params,
        'seasonal_ar_params': [],
        'seasonal_ma_params': [],
        'sigma2': sigma2})

    # Test the spec created with order, seasonal_order
    spec = specification.SARIMAXSpecification(
            order=(p, d, q), seasonal_order=(P, D, Q, s))

    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)

    # Test the spec created with ar_order, etc.
    spec = specification.SARIMAXSpecification(
            ar_order=p, diff=d, ma_order=q, seasonal_ar_order=P,
            seasonal_diff=D, seasonal_ma_order=Q, seasonal_periods=s)

    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)


@pytest.mark.parametrize(("endog,exog,p,d,q,P,D,Q,s,"
                          "enforce_stationarity,enforce_invertibility,"
                          "concentrate_scale"), [
    (None, None, 0, 0, 0, 0, 0, 0, 0, True, True, False),
    (None, None, 1, 0, 1, 0, 0, 0, 0, True, True, False),
    (None, None, 1, 1, 1, 0, 0, 0, 0, True, True, False),
    (None, None, 1, 0, 0, 0, 0, 0, 4, True, True, False),
    (None, None, 0, 0, 0, 1, 1, 1, 4, True, True, False),
    (None, None, 1, 0, 0, 1, 0, 0, 4, True, True, False),
    (None, None, 1, 0, 0, 1, 1, 1, 4, True, True, False),
    (None, None, 2, 1, 3, 4, 1, 3, 12, True, True, False),

    # Non-consecutive lag orders
    (None, None, [1, 3], 0, 0, 1, 0, 0, 4, True, True, False),
    (None, None, 0, 0, 0, 0, 0, [1, 3], 4, True, True, False),
    (None, None, [2], 0, [1, 3], [1, 3], 0, [1, 4], 4, True, True, False),

    # Modify enforce / concentrate
    (None, None, 2, 1, 3, 4, 1, 3, 12, False, False, True),
    (None, None, 2, 1, 3, 4, 1, 3, 12, True, False, True),
    (None, None, 2, 1, 3, 4, 1, 3, 12, False, True, True),

    # Endog / exog
    (True, None, 2, 1, 3, 4, 1, 3, 12, False, True, True),
    (None, 2, 2, 1, 3, 4, 1, 3, 12, False, True, True),
    (True, 2, 2, 1, 3, 4, 1, 3, 12, False, True, True),
    ('y', None, 2, 1, 3, 4, 1, 3, 12, False, True, True),
    (None, ['x1'], 2, 1, 3, 4, 1, 3, 12, False, True, True),
    ('y', ['x1'], 2, 1, 3, 4, 1, 3, 12, False, True, True),
    ('y', ['x1', 'x2'], 2, 1, 3, 4, 1, 3, 12, False, True, True),
    (True, ['x1', 'x2'], 2, 1, 3, 4, 1, 3, 12, False, True, True),
    ('y', 2, 2, 1, 3, 4, 1, 3, 12, False, True, True),
])
def test_specification(endog, exog, p, d, q, P, D, Q, s,
                       enforce_stationarity, enforce_invertibility,
                       concentrate_scale):
    # Assumptions:
    # - p, q, P, Q are either integers or lists of non-consecutive integers
    #   (i.e. we are not testing boolean lists or consecutive lists here, which
    #   should be tested in the `standardize_lag_order` tests)

    # Construct the specification
    if isinstance(p, list):
        k_ar_params = len(p)
        max_ar_order = p[-1]
    else:
        k_ar_params = max_ar_order = p

    if isinstance(q, list):
        k_ma_params = len(q)
        max_ma_order = q[-1]
    else:
        k_ma_params = max_ma_order = q

    if isinstance(P, list):
        k_seasonal_ar_params = len(P)
        max_seasonal_ar_order = P[-1]
    else:
        k_seasonal_ar_params = max_seasonal_ar_order = P

    if isinstance(Q, list):
        k_seasonal_ma_params = len(Q)
        max_seasonal_ma_order = Q[-1]
    else:
        k_seasonal_ma_params = max_seasonal_ma_order = Q

    # Get endog / exog
    nobs = d + D * s + max(3 * max_ma_order + 1,
                           3 * max_seasonal_ma_order * s + 1,
                           max_ar_order,
                           max_seasonal_ar_order * s) + 1

    if endog is True:
        endog = np.arange(nobs) * 1.0
    elif isinstance(endog, str):
        endog = pd.Series(np.arange(nobs) * 1.0, name=endog)
    elif endog is not None:
        raise ValueError('Invalid `endog` in test setup.')

    if isinstance(exog, int):
        exog_names = ['x%d' % (i + 1) for i in range(exog)]
        exog = np.arange(nobs * len(exog_names)).reshape(nobs, len(exog_names))
    elif isinstance(exog, list):
        exog_names = exog
        exog = np.arange(nobs * len(exog_names)).reshape(nobs, len(exog_names))
        exog = pd.DataFrame(exog, columns=exog_names)
    elif exog is None:
        exog_names = []
    else:
        raise ValueError('Invalid `exog` in test setup.')

    # Setup args, kwargs
    args = ((p, d, q), (P, D, Q, s))
    kwargs = {
        'enforce_stationarity': enforce_stationarity,
        'enforce_invertibility': enforce_invertibility,
        'concentrate_scale': concentrate_scale
    }
    properties_kwargs = kwargs.copy()
    is_ar_consecutive = not isinstance(p, list) and max_seasonal_ar_order == 0
    is_ma_consecutive = not isinstance(q, list) and max_seasonal_ma_order == 0
    properties_kwargs.update({
        'is_ar_consecutive': is_ar_consecutive,
        'is_ma_consecutive': is_ma_consecutive,
        'exog_names': exog_names,
        'ar_names': [
            'ar.L%d' % i
            for i in (p if isinstance(p, list) else range(1, p + 1))],
        'ma_names': [
            'ma.L%d' % i
            for i in (q if isinstance(q, list) else range(1, q + 1))],
        'seasonal_ar_names': [
            'ar.S.L%d' % (i * s)
            for i in (P if isinstance(P, list) else range(1, P + 1))],
        'seasonal_ma_names': [
            'ma.S.L%d' % (i * s)
            for i in (Q if isinstance(Q, list) else range(1, Q + 1))]})

    methods_kwargs = kwargs.copy()
    methods_kwargs.update({
        'exog_params': np.arange(len(exog_names)),
        'ar_params': (
            [] if k_ar_params == 0 else
            constrain(np.arange(k_ar_params) / 10)),
        'ma_params': (
            [] if k_ma_params == 0 else
            constrain((np.arange(k_ma_params) + 10) / 100)),
        'seasonal_ar_params': (
            [] if k_seasonal_ar_params == 0 else
            constrain(np.arange(k_seasonal_ar_params) - 4)),
        'seasonal_ma_params': (
            [] if k_seasonal_ma_params == 0 else
            constrain((np.arange(k_seasonal_ma_params) - 10) / 100)),
        'sigma2': [] if concentrate_scale else 2.3424})

    # Test the spec created with order, seasonal_order
    spec = specification.SARIMAXSpecification(
        endog, exog=exog,
        order=(p, d, q), seasonal_order=(P, D, Q, s),
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale)

    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)

    # Test the spec created with ar_order, etc.
    spec = specification.SARIMAXSpecification(
        endog, exog=exog,
        ar_order=p, diff=d, ma_order=q, seasonal_ar_order=P,
        seasonal_diff=D, seasonal_ma_order=Q, seasonal_periods=s,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale)

    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)


def test_misc():
    # Check that no arguments results in all zero orders
    spec = specification.SARIMAXSpecification()
    assert_equal(spec.order, (0, 0, 0))
    assert_equal(spec.seasonal_order, (0, 0, 0, 0))

    # Check for repr
    spec = specification.SARIMAXSpecification(
        endog=pd.Series([0], name='y'),
        exog=pd.DataFrame([[0, 0]], columns=['x1', 'x2']),
        order=(1, 1, 2), seasonal_order=(2, 1, 0, 12),
        enforce_stationarity=False, enforce_invertibility=False,
        concentrate_scale=True)
    desired = ("SARIMAXSpecification(endog=y, exog=['x1', 'x2'],"
               " order=(1, 1, 2), seasonal_order=(2, 1, 0, 12),"
               " enforce_stationarity=False, enforce_invertibility=False,"
               " concentrate_scale=True)")
    assert_equal(repr(spec), desired)


def test_invalid():
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(1, 0, 0), ar_order=1)
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1, 0, 0), seasonal_ar_order=1)
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(-1, 0, 0))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(1.5, 0, 0))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(0, -1, 0))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(0, 1.5, 0))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(0,))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(0, 1.5, 0, 4))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(-1, 0, 0, 4))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1.5, 0, 0, 4))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(0, -1, 0, 4))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(0, 1.5, 0, 4))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1, 0, 0, 0))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1, 0, 0, -1))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1, 0, 0, 1))
    assert_raises(ValueError, specification.SARIMAXSpecification,
                  seasonal_order=(1,))

    assert_raises(ValueError, specification.SARIMAXSpecification,
                  order=(1, 0, 0), endog=np.zeros((10, 2)))

    spec = specification.SARIMAXSpecification(ar_order=1)
    assert_raises(ValueError, spec.join_params)
    assert_raises(ValueError, spec.join_params, ar_params=[0.2, 0.3])


@pytest.mark.parametrize(
    "order,seasonal_order,enforce_stationarity,"
    "enforce_invertibility,concentrate_scale,valid", [
        # Different orders
        ((0, 0, 0), (0, 0, 0, 0), None, None, None,
            ['yule_walker', 'burg', 'innovations', 'hannan_rissanen',
             'innovations_mle', 'statespace']),
        ((1, 0, 0), (0, 0, 0, 0), None, None, None,
            ['yule_walker', 'burg', 'hannan_rissanen',
             'innovations_mle', 'statespace']),
        ((0, 0, 1), (0, 0, 0, 0), None, None, None,
            ['innovations', 'hannan_rissanen', 'innovations_mle',
             'statespace']),
        ((1, 0, 1), (0, 0, 0, 0), None, None, None,
            ['hannan_rissanen', 'innovations_mle', 'statespace']),
        ((0, 0, 0), (1, 0, 0, 4), None, None, None,
            ['innovations_mle', 'statespace']),

        # Different options
        ((1, 0, 0), (0, 0, 0, 0), True, None, None,
            ['innovations_mle', 'statespace']),
        ((1, 0, 0), (0, 0, 0, 0), False, None, None,
            ['yule_walker', 'burg', 'hannan_rissanen', 'statespace']),
        ((1, 0, 0), (0, 0, 0, 0), None, True, None,
            ['yule_walker', 'burg', 'hannan_rissanen', 'innovations_mle',
             'statespace']),
        ((1, 0, 0), (0, 0, 0, 0), None, False, None,
            ['yule_walker', 'burg', 'hannan_rissanen', 'innovations_mle',
             'statespace']),
        ((1, 0, 0), (0, 0, 0, 0), None, None, True,
            ['yule_walker', 'burg', 'hannan_rissanen', 'statespace']),
    ])
def test_valid_estimators(order, seasonal_order, enforce_stationarity,
                          enforce_invertibility, concentrate_scale, valid):
    # Basic specification
    spec = specification.SARIMAXSpecification(
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale)

    estimators = set(['yule_walker', 'burg', 'innovations',
                      'hannan_rissanen', 'innovations_mle', 'statespace'])
    desired = set(valid)
    assert_equal(spec.valid_estimators, desired)
    for estimator in desired:
        assert_equal(spec.validate_estimator(estimator), None)
    for estimator in estimators.difference(desired):
        print(estimator, enforce_stationarity)
        assert_raises(ValueError, spec.validate_estimator, estimator)

    # Now try specification with missing values in endog
    spec = specification.SARIMAXSpecification(
        endog=[np.nan],
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale)

    assert_equal(spec.valid_estimators, set(['statespace']))
    assert_equal(spec.validate_estimator('statespace'), None)
    for estimator in estimators.difference(['statespace']):
        assert_raises(ValueError, spec.validate_estimator, estimator)


def test_invalid_estimator():
    spec = specification.SARIMAXSpecification()
    assert_raises(ValueError, spec.validate_estimator, 'not_an_estimator')
