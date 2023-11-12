r"""
Tests for news results

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import (
    assert_frame_equal,
    assert_series_equal,
)

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import (
    dynamic_factor,
    sarimax,
    structural,
    varmax,
)

dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start='1959Q1', end='2009Q3', freq='Q')


def check_impact_indices(news, impact_dates, impacted_variables):
    # Note: the index for impacts is only a time index, because we compute
    # impacts for all variables during these time periods.
    for attr in ['total_impacts', 'update_impacts', 'revision_impacts',
                 'post_impacted_forecasts', 'prev_impacted_forecasts']:
        val = getattr(news, attr)
        assert_(val.index.equals(impact_dates))
        assert_equal(val.columns.tolist(), impacted_variables)


def check_revision_indices(news, revisions_index):
    # Note: revision indices must be a MultiIndex, because for each time
    # period, not all variables necessarily revised.

    # If there are no revisions, we just check the length is zero
    if news.news_results.revision_impacts is None:
        assert_equal(len(news.revisions_iloc['revision date']), 0)
        assert_equal(len(news.revisions_iloc['revised variable']), 0)
        assert_equal(len(news.revisions_ix['revision date']), 0)
        assert_equal(len(news.revisions_ix['revised variable']), 0)
    # Otherwise, check that the revision indexes are correct
    else:
        # Get the integer locations of the given indexes
        dates = news.previous.model._index
        endog_names = news.previous.model.endog_names
        if isinstance(endog_names, str):
            endog_names = [endog_names]
        desired_ix = revisions_index.to_frame().reset_index(drop=True)
        desired_iloc = desired_ix.copy()
        desired_iloc['revision date'] = [
            dates.get_loc(date) for date in desired_ix['revision date']]
        desired_iloc['revised variable'] = [
            endog_names.index(name)
            for name in desired_ix['revised variable']]

        assert_(news.revisions_iloc.equals(
            desired_iloc.astype(news.revisions_iloc.dtypes)))
        assert_(news.revisions_ix.equals(desired_ix))


def check_update_indices(news, updates_index):
    # Note: update indices are also a MultiIndex, for the same reason as the
    # revision indices.
    # If there are no updates, we just check the length is zero
    if news.news_results.update_impacts is None:
        assert_equal(len(news.updates_iloc['update date']), 0)
        assert_equal(len(news.updates_iloc['updated variable']), 0)
        assert_equal(len(news.updates_ix['update date']), 0)
        assert_equal(len(news.updates_ix['updated variable']), 0)
    # Otherwise, check that the revision indexes are correct
    else:
        # Get the integer locations of the given indexes
        dates = news.updated.model._index
        endog_names = news.updated.model.endog_names
        if isinstance(endog_names, str):
            endog_names = [endog_names]
        desired_ix = updates_index.to_frame().reset_index(drop=True)
        desired_iloc = desired_ix.copy()
        desired_iloc['update date'] = [
            dates.get_loc(date) for date in desired_ix['update date']]
        desired_iloc['updated variable'] = [
            endog_names.index(name)
            for name in desired_ix['updated variable']]

        assert_(news.updates_iloc.equals(
            desired_iloc.astype(news.updates_iloc.dtypes)))
        assert_(news.updates_ix.equals(desired_ix))


def check_news_indices(news, updates_index, impact_dates):
    # News are computed only from updates, so the news indices are the same as
    # the update indices
    if len(updates_index):
        news_index = updates_index
    else:
        news_index = pd.MultiIndex.from_product(
            [[], []], names=['update date', 'updated variable'])
    endog_names = news.previous.model.endog_names
    if isinstance(endog_names, str):
        endog_names = [endog_names]

    assert_(news.news.index.equals(news_index))
    assert_(news.update_forecasts.index.equals(news_index))
    assert_(news.update_realized.index.equals(news_index))
    assert_(news.weights.index.equals(news_index))

    weights_columns = pd.MultiIndex.from_product([impact_dates, endog_names])
    assert_(news.weights.columns.equals(weights_columns))


def check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights):
    # Note: we use atol=1e-12 to handle cases where impacts, etc. are equal to
    # zero, but numerical precision of the Kalman filter procedures gives an
    # answer of e.g. 1e-16.

    # Note: Here we set the tolerance to be slightly negative, since some of
    # the tests have weights or impacts exactly equal to zero, while we still
    # want to include those in tests.
    news.tolerance = -1e-10

    # - Indexes --------------------------------------------------------------
    # Index of impacts
    check_impact_indices(news, impact_dates, impacted_variables)

    # Reivision indices
    check_revision_indices(news, revisions_index)

    # Update indices
    check_update_indices(news, updates_index)

    # News indices
    check_news_indices(news, updates_index, impact_dates)

    # - Impacts --------------------------------------------------------------
    if updates:
        assert_allclose(news.update_impacts, update_impacts, atol=1e-12)
    else:
        assert_(np.all(news.update_impacts.isnull()))

    # Impacts from revisions
    if revisions:
        assert_allclose(news.revision_impacts, revision_impacts, atol=1e-12)
    else:
        assert_(news.news_results.revision_impacts is None)
        assert_(np.all(news.revision_impacts.isnull()))

    # Total impacts
    total_impacts = (news.revision_impacts.fillna(0) +
                     news.update_impacts.fillna(0))
    assert_allclose(news.total_impacts, total_impacts, atol=1e-12)

    # - Impacted variable forecasts ------------------------------------------
    assert_allclose(news.prev_impacted_forecasts, prev_impacted_forecasts,
                    atol=1e-12)
    assert_allclose(news.post_impacted_forecasts, post_impacted_forecasts,
                    atol=1e-12)

    # - News -----------------------------------------------------------------
    assert_allclose(news.update_forecasts, update_forecasts, atol=1e-12)
    assert_allclose(news.update_realized, update_realized, atol=1e-12)
    # The "news" is simply the forecast error
    assert_allclose(news.news, news_desired, atol=1e-12)
    # The weight is zero on previously known data, and is geometrically
    # declining (according to the AR parameter) in the forecast period
    assert_allclose(news.weights, weights, atol=1e-12)

    # - Table: data revisions ------------------------------------------------
    assert_equal(news.data_revisions.columns.tolist(),
                 ['revised', 'observed (prev)'])
    assert_equal(news.data_revisions.index.names,
                 ['revision date', 'revised variable'])
    assert_(news.data_revisions.index.equals(revisions_index))

    # - Table: data updates --------------------------------------------------
    assert_equal(news.data_updates.columns.tolist(),
                 ['observed', 'forecast (prev)'])
    assert_equal(news.data_updates.index.names,
                 ['update date', 'updated variable'])
    assert_(news.data_updates.index.equals(news.news.index))
    assert_allclose(news.data_updates['forecast (prev)'],
                    news.update_forecasts, atol=1e-12)
    assert_allclose(news.data_updates['observed'], news.update_realized,
                    atol=1e-12)

    # - Table: details_by_impact ---------------------------------------------
    details_by_impact = news.details_by_impact
    desired = ['observed', 'forecast (prev)', 'news', 'weight', 'impact']
    assert_equal(details_by_impact.columns.tolist(), desired)
    desired = ['impact date', 'impacted variable',
               'update date', 'updated variable']
    assert_equal(details_by_impact.index.names, desired)

    if updates:
        actual = (news.details_by_impact['forecast (prev)']
                      .drop_duplicates()
                      .reset_index([0, 1])['forecast (prev)'])
        assert_allclose(actual, news.update_forecasts, atol=1e-12)
        actual = (news.details_by_impact['observed']
                      .drop_duplicates().reset_index([0, 1])['observed'])
        assert_allclose(actual, news.update_realized, atol=1e-12)
        actual = (news.details_by_impact['news']
                      .drop_duplicates().reset_index([0, 1])['news'])
        assert_allclose(actual, news.news, atol=1e-12)

        # Weights
        assert_allclose(details_by_impact['weight'].unstack([0, 1]),
                        news.weights, atol=1e-12)

        # Impact of news
        actual = (news.details_by_impact['impact']
                      .unstack([2, 3]).sum(axis=1).unstack(1))
        assert_allclose(actual, news.update_impacts, atol=1e-12)

    # - Table: details_by_update ---------------------------------------------
    details_by_update = news.details_by_update
    desired = ['news', 'weight', 'impact']
    assert_equal(details_by_update.columns.tolist(), desired)
    desired = ['update date', 'updated variable', 'observed',
               'forecast (prev)', 'impact date', 'impacted variable']
    assert_equal(details_by_update.index.names, desired)

    if updates:
        # News
        # Special case for Pandas = 0.23, see above
        actual = (news.details_by_update['news']
                      .drop_duplicates().reset_index([2, 3, 4, 5])['news'])
        assert_allclose(actual, news.news, atol=1e-12)

        # Weights
        assert_allclose(news.details_by_update['weight'].unstack([4, 5]),
                        news.weights, atol=1e-12)

        # Impact of news
        actual = (news.details_by_update['impact']
                      .unstack([4, 5]).sum(axis=0).unstack(1))
        assert_allclose(actual, news.update_impacts, atol=1e-12)

    # - Table: impacts -------------------------------------------------------
    impacts = news.impacts
    desired = ['estimate (prev)', 'impact of revisions', 'impact of news',
               'total impact', 'estimate (new)']
    assert_equal(impacts.columns.tolist(), desired)
    desired = ['impact date', 'impacted variable']
    assert_equal(impacts.index.names, desired)

    assert_allclose(impacts.loc[:, 'estimate (prev)'],
                    news.prev_impacted_forecasts.stack(), atol=1e-12)
    assert_allclose(impacts.loc[:, 'impact of revisions'],
                    news.revision_impacts.fillna(0).stack(), atol=1e-12)
    assert_allclose(impacts.loc[:, 'impact of news'],
                    news.update_impacts.fillna(0).stack(), atol=1e-12)
    assert_allclose(impacts.loc[:, 'total impact'],
                    news.total_impacts.stack(), atol=1e-12)
    assert_allclose(impacts.loc[:, 'estimate (new)'],
                    news.post_impacted_forecasts.stack(), atol=1e-12)


# @pytest.mark.parametrize('revisions', [True, False])
# @pytest.mark.parametrize('updates', [True, False])
@pytest.mark.parametrize('revisions', [True])
@pytest.mark.parametrize('updates', [True])
def test_sarimax_time_invariant(revisions, updates):
    # Construct previous and updated datasets
    endog = dta['infl'].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod = sarimax.SARIMAX(endog1)
    res = mod.smooth([0.5, 1.0])
    news = res.news(endog2, start='2009Q2', end='2010Q1',
                    comparison_type=comparison_type)

    # Compute the true values for each combination of (revsions, updates)
    impact_dates = pd.period_range(start='2009Q2', end='2010Q1', freq='Q')
    impacted_variables = ['infl']

    # Revisions
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_arrays(
            [endog1.index[-1:], ['infl']],
            names=['revision date', 'revised variable'])
        # If we have updates, the revision is to 2009Q2
        revision_impacts = endog2.iloc[-2] * 0.5**np.arange(4).reshape(4, 1)
    elif revisions:
        revisions_index = pd.MultiIndex.from_arrays(
            [endog1.index[-1:], ['infl']],
            names=['revision date', 'revised variable'])
        # With no updates, the revision is to 2009Q3
        revision_impacts = np.r_[
            0, endog2.iloc[-1] * 0.5**np.arange(3)].reshape(4, 1)
    else:
        revisions_index = pd.MultiIndex.from_arrays(
            [[], []], names=['revision date', 'revised variable'])
        revision_impacts = None

    # Updates
    if updates:
        updates_index = pd.MultiIndex.from_arrays(
            [pd.period_range(start='2009Q3', periods=1, freq='Q'), ['infl']],
            names=['update date', 'updated variable'])
        update_impacts = np.array([[
            0, endog.loc['2009Q3'] - 0.5 * endog.loc['2009Q2'],
            0.5 * endog.loc['2009Q3'] - 0.5**2 * endog.loc['2009Q2'],
            0.5**2 * endog.loc['2009Q3'] - 0.5**3 * endog.loc['2009Q2']]]).T
    else:
        updates_index = pd.MultiIndex.from_arrays(
            [[], []], names=['update date', 'updated variable'])
        update_impacts = None
    print(update_impacts)

    # Impact forecasts
    if updates:
        prev_impacted_forecasts = np.r_[
            endog1.iloc[-1] * 0.5**np.arange(4)].reshape(4, 1)
    else:
        prev_impacted_forecasts = np.r_[
            endog1.iloc[-2], endog1.iloc[-1] * 0.5**np.arange(3)].reshape(4, 1)
    post_impacted_forecasts = np.r_[
        endog2.iloc[-2], 0.5 ** np.arange(3) * endog2.iloc[-1]].reshape(4, 1)

    # News
    if updates:
        # Note: update_forecasts is created using the endog2 dataset even if
        # there were revisions, because it should be computed after revisions
        # have already been taken into account
        update_forecasts = [0.5 * endog2.loc['2009Q2']]
        update_realized = [endog2.loc['2009Q3']]
        news_desired = [update_realized[i] - update_forecasts[i]
                        for i in range(len(update_forecasts))]
        weights = pd.DataFrame(np.r_[0, 0.5**np.arange(3)]).T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 4)))

    # Run unit tests
    check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
@pytest.mark.parametrize('which', ['exog', 'trend'])
def test_sarimax_time_varying(revisions, updates, which):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e. time-varying state space matrices). It tests a time-varying
    # SARIMAX model where the time-varying component has been set to zeros
    # against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta['infl'].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        endog1.iloc[-1] = 0.

    exog1 = None
    exog2 = None
    trend = 'n'
    if which == 'exog':
        exog1 = np.ones_like(endog1)
        exog2 = np.ones_like(endog2)
    elif which == 'trend':
        trend = 't'

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = sarimax.SARIMAX(endog1, exog=exog1, trend=trend)
    res1 = mod1.smooth([0., 0.5, 1.0])
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = sarimax.SARIMAX(endog1)
    res2 = mod2.smooth([0.5, 1.0])
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news',
             'weights', 'update_forecasts', 'update_realized',
             'prev_impacted_forecasts', 'post_impacted_forecasts',
             'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_unobserved_components_time_varying(revisions, updates):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e. time-varying state space matrices). It tests a time-varying
    # UnobservedComponents model where the time-varying component has been set
    # to zeros against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta['infl'].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        endog1.iloc[-1] = 0.

    exog1 = np.ones_like(endog1)
    exog2 = np.ones_like(endog2)

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = structural.UnobservedComponents(endog1, 'llevel', exog=exog1)
    res1 = mod1.smooth([0.5, 0.2, 0.0])
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = structural.UnobservedComponents(endog1, 'llevel')
    res2 = mod2.smooth([0.5, 0.2])
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news',
             'weights', 'update_forecasts', 'update_realized',
             'prev_impacted_forecasts', 'post_impacted_forecasts',
             'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_varmax_time_invariant(revisions, updates):
    # Construct previous and updated datasets
    endog = dta[['realgdp', 'unemp']].copy()
    endog['realgdp'] = np.log(endog['realgdp']).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod = varmax.VARMAX(endog1, trend='n')
    params = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
    res = mod.smooth(params)
    news = res.news(endog2, start='2009Q2', end='2010Q1',
                    comparison_type=comparison_type)

    # Compute the true values for each combination of (revsions, updates)
    impact_dates = pd.period_range(start='2009Q2', end='2010Q1', freq='Q')
    impacted_variables = ['realgdp', 'unemp']

    # Matrix powers of transition matrix
    Z = np.zeros((2, 2))
    T0 = np.eye(2)
    T1 = mod['transition']
    T2 = T1 @ T1
    T3 = T1 @ T2

    # Revisions
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_product(
            [endog1.index[-1:], ['realgdp', 'unemp']],
            names=['revision date', 'revised variable'])
        # If we have updates, the revision is to 2009Q2
        # Note: this ".values" and all of those below are only required for
        # Pandas = 0.23, and can be removed once that is no longer a supported
        # dependency
        tmp = endog2.iloc[-2].values
        revision_impacts = np.c_[T0 @ tmp, T1 @ tmp, T2 @ tmp, T3 @ tmp].T
    elif revisions:
        revisions_index = pd.MultiIndex.from_product(
            [endog1.index[-1:], ['realgdp', 'unemp']],
            names=['revision date', 'revised variable'])
        # With no updates, the revision is to 2009Q3
        tmp = endog2.iloc[-1].values
        revision_impacts = np.c_[Z @ tmp, T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    else:
        revisions_index = pd.MultiIndex.from_product(
            [[], []],
            names=['revision date', 'revised variable'])
        revision_impacts = None

    # Impact forecasts
    if updates:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp,
                                        T2 @ tmp, T3 @ tmp].T
        tmp = endog2.iloc[-2].values
        rev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp,
                                       T2 @ tmp, T3 @ tmp].T
    else:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[
            T0 @ endog1.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
        tmp = endog2.iloc[-1].values
        rev_impacted_forecasts = np.c_[
            T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    tmp = endog2.iloc[-1].values
    post_impacted_forecasts = np.c_[
        T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T

    # Updates
    if updates:
        updates_index = pd.MultiIndex.from_product(
            [pd.period_range(start='2009Q3', periods=1, freq='Q'),
             ['realgdp', 'unemp']], names=['update date', 'updated variable'])
        update_impacts = post_impacted_forecasts - rev_impacted_forecasts
    else:
        updates_index = pd.MultiIndex.from_product(
            [[], []], names=['update date', 'updated variable'])
        update_impacts = None

    # News
    if updates:
        # Note: update_forecasts is created using the endog2 dataset even if
        # there were revisions, because it should be computed after revisions
        # have already been taken into account
        update_forecasts = T1 @ endog2.loc['2009Q2'].values
        update_realized = endog2.loc['2009Q3'].values
        news_desired = [update_realized[i] - update_forecasts[i]
                        for i in range(len(update_forecasts))]
        columns = pd.MultiIndex.from_product(
            [impact_dates, impacted_variables],
            names=['impact dates', 'impacted variables'])
        weights = pd.DataFrame(np.zeros((2, 8)), index=updates_index,
                               columns=columns)
        weights.loc[:, '2009Q2'] = Z
        weights.loc[:, '2009Q3'] = T0
        weights.loc[:, '2009Q4'] = T1.T
        weights.loc[:, '2010Q1'] = T2.T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 8)))

    # Run unit tests
    check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
@pytest.mark.parametrize('which', ['exog', 'trend'])
def test_varmax_time_varying(revisions, updates, which):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e. time-varying state space matrices). It tests a time-varying
    # VARMAX model where the time-varying component has been set to zeros
    # against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta[['realgdp', 'unemp']].copy()
    endog['realgdp'] = np.log(endog['realgdp']).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    exog1 = None
    exog2 = None
    trend = 'n'
    if which == 'exog':
        exog1 = np.ones_like(endog1['realgdp'])
        exog2 = np.ones_like(endog2['realgdp'])
        params1 = np.r_[0.5, 0.1, 0.2, 0.9, 0., 0., 1., 0.1, 1.1]
        params2 = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
    elif which == 'trend':
        trend = 't'
        params1 = np.r_[0., 0., 0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
        params2 = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = varmax.VARMAX(endog1, exog=exog1, trend=trend)
    res1 = mod1.smooth(params1)
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = varmax.VARMAX(endog1, trend='n')
    res2 = mod2.smooth(params2)
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news',
             'weights', 'update_forecasts', 'update_realized',
             'prev_impacted_forecasts', 'post_impacted_forecasts',
             'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_dynamic_factor_time_varying(revisions, updates):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e. time-varying state space matrices). It tests a time-varying
    # DynamicFactor model where the time-varying component has been set to
    # zeros against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta[['realgdp', 'unemp']].copy()
    endog['realgdp'] = np.log(endog['realgdp']).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = 'updated'
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    exog1 = np.ones_like(endog1['realgdp'])
    exog2 = np.ones_like(endog2['realgdp'])
    params1 = np.r_[0.9, 0.2, 0.0, 0.0, 1.2, 1.1, 0.5, 0.2]
    params2 = np.r_[0.9, 0.2, 1.2, 1.1, 0.5, 0.2]

    # Compute the news from a model with an exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = dynamic_factor.DynamicFactor(endog1, exog=exog1,
                                        k_factors=1, factor_order=2)
    res1 = mod1.smooth(params1)
    news1 = res1.news(endog2, exog=exog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = dynamic_factor.DynamicFactor(endog1, k_factors=1, factor_order=2)
    res2 = mod2.smooth(params2)
    news2 = res2.news(endog2, start='2008Q1', end='2009Q3',
                      comparison_type=comparison_type)

    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news',
             'weights', 'update_forecasts', 'update_realized',
             'prev_impacted_forecasts', 'post_impacted_forecasts',
             'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize('revisions', [True, False])
@pytest.mark.parametrize('updates', [True, False])
def test_defaults(revisions, updates):
    # Construct previous and updated datasets
    endog = dta['infl'].copy()
    if updates:
        endog1 = endog.loc[:'2009Q2'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    else:
        endog1 = endog.loc[:'2009Q3'].copy()
        endog2 = endog.loc[:'2009Q3'].copy()
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod1 = sarimax.SARIMAX(endog1)
    res1 = mod1.smooth([0.5, 1.0])

    mod2 = sarimax.SARIMAX(endog2)
    res2 = mod2.smooth([0.5, 1.0])

    news_updated_data = res1.news(endog2, comparison_type='updated')
    news_previous_data = res2.news(endog1, comparison_type='previous')
    news_updated_results = res1.news(res2, comparison_type='updated')
    news_previous_results = res2.news(res1, comparison_type='previous')

    attrs = ['total_impacts', 'update_impacts', 'revision_impacts', 'news',
             'weights', 'update_forecasts', 'update_realized',
             'prev_impacted_forecasts', 'post_impacted_forecasts',
             'revisions_iloc', 'revisions_ix', 'updates_iloc', 'updates_ix']

    for attr in attrs:
        w = getattr(news_updated_data, attr)
        x = getattr(news_previous_data, attr)
        y = getattr(news_updated_results, attr)
        z = getattr(news_previous_results, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
            assert_series_equal(w, y)
            assert_series_equal(w, z)
        else:
            assert_frame_equal(w, x)
            assert_frame_equal(w, y)
            assert_frame_equal(w, z)


def test_comparison_types():
    endog = dta['infl'].copy()
    endog.iloc[-1] = np.nan
    msg = 'Could not automatically determine the type of comparison'

    mod = sarimax.SARIMAX(endog)
    res = mod.smooth([0.5, 1.0])

    # Test that the comparison type cannot be determined if the given dataset
    # or results dataset is identical to the previous dataset
    with pytest.raises(ValueError, match=msg):
        res.news(endog)
    with pytest.raises(ValueError, match=msg):
        res.news(res)

    # Test that if the the comparison type is specified, the news can be
    # computed from data that has the same shape and number of NaNs (this can
    # happen if there are only revisions but no updates)
    news = res.news(endog, comparison_type='previous')
    assert_allclose(news.total_impacts, 0)
    news = res.news(endog, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type='updated')
    assert_allclose(news.total_impacts, 0)


@pytest.mark.parametrize('use_periods', [True, False])
def test_start_end_dates(use_periods):
    endog = dta['infl'].copy()
    if use_periods:
        index_range = pd.period_range
    else:
        def index_range(*args, **kwargs):
            return pd.period_range(*args, **kwargs).to_timestamp(freq='Q')
        endog = endog.to_timestamp(freq='Q')
    mod = sarimax.SARIMAX(endog.iloc[:-1])
    res = mod.smooth([0.5, 1.0])

    # Default is the first out-of-sample period
    news = res.news(endog)
    desired = index_range(start='2009Q2', periods=1, freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), periods
    news = res.news(endog, start='2009Q1', periods=1)
    desired = index_range(start='2009Q1', periods=1, freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start='2009Q1', periods=2)
    desired = index_range(start='2009Q1', periods=2, freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # Start (int), periods
    news = res.news(endog, start=mod.nobs - 1, periods=1)
    desired = index_range(start='2009Q2', periods=1, freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start=mod.nobs - 2, periods=2)
    desired = index_range(start='2009Q1', periods=2, freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # End (dates), periods
    news = res.news(endog, end='2009Q1', periods=1)
    desired = index_range(end='2009Q1', periods=1, freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end='2009Q1', periods=2)
    desired = index_range(end='2009Q1', periods=2, freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # End (int), periods
    news = res.news(endog, end=mod.nobs - 1, periods=1)
    desired = index_range(end='2009Q2', periods=1, freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end=mod.nobs - 2, periods=2)
    desired = index_range(end='2009Q1', periods=2, freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), end (dates)
    news = res.news(endog, start='2009Q1', end='2009Q1')
    desired = index_range(start='2009Q1', end='2009Q1', freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start='2009Q1', end='2009Q2')
    desired = index_range(start='2009Q1', end='2009Q2', freq='Q')
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), end (int)
    news = res.news(endog, start='2009Q1', end=mod.nobs - 2)
    desired = index_range(start='2009Q1', end='2009Q1', freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start='2009Q1', end=mod.nobs - 2)
    assert_(news.total_impacts.index.equals(predicted.index))
    news = res.news(endog, start='2009Q1', end=mod.nobs - 1)
    desired = index_range(start='2009Q1', end='2009Q2', freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start='2009Q1', end=mod.nobs - 1)
    assert_(news.total_impacts.index.equals(predicted.index))

    # Start (int), end (dates)
    news = res.news(endog, start=mod.nobs - 2, end='2009Q1')
    desired = index_range(start='2009Q1', end='2009Q1', freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start=mod.nobs - 2, end='2009Q1')
    assert_(news.total_impacts.index.equals(predicted.index))
    news = res.news(endog, start=mod.nobs - 2, end='2009Q2')
    desired = index_range(start='2009Q1', end='2009Q2', freq='Q')
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start=mod.nobs - 2, end='2009Q2')
    assert_(news.total_impacts.index.equals(predicted.index))

    # Negative indexes
    # Note that negative indexes are always computed relative to the updated
    # sample, which in this case is 1 observation more than is in `mod.nobs`
    total_nobs = len(endog)
    assert_equal(total_nobs, mod.nobs + 1)

    # Start (dates), end (int)
    desired = index_range(start='2009Q1', end='2009Q1', freq='Q')
    for end in [mod.nobs - 2, total_nobs - 3, -3]:
        news = res.news(endog, start='2009Q1', end=end)
        assert_(news.total_impacts.index.equals(desired))
        # Note: predict does not allow negative indexing
        if end > 0:
            predicted = res.predict(start='2009Q1', end=end)
            assert_(news.total_impacts.index.equals(predicted.index))

    # Start (int), end (dates)
    desired = index_range(start='2009Q1', end='2009Q1', freq='Q')
    for start in [mod.nobs - 2, total_nobs - 3, -3]:
        news = res.news(endog, start=start, end='2009Q1')
        assert_(news.total_impacts.index.equals(desired))
        # Note: predict does not allow negative indexing
        if end > 0:
            predicted = res.predict(start=start, end='2009Q1')
            assert_(news.total_impacts.index.equals(predicted.index))


@pytest.mark.parametrize('which', ['range', 'range2', 'int64',
                                   'numpy', 'list'])
def test_start_end_int(which):
    endog = dta['infl'].copy()
    nobs = len(endog)
    if which == 'range':
        endog.index = pd.RangeIndex(nobs)
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)
    if which == 'range2':
        endog.index = pd.RangeIndex(stop=nobs * 2, step=2)
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.RangeIndex((nobs + 2) * 2, step=2)
    elif which == 'int64':
        endog.index = pd.Index(np.arange(nobs))
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.Index(np.arange(nobs + 2))
    elif which == 'numpy':
        endog = endog.values
        endog_init = endog[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)
    elif which == 'list':
        endog = endog.tolist()
        endog_init = endog[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)

    mod = sarimax.SARIMAX(endog_init)
    res = mod.smooth([0.5, 1.0])

    # Default is the last in-sample period
    news = res.news(endog)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))

    # Start, periods
    news = res.news(endog, start=mod.nobs - 1, periods=1)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start=mod.nobs - 2, periods=2)
    desired = index_plus2[-5:-3]
    assert_(news.total_impacts.index.equals(desired))

    # End, periods
    news = res.news(endog, end=mod.nobs - 1, periods=1)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end=mod.nobs - 2, periods=2)
    desired = index_plus2[-6:-4]
    assert_(news.total_impacts.index.equals(desired))

    # Start, end
    # Note: end is inclusive, like `get_prediction`.
    news = res.news(endog, start=mod.nobs - 2, end=mod.nobs - 1)
    desired = index_plus2[-5:-3]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ['numpy', 'list']:
        predicted = res.predict(start=mod.nobs - 2, end=mod.nobs - 1)
        assert_(news.total_impacts.index.equals(predicted.index))

    news = res.news(endog, start=mod.nobs, end=mod.nobs)
    desired = index_plus2[-3:-2]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ['numpy', 'list']:
        predicted = res.predict(start=mod.nobs, end=mod.nobs)
        assert_(news.total_impacts.index.equals(predicted.index))

    news = res.news(endog, start=mod.nobs, end=mod.nobs + 1)
    desired = index_plus2[-3:-1]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ['numpy', 'list']:
        predicted = res.predict(start=mod.nobs, end=mod.nobs + 1)
        assert_(news.total_impacts.index.equals(predicted.index))


def test_invalid():
    endog = dta['infl'].copy()
    mod = sarimax.SARIMAX(endog.iloc[:-1])
    res = mod.smooth([0.5, 1.0])

    msg = 'The index associated with the updated results is not a superset'
    with pytest.raises(ValueError, match=msg):
        res.news(endog.values)
