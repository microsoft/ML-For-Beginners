import os
import numpy as np
import pandas as pd
from scipy.io import matlab

import pytest
from numpy.testing import assert_allclose

from statsmodels.tsa.statespace import initialization, dynamic_factor_mq

# Load dataset
current_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_path, 'results', 'frbny_nowcast')
data_path = os.path.join(results_path, 'Nowcasting', 'data', 'US')
us_data = pd.read_csv(os.path.join(data_path, '2016-06-29.csv'))
us_data.index = pd.PeriodIndex(us_data.Date.tolist(), freq='M')
del us_data['Date']
us_data_update = pd.read_csv(os.path.join(data_path, '2016-07-29.csv'))
us_data_update.index = pd.PeriodIndex(us_data_update.Date.tolist(), freq='M')
del us_data_update['Date']

# Some definitions to re-use
BLOCK_FACTORS_KP1 = {
    'CPIAUCSL': ['global', 'test1'],
    'UNRATE': ['global', 'test2'],
    'PAYEMS': ['global', 'test2'],
    'RSAFS': ['global', 'test1', 'test2'],
    'TTLCONS': ['global', 'test1'],
    'TCU': ['global', 'test2'],
    'GDPC1': ['global', 'test1', 'test2'],
    'ULCNFB': ['global'],
}
BLOCK_FACTOR_ORDERS_KP1 = {
    'global': 1,
    'test1': 1,
    'test2': 1
}
BLOCK_FACTOR_ORDERS_KP2 = {
    'global': 2,
    'test1': 2,
    'test2': 2
}
BLOCK_FACTOR_MULTIPLICITIES_KP2 = {
    'global': 2,
    'test1': 2,
    'test2': 2
}


@pytest.fixture(scope="module")
def matlab_results():
    # Get estimation output from FRBNY programs
    results = {}

    # DFM results with a single block of factors
    for run in ['111', '112', '11F', '221', '222', '22F']:
        res = matlab.loadmat(os.path.join(results_path, f'test_dfm_{run}.mat'))

        # The FRBNY version orders the idiosyncratic AR(1) factors differently,
        # so we need to re-order the initial state mean and covariance matrix
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_orders = res['Spec']['p'][0, 0][0, 0]
        _factor_orders = max(5, factor_orders)

        idio = k_factors * _factor_orders + 3
        ix = np.r_[np.arange(idio),
                   idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]

        # In the 2-factor case, we want both factors in a single block
        if k_factors == 2:
            factor_orders = {('0', '1'): factor_orders}

        results[run] = {
            'k_endog_M': 3,
            'factors': k_factors,
            'factor_orders': factor_orders,
            'factor_multiplicities': None,
            'params': res['params'][:, 0],
            'llf': res['Res']['loglik'][0, 0][0, 0],
            'initial_state': initial_state,
            'initial_state_cov': initial_state_cov,
            'smoothed_forecasts': res['Res']['x_sm'][0, 0]}

    # News output with a single block of factors
    for run in ['112', '222']:
        res = matlab.loadmat(os.path.join(results_path,
                                          f'test_news_{run}.mat'))

        # The FRBNY version orders the idiosyncratic AR(1) factors differently,
        # so we need to re-order the initial state mean and covariance matrix
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_orders = res['Spec']['p'][0, 0][0, 0]
        _factor_orders = max(5, factor_orders)

        idio = k_factors * _factor_orders + 3
        ix = np.r_[np.arange(idio),
                   idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]

        # In the 2-factor case, we want both factors in a single block
        if k_factors == 2:
            factor_orders = {('0', '1'): factor_orders}

        results[f'news_{run}'] = {
            'k_endog_M': 3,
            'factors': k_factors,
            'factor_orders': factor_orders,
            'factor_multiplicities': None,
            'params': res['params'][:, 0],
            'initial_state': initial_state,
            'initial_state_cov': initial_state_cov,
            'revision_impacts': res['Res']['impact_revisions'][0, 0],
            'weight': res['Res']['weight'],
            'news_table': res['Res']['news_table'][0, 0]}

    # DFM results with three blocks of factors
    for run in ['111', '112', '221', '222']:
        res = matlab.loadmat(os.path.join(results_path,
                                          f'test_dfm_blocks_{run}.mat'))

        # The FRBNY version orders the idiosyncratic AR(1) factors differently,
        # so we need to re-order the initial state mean and covariance matrix
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_order = res['Spec']['p'][0, 0][0, 0]
        _factor_order = max(5, factor_order)

        idio = 3 * k_factors * _factor_order + 6
        ix = np.r_[np.arange(idio),
                   idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]

        # Setup factors / blocks
        if k_factors == 1:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP1.copy()
            factor_multiplicities = None
        else:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP2.copy()
            factor_multiplicities = BLOCK_FACTOR_MULTIPLICITIES_KP2.copy()

        results[f'block_{run}'] = {
            'k_endog_M': 6,
            'factors': factors,
            'factor_orders': factor_orders,
            'factor_multiplicities': factor_multiplicities,
            'params': res['params'][:, 0],
            'llf': res['Res']['loglik'][0, 0][0, 0],
            'initial_state': initial_state,
            'initial_state_cov': initial_state_cov,
            'smoothed_forecasts': res['Res']['x_sm'][0, 0]}

    # News output with three blocks of factors
    for run in ['112', '222']:
        res = matlab.loadmat(os.path.join(results_path,
                                          f'test_news_blocks_{run}.mat'))

        # The FRBNY version orders the idiosyncratic AR(1) factors differently,
        # so we need to re-order the initial state mean and covariance matrix
        k_factors = res['Spec']['k'][0, 0][0, 0]
        factor_order = res['Spec']['p'][0, 0][0, 0]
        _factor_order = max(5, factor_order)

        idio = 3 * k_factors * _factor_order + 6
        ix = np.r_[np.arange(idio),
                   idio + np.arange(10).reshape(2, 5).ravel(order='F')]
        initial_state = res['Res']['Z_0'][0, 0][:, 0][ix]
        initial_state_cov = res['Res']['V_0'][0, 0][ix][:, ix]

        # Setup factors / blocks
        if k_factors == 1:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP1.copy()
            factor_multiplicities = None
        else:
            factors = BLOCK_FACTORS_KP1.copy()
            factor_orders = BLOCK_FACTOR_ORDERS_KP2.copy()
            factor_multiplicities = BLOCK_FACTOR_MULTIPLICITIES_KP2.copy()

        results[f'news_block_{run}'] = {
            'k_endog_M': 6,
            'factors': factors,
            'factor_orders': factor_orders,
            'factor_multiplicities': factor_multiplicities,
            'params': res['params'][:, 0],
            'initial_state': initial_state,
            'initial_state_cov': initial_state_cov,
            'revision_impacts': res['Res']['impact_revisions'][0, 0],
            'weight': res['Res']['weight'],
            'news_table': res['Res']['news_table'][0, 0]}

    # Construct the test dataset
    def get_data(us_data, mean_M=None, std_M=None, mean_Q=None, std_Q=None):
        dta_M = us_data[['CPIAUCSL', 'UNRATE', 'PAYEMS', 'RSAFS', 'TTLCONS',
                         'TCU']].copy()
        dta_Q = us_data[['GDPC1', 'ULCNFB']].copy().resample('Q').last()

        dta_M['CPIAUCSL'] = (dta_M['CPIAUCSL'] /
                             dta_M['CPIAUCSL'].shift(1) - 1) * 100
        dta_M['UNRATE'] = dta_M['UNRATE'].diff()
        dta_M['PAYEMS'] = dta_M['PAYEMS'].diff()
        dta_M['TCU'] = dta_M['TCU'].diff()
        dta_M['RSAFS'] = (dta_M['RSAFS'] /
                          dta_M['RSAFS'].shift(1) - 1) * 100
        dta_M['TTLCONS'] = (dta_M['TTLCONS'] /
                            dta_M['TTLCONS'].shift(1) - 1) * 100
        dta_Q = ((dta_Q / dta_Q.shift(1))**4 - 1) * 100

        start = '2000'
        dta_M = dta_M.loc[start:]
        dta_Q = dta_Q.loc[start:]

        first_ix = dta_M.first_valid_index()
        last_ix = dta_M.last_valid_index()
        dta_M = dta_M.loc[first_ix:last_ix]

        first_ix = dta_Q.first_valid_index()
        last_ix = dta_Q.last_valid_index()
        dta_Q = dta_Q.loc[first_ix:last_ix]

        return dta_M, dta_Q

    # Usual test dataset
    endog_M, endog_Q = get_data(us_data)
    # Updated test dataset (for computing the news)
    updated_M, updated_Q = get_data(us_data_update)

    return endog_M, endog_Q, results, updated_M, updated_Q


@pytest.mark.parametrize("run", ['111', '112', '11F', '221', '222', '22F',
                                 'block_111', 'block_112', 'block_221',
                                 'block_222'])
def test_known(matlab_results, run):
    endog_M, endog_Q = matlab_results[:2]
    results = matlab_results[2][run]

    # Construct the model
    mod = dynamic_factor_mq.DynamicFactorMQ(
            endog_M.iloc[:, :results['k_endog_M']], endog_quarterly=endog_Q,
            factors=results['factors'],
            factor_orders=results['factor_orders'],
            factor_multiplicities=results['factor_multiplicities'],
            idiosyncratic_ar1=True, init_t0=True, obs_cov_diag=True,
            standardize=True)

    mod.initialize_known(results['initial_state'],
                         results['initial_state_cov'])
    res = mod.smooth(results['params'], cov_type='none')
    assert_allclose(res.llf - mod.loglike_constant, results['llf'])
    assert_allclose(res.filter_results.smoothed_forecasts.T[1:],
                    results['smoothed_forecasts'][:-1])
    assert_allclose(res.forecast(1, original_scale=False).iloc[0],
                    results['smoothed_forecasts'][-1])


@pytest.mark.parametrize("run", ['11', '22', 'block_11', 'block_22'])
def test_emstep1(matlab_results, run):
    # Test that our EM step gets params2 from params1
    # Uses our default method for the observation equation, which is an
    # optimized version of the method presented in Bańbura and Modugno (2014)
    # (e.g. our version doesn't require the loop over T or the Kronecker
    # product)
    endog_M, endog_Q = matlab_results[:2]
    results1 = matlab_results[2][f'{run}1']
    results2 = matlab_results[2][f'{run}2']

    # Construct the model
    mod = dynamic_factor_mq.DynamicFactorMQ(
            endog_M.iloc[:, :results1['k_endog_M']], endog_quarterly=endog_Q,
            factors=results1['factors'],
            factor_orders=results1['factor_orders'],
            factor_multiplicities=results1['factor_multiplicities'],
            idiosyncratic_ar1=True, init_t0=True, obs_cov_diag=True,
            standardize=True)

    init = initialization.Initialization(
        mod.k_states, 'known', constant=results1['initial_state'],
        stationary_cov=results1['initial_state_cov'])
    res2, params2 = mod._em_iteration(results1['params'], init=init,
                                      mstep_method='missing')

    # Test parameters
    true2 = results2['params']
    assert_allclose(params2[mod._p['loadings']], true2[mod._p['loadings']])
    assert_allclose(params2[mod._p['factor_ar']], true2[mod._p['factor_ar']])
    assert_allclose(params2[mod._p['factor_cov']], true2[mod._p['factor_cov']])
    assert_allclose(params2[mod._p['idiosyncratic_ar1']],
                    true2[mod._p['idiosyncratic_ar1']])
    assert_allclose(params2[mod._p['idiosyncratic_var']],
                    true2[mod._p['idiosyncratic_var']])


@pytest.mark.parametrize(
    "k_factors,factor_orders,factor_multiplicities,idiosyncratic_ar1",
    [(1, 1, 1, True), (3, 1, 1, True), (1, 6, 1, True), (3, 6, 1, True),
     (1, 1, 1, False), (3, 1, 1, False), (1, 6, 1, False), (3, 6, 1, False),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, True),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, False),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(),
        BLOCK_FACTOR_MULTIPLICITIES_KP2, True),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(),
        BLOCK_FACTOR_MULTIPLICITIES_KP2, False)])
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_emstep_methods_missing(matlab_results, k_factors, factor_orders,
                                factor_multiplicities, idiosyncratic_ar1):
    # Test that in the case of missing data, the direct and optimized EM step
    # methods for the observation equation give identical results across a
    # variety of parameterizations
    endog_M = matlab_results[0].iloc[:, :10]
    endog_Q = matlab_results[1].iloc[:, :10]

    # Construct the model
    mod = dynamic_factor_mq.DynamicFactorMQ(
            endog_M, endog_quarterly=endog_Q, factors=k_factors,
            factor_orders=factor_orders,
            factor_multiplicities=factor_multiplicities,
            idiosyncratic_ar1=idiosyncratic_ar1, standardize=True)
    mod.ssm.filter_univariate = True

    params0 = mod.start_params
    _, params1 = mod._em_iteration(params0, mstep_method='missing')

    # Now double-check the observation equation M step for identical H and
    # Lambda directly
    mod.update(params1)
    res = mod.ssm.smooth()

    a = res.smoothed_state.T[..., None]
    cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
    Eaa = cov_a + np.matmul(a, a.transpose(0, 2, 1))

    Lambda, H = mod._em_maximization_obs_missing(res, Eaa, a, compute_H=True)


@pytest.mark.parametrize(
    "k_factors,factor_orders,factor_multiplicities,idiosyncratic_ar1",
    [(1, 1, 1, True), (3, 1, 1, True), (1, 6, 1, True),
     (3, {('0', '1', '2'): 6}, 1, True),
     (1, 1, 1, False), (3, 1, 1, False), (1, 6, 1, False),
     (3, {('0', '1', '2'): 6}, 1, False),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, True),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP1.copy(), 1, False),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(),
        BLOCK_FACTOR_MULTIPLICITIES_KP2, True),
     (BLOCK_FACTORS_KP1.copy(), BLOCK_FACTOR_ORDERS_KP2.copy(),
        BLOCK_FACTOR_MULTIPLICITIES_KP2, False)])
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_emstep_methods_nonmissing(matlab_results, k_factors, factor_orders,
                                   factor_multiplicities, idiosyncratic_ar1):
    # Test that in the case of non-missing data, our three EM step methods for
    # the observation equation (nonmissing, missing_direct, missing) give
    # identical results across a variety of parameterizations
    # Note that including quarterly series will always imply missing values,
    # so we have to only provide monthly series
    dta_M = matlab_results[0].iloc[:, :8]
    dta_M = (dta_M - dta_M.mean()) / dta_M.std()
    endog_M = dta_M.interpolate().bfill()

    # Remove the quarterly endog->factor maps
    if isinstance(k_factors, dict):
        if 'GDPC1' in k_factors:
            del k_factors['GDPC1']
        if 'ULCNFB' in k_factors:
            del k_factors['ULCNFB']

    # Construct the model
    mod = dynamic_factor_mq.DynamicFactorMQ(
        endog_M, factors=k_factors, factor_orders=factor_orders,
        factor_multiplicities=factor_multiplicities,
        idiosyncratic_ar1=idiosyncratic_ar1)
    mod.ssm.filter_univariate = True

    params0 = mod.start_params
    _, params1 = mod._em_iteration(params0, mstep_method='missing')
    _, params1_nonmissing = mod._em_iteration(
        params0, mstep_method='nonmissing')

    assert_allclose(params1_nonmissing, params1, atol=1e-13)

    # Now double-check the observation equation M step for identical H and
    # Lambda directly
    mod.update(params1)
    res = mod.ssm.smooth()

    a = res.smoothed_state.T[..., None]
    cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
    Eaa = cov_a + np.matmul(a, a.transpose(0, 2, 1))

    Lambda, H = mod._em_maximization_obs_missing(res, Eaa, a, compute_H=True)
    Lambda_nonmissing, H_nonmissing = mod._em_maximization_obs_nonmissing(
        res, Eaa, a, compute_H=True)

    assert_allclose(Lambda_nonmissing, Lambda, atol=1e-13)
    assert_allclose(H_nonmissing, H, atol=1e-13)


@pytest.mark.parametrize("run", ['news_112', 'news_222', 'news_block_112',
                                 'news_block_222'])
def test_news(matlab_results, run):
    endog_M, endog_Q = matlab_results[:2]
    results = matlab_results[2][run]
    updated_M, updated_Q = matlab_results[-2:]

    # Construct the base model
    mod1 = dynamic_factor_mq.DynamicFactorMQ(
        endog_M.iloc[:, :results['k_endog_M']], endog_quarterly=endog_Q,
        factors=results['factors'],
        factor_orders=results['factor_orders'],
        factor_multiplicities=results['factor_multiplicities'],
        idiosyncratic_ar1=True, init_t0=True, obs_cov_diag=True,
        standardize=True)

    mod1.initialize_known(results['initial_state'],
                          results['initial_state_cov'])
    res1 = mod1.smooth(results['params'], cov_type='none')

    # Construct the updated model
    res2 = res1.apply(updated_M.iloc[:, :results['k_endog_M']],
                      endog_quarterly=updated_Q, retain_standardization=True)

    # Compute the news
    news = res2.news(res1, impact_date='2016-09', comparison_type='previous')

    assert_allclose(news.revision_impacts.loc['2016-09', 'GDPC1'],
                    results['revision_impacts'])

    columns = ['forecast (prev)', 'observed', 'weight', 'impact']
    actual = news.details_by_impact.loc['2016-09', 'GDPC1'][columns]
    assert_allclose(actual.loc[('2016-06', 'CPIAUCSL')],
                    results['news_table'][0])
    assert_allclose(actual.loc[('2016-06', 'UNRATE')],
                    results['news_table'][1])
    assert_allclose(actual.loc[('2016-06', 'PAYEMS')],
                    results['news_table'][2])
    if mod1.k_endog_M == 6:
        i = 6
        assert_allclose(actual.loc[('2016-06', 'RSAFS')],
                        results['news_table'][3])
        assert_allclose(actual.loc[('2016-05', 'TTLCONS')],
                        results['news_table'][4])
        assert_allclose(actual.loc[('2016-06', 'TCU')],
                        results['news_table'][5])
    else:
        i = 3
    assert_allclose(actual.loc[('2016-06', 'GDPC1')],
                    results['news_table'][i])
