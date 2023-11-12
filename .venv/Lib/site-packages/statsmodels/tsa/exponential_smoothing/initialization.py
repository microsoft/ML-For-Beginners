"""
Initialization methods for states of exponential smoothing models
"""

import numpy as np
import pandas as pd


def _initialization_simple(endog, trend=False, seasonal=False,
                           seasonal_periods=None):
    # See Section 7.6 of Hyndman and Athanasopoulos
    nobs = len(endog)
    initial_trend = None
    initial_seasonal = None

    # Non-seasonal
    if seasonal is None or not seasonal:
        initial_level = endog[0]
        if trend == 'add':
            initial_trend = endog[1] - endog[0]
        elif trend == 'mul':
            initial_trend = endog[1] / endog[0]
    # Seasonal
    else:
        if nobs < 2 * seasonal_periods:
            raise ValueError('Cannot compute initial seasonals using'
                             ' heuristic method with less than two full'
                             ' seasonal cycles in the data.')

        initial_level = np.mean(endog[:seasonal_periods])
        m = seasonal_periods

        if trend is not None:
            initial_trend = (pd.Series(endog).diff(m)[m:2 * m] / m).mean()

        if seasonal == 'add':
            initial_seasonal = endog[:m] - initial_level
        elif seasonal == 'mul':
            initial_seasonal = endog[:m] / initial_level

    return initial_level, initial_trend, initial_seasonal


def _initialization_heuristic(endog, trend=False, seasonal=False,
                              seasonal_periods=None):
    # See Section 2.6 of Hyndman et al.
    endog = endog.copy()
    nobs = len(endog)

    if nobs < 10:
        raise ValueError('Cannot use heuristic method with less than 10'
                         ' observations.')

    # Seasonal component
    initial_seasonal = None
    if seasonal:
        # Calculate the number of full cycles to use
        if nobs < 2 * seasonal_periods:
            raise ValueError('Cannot compute initial seasonals using'
                             ' heuristic method with less than two full'
                             ' seasonal cycles in the data.')
        # We need at least 10 periods for the level initialization
        # and we will lose self.seasonal_periods // 2 values at the
        # beginning and end of the sample, so we need at least
        # 10 + 2 * (self.seasonal_periods // 2) values
        min_obs = 10 + 2 * (seasonal_periods // 2)
        if nobs < min_obs:
            raise ValueError('Cannot use heuristic method to compute'
                             ' initial seasonal and levels with less'
                             ' than 10 + 2 * (seasonal_periods // 2)'
                             ' datapoints.')
        # In some datasets we may only have 2 full cycles (but this may
        # still satisfy the above restriction that we will end up with
        # 10 seasonally adjusted observations)
        k_cycles = min(5, nobs // seasonal_periods)
        # In other datasets, 3 full cycles may not be enough to end up
        # with 10 seasonally adjusted observations
        k_cycles = max(k_cycles, int(np.ceil(min_obs / seasonal_periods)))

        # Compute the moving average
        series = pd.Series(endog[:seasonal_periods * k_cycles])
        initial_trend = series.rolling(seasonal_periods, center=True).mean()
        if seasonal_periods % 2 == 0:
            initial_trend = initial_trend.shift(-1).rolling(2).mean()

        # Detrend
        if seasonal == 'add':
            detrended = series - initial_trend
        elif seasonal == 'mul':
            detrended = series / initial_trend

        # Average seasonal effect
        tmp = np.zeros(k_cycles * seasonal_periods) * np.nan
        tmp[:len(detrended)] = detrended.values
        initial_seasonal = np.nanmean(
            tmp.reshape(k_cycles, seasonal_periods).T, axis=1)

        # Normalize the seasonals
        if seasonal == 'add':
            initial_seasonal -= np.mean(initial_seasonal)
        elif seasonal == 'mul':
            initial_seasonal /= np.mean(initial_seasonal)

        # Replace the data with the trend
        endog = initial_trend.dropna().values

    # Trend / Level
    exog = np.c_[np.ones(10), np.arange(10) + 1]
    if endog.ndim == 1:
        endog = np.atleast_2d(endog).T
    beta = np.squeeze(np.linalg.pinv(exog).dot(endog[:10]))
    initial_level = beta[0]

    initial_trend = None
    if trend == 'add':
        initial_trend = beta[1]
    elif trend == 'mul':
        initial_trend = 1 + beta[1] / beta[0]

    return initial_level, initial_trend, initial_seasonal
