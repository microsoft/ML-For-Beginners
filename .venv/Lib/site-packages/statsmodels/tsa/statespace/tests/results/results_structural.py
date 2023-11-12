"""
Results for structural tests

Results from R, KFAS library using script `test_ucm.R`.
See also Stata time series documentation.

Author: Chad Fulton
License: Simplified-BSD
"""
from numpy import pi

LOG2PI = 0.9189385332046727

irregular = {
    'models': [
        {'irregular': True},
        {'level': 'irregular'},
        {'level': 'ntrend'},
    ],
    'params': [36.74687342],
    'start_params': [30],
    'llf': -653.8562525,
    'kwargs': {}
}

# this model will issue a warning that there is no stochastic component, and
# will then add an irregular component. Thus it's output will be just like
# the "deterministic constant" model.
fixed_intercept = {
    'models': [
        {'level': True},
        {'level': 'fixed intercept'},
    ],
    'params': [2.127438969],
    'start_params': [1.],
    'llf': -365.5289923,
    'kwargs': {}
}

deterministic_constant = {
    'models': [
        {'irregular': True, 'level': True},
        {'level': 'deterministic constant'},
        {'level': 'dconstant'},
    ],
    'params': [2.127438969],
    'start_params': [1.],
    'llf': -365.5289923,
    'kwargs': {}
}

local_level = {
    'models': [
        {'irregular': True, 'level': True, 'stochastic_level': True},
        {'level': 'local level'},
        {'level': 'llevel'}
    ],
    'params': [4.256647886e-06, 1.182078808e-01],
    'start_params': [1e-4, 1e-2],
    'llf': -70.97242557,
    'kwargs': {}
}

random_walk = {
    'models': [
        {'level': True, 'stochastic_level': True},
        {'level': 'random walk'},
        {'level': 'rwalk'},
    ],
    'params': [0.1182174646],
    'start_params': [0.5],
    'llf': -70.96771641,
    'kwargs': {}
}


# this model will issue a warning that there is no stochastic component, and
# will then add an irregular component. Thus it's output will be just like
# the "deterministic trend" model.
fixed_slope = {
    'models': [
        {'level': True, 'trend': True},
        {'level': 'fixed slope'},
    ],
    'params': [2.134137554],
    'start_params': [1.],
    'llf': -370.7758666,
    'kwargs': {}
}

deterministic_trend = {
    'models': [
        {'irregular': True, 'level': True, 'trend': True},
        {'level': 'deterministic trend'},
        {'level': 'dtrend'},
    ],
    'params': [2.134137554],
    'start_params': [1.],
    'llf': -370.7758666,
    'kwargs': {}
}

local_linear_deterministic_trend = {
    'models': [
        {'irregular': True, 'level': True, 'stochastic_level': True,
         'trend': True},
        {'level': 'local linear deterministic trend'},
        {'level': 'lldtrend'},
    ],
    'params': [4.457592057e-06, 1.184455029e-01],
    'start_params': [1e-5, 1e-2],
    'llf': -73.47291031,
    'kwargs': {}
}

random_walk_with_drift = {
    'models': [
        {'level': True, 'stochastic_level': True, 'trend': True},
        {'level': 'random walk with drift'},
        {'level': 'rwdrift'},
    ],
    'params': [0.1184499547],
    'start_params': [0.5],
    'llf': -73.46798576,
    'kwargs': {}
}

local_linear_trend = {
    'models': [
        {'irregular': True, 'level': True, 'stochastic_level': True,
         'trend': True, 'stochastic_trend': True},
        {'level': 'local linear trend'},
        {'level': 'lltrend'}
    ],
    'params': [1.339852549e-06, 1.008704925e-02, 6.091760810e-02],
    'start_params': [1e-5, 1e-4, 1e-4],
    'llf': -31.15640107,
    'kwargs': {}
}

smooth_trend = {
    'models': [
        {'irregular': True, 'level': True, 'trend': True,
         'stochastic_trend': True},
        {'level': 'smooth trend'},
        {'level': 'strend'},
    ],
    'params': [0.0008824099119, 0.0753064234342],
    'start_params': [1e-2, 1e-2],
    'llf': -31.92261408,
    'kwargs': {}
}

random_trend = {
    'models': [
        {'level': True, 'trend': True, 'stochastic_trend': True},
        {'level': 'random trend'},
        {'level': 'rtrend'},
    ],
    'params': [0.08054724989],
    'start_params': [1e-2],
    'llf': -32.05607557,
    'kwargs': {}
}

cycle = {
    'models': [{'irregular': True, 'cycle': True, 'stochastic_cycle': True,
                'damped_cycle': True}],
    'params': [37.57197079, 0.1, 2*pi/10, 1],
    'start_params': [30, 0.001, 2*pi/20, 0.5],
    'llf': -656.6568684,
    'kwargs': {}
}

cycle_approx_diffuse = cycle.copy()
cycle_approx_diffuse.update({
    'params': [37.57197224, 0.1, 2*pi/10, 1],
    'llf': -672.3102588,
    'kwargs': {'loglikelihood_burn': 0}})

seasonal = {
    'models': [{'irregular': True, 'seasonal': 4}],
    'params': [38.17043009, 0.1],
    'start_params': [30, 0.001],
    'llf': -655.3337155,
    'kwargs': {},
    'rtol': 1e-6
}

seasonal_approx_diffuse = seasonal.copy()
seasonal_approx_diffuse.update({
    'params': [38.1704278, 0.1],
    'llf': -678.8138005,
    'kwargs': {'loglikelihood_burn': 0}})

freq_seasonal = {
    'models': [{'irregular': True,
                'freq_seasonal': [{'period': 5}]}],
    'params': [38.95352534, 0.05],
    'start_params': [30, 0.001],
    'llf': -657.6629457,
    'rtol': 1e-6,
    'kwargs': {}
}

freq_seasonal_approx_diffuse = freq_seasonal.copy()
freq_seasonal_approx_diffuse.update({
    'params': [38.95353592, 0.05],
    'llf': -688.9697249,
    'kwargs': {'loglikelihood_burn': 0}})

reg = {
    # Note: The test needs to fill in exog=np.log(dta['realgdp'])
    'models': [
        {'irregular': True, 'exog': True, 'mle_regression': False},
        {'level': 'irregular', 'exog': True, 'mle_regression': False},
        {'level': 'ntrend', 'exog': True, 'mle_regression': False},
        {'level': 'ntrend', 'exog': 'numpy', 'mle_regression': False},
    ],
    'params': [2.215438082],
    'start_params': [1.],
    'llf': -371.7966543,
    'kwargs': {}
}

reg_approx_diffuse = reg.copy()
reg_approx_diffuse.update({
    'params': [2.215447924],
    'llf': -379.6233483,
    'kwargs': {'loglikelihood_burn': 0}})

rtrend_ar1 = {
    'models': [
        {'level': True, 'trend': True, 'stochastic_trend': True,
         'autoregressive': 1},
        {'level': 'random trend', 'autoregressive': 1},
        {'level': 'rtrend', 'autoregressive': 1}
    ],
    'params': [0.0609, 0.0097, 0.9592],
    'start_params': [0.01, 0.001, 0.1],
    'llf': -31.15629379,
    'kwargs': {}
}

lltrend_cycle_seasonal_reg_ar1 = {
    # Note: The test needs to fill in exog=np.log(dta['realgdp'])
    'models': [
        # Complete specification
        {'irregular': True, 'level': True, 'stochastic_level': True,
         'trend': True, 'stochastic_trend': True, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False},
    ],
    'params': [0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*pi / 10, 0.2],
    'start_params': [0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*pi / 10, 0.2],
    'llf': -105.8936568,
    'kwargs': {},
    # No need to try to find the maximum for this really weird model.
    'maxiter': 5
}


lltrend_cycle_seasonal_reg_ar1_approx_diffuse = {
    # Note: The test needs to fill in exog=np.log(dta['realgdp'])
    'models': [
        # Complete specification
        {'irregular': True, 'level': True, 'stochastic_level': True,
         'trend': True, 'stochastic_trend': True, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False},
        # Verbose string specification
        {'level': 'local linear trend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False},
        # Abbreviated string specification
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False},
        # Numpy exog dataset
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': 'numpy', 'mle_regression': False},
        # Annual frequency dataset
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False, 'freq': 'AS'},
        # Quarterly frequency dataset
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False, 'freq': 'QS'},
        # Monthly frequency dataset
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False, 'freq': 'MS'},
        # Minutely frequency dataset
        {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
         'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
         'exog': True, 'mle_regression': False, 'freq': 'T',
         'cycle_period_bounds': (1.5*12, 12*12)},
    ],
    'params': [0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*pi / 10, 0.2],
    'start_params': [0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*pi / 10, 0.2],
    'llf': -168.5258709,
    'kwargs': {
        # Required due to the way KFAS estimated loglikelihood which P1inf is
        # set in the R code
        'loglikelihood_burn': 0
    },
    # No need to try to find the maximum for this really weird model.
    'maxiter': 5
}
