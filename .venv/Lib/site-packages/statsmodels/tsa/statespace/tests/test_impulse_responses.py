"""
Tests for impulse responses of time series

Author: Chad Fulton
License: Simplified-BSD
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ortho_group
import pytest
from numpy.testing import assert_, assert_allclose

from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (mlemodel, sarimax, structural, varmax,
                                        dynamic_factor)
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata


def test_sarimax():
    # AR(1)
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    phi = 0.5
    actual = mod.impulse_responses([phi, 1], steps=10)
    desired = np.r_[[phi**i for i in range(11)]]
    assert_allclose(actual, desired)

    # MA(1)
    mod = sarimax.SARIMAX([0], order=(0, 0, 1))
    theta = 0.5
    actual = mod.impulse_responses([theta, 1], steps=10)
    desired = np.r_[1, theta, [0]*9]
    assert_allclose(actual, desired)

    # ARMA(2, 2) + constant
    # Stata:
    # webuse lutkepohl2
    # arima dln_inc, arima(2, 0, 2)
    # irf create irf1, set(irf1) step(10)
    # irf table irf
    params = [.01928228, -.03656216, .7588994,
              .27070341, -.72928328, .01122177**0.5]
    mod = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, .234141, .021055, .17692, .00951, .133917, .002321, .101544,
               -.001951, .077133, -.004301]
    assert_allclose(actual, desired, atol=1e-6)

    # SARIMAX(1,1,1)x(1,0,1,4) + constant + exog
    # Stata:
    # webuse lutkepohl2
    # gen exog = _n^2
    # arima inc exog, arima(1,1,1) sarima(1,0,1,4)
    # irf create irf2, set(irf2) step(10)
    # irf table irf
    params = [.12853289, 12.207156, .86384742, -.71463236,
              .81878967, -.9533955, 14.043884**0.5]
    exog = np.arange(1, 92)**2
    mod = sarimax.SARIMAX(np.zeros(91), order=(1, 1, 1),
                          seasonal_order=(1, 0, 1, 4), trend='c', exog=exog,
                          simple_differencing=True)
    actual = mod.impulse_responses(params, steps=10)
    desired = [1, .149215, .128899, .111349, -.038417, .063007, .054429,
               .047018, -.069598, .018641, .016103]
    assert_allclose(actual, desired, atol=1e-6)


def test_structural():
    steps = 10

    # AR(1)
    mod = structural.UnobservedComponents([0], autoregressive=1)
    phi = 0.5
    actual = mod.impulse_responses([1, phi], steps)
    desired = np.r_[[phi**i for i in range(steps + 1)]]
    assert_allclose(actual, desired)

    # ARX(1)
    # This is adequately tested in test_simulate.py, since in the time-varying
    # case `impulse_responses` just calls `simulate`

    # Irregular
    mod = structural.UnobservedComponents([0], 'irregular')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Fixed intercept
    # (in practice this is a deterministic constant, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed intercept')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Deterministic constant
    mod = structural.UnobservedComponents([0], 'deterministic constant')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Local level
    mod = structural.UnobservedComponents([0], 'local level')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, 1)

    # Random walk
    mod = structural.UnobservedComponents([0], 'random walk')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 1)

    # Fixed slope
    # (in practice this is a deterministic trend, because an irregular
    #  component must be added)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = structural.UnobservedComponents([0], 'fixed slope')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Deterministic trend
    mod = structural.UnobservedComponents([0], 'deterministic trend')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Local linear deterministic trend
    mod = structural.UnobservedComponents(
        [0], 'local linear deterministic trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, 1)

    # Random walk with drift
    mod = structural.UnobservedComponents([0], 'random walk with drift')
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 1)

    # Local linear trend
    mod = structural.UnobservedComponents([0], 'local linear trend')
    # - shock the level
    actual = mod.impulse_responses([1., 1., 1.], steps)
    assert_allclose(actual, 1)
    # - shock the trend
    actual = mod.impulse_responses([1., 1., 1.], steps, impulse=1)
    assert_allclose(actual, np.arange(steps + 1))

    # Smooth trend
    mod = structural.UnobservedComponents([0], 'smooth trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, np.arange(steps + 1))

    # Random trend
    mod = structural.UnobservedComponents([0], 'random trend')
    actual = mod.impulse_responses([1., 1.], steps)
    assert_allclose(actual, np.arange(steps + 1))

    # Seasonal (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2,
                                          stochastic_seasonal=False)
    actual = mod.impulse_responses([1.], steps)
    assert_allclose(actual, 0)

    # Seasonal (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', seasonal=2)
    actual = mod.impulse_responses([1., 1.], steps)
    desired = np.r_[1, np.tile([-1, 1], steps // 2)]
    assert_allclose(actual, desired)

    # Cycle (deterministic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True)
    actual = mod.impulse_responses([1., 1.2], steps)
    assert_allclose(actual, 0)

    # Cycle (stochastic)
    mod = structural.UnobservedComponents([0], 'irregular', cycle=True,
                                          stochastic_cycle=True)
    actual = mod.impulse_responses([1., 1., 1.2], steps=10)
    x1 = [np.cos(1.2), np.sin(1.2)]
    x2 = [-np.sin(1.2), np.cos(1.2)]
    T = np.array([x1, x2])
    desired = np.zeros(steps + 1)
    states = [1, 0]
    for i in range(steps + 1):
        desired[i] += states[0]
        states = np.dot(T, states)
    assert_allclose(actual, desired)


def test_varmax():
    steps = 10

    # Clear warnings
    varmax.__warningregistry__ = {}

    # VAR(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(2, 0), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)

    # VMA(2) - single series
    mod1 = varmax.VARMAX([[0]], order=(0, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(0, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual, desired)

    # VARMA(2, 2) - single series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='n')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2))
    actual = mod1.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)

    # VARMA(2, 2) + trend - single series
    warning = EstimationWarning
    match = r'VARMA\(p,q\) models is not'
    with pytest.warns(warning, match=match):
        mod1 = varmax.VARMAX([[0]], order=(2, 2), trend='c')
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 2), trend='c')
    actual = mod1.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    desired = mod2.impulse_responses([10, 0.5, 0.2, 0.1, -0.2, 1], steps)
    assert_allclose(actual, desired)

    # VAR(2) + constant
    # Stata:
    # webuse lutkepohl2
    # var dln_inv dln_inc, lags(1/2)
    # irf create irf3, set(irf3) step(10)
    # irf table irf
    # irf table oirf
    params = [-.00122728, .01503679,
              -.22741923, .71030531, -.11596357, .51494891,
              .05974659, .02094608, .05635125, .08332519,
              .04297918, .00159473, .01096298]
    irf_00 = [1, -.227419, -.021806, .093362, -.001875, -.00906, .009605,
              .001323, -.001041, .000769, .00032]
    irf_01 = [0, .059747, .044015, -.008218, .007845, .004629, .000104,
              .000451, .000638, .000063, .000042]
    irf_10 = [0, .710305, .36829, -.065697, .084398, .043038, .000533,
              .005755, .006051, .000548, .000526]
    irf_11 = [1, .020946, .126202, .066419, .028735, .007477, .009878,
              .003287, .001266, .000986, .0005]
    oirf_00 = [0.042979, -0.008642, -0.00035, 0.003908, 0.000054, -0.000321,
               0.000414, 0.000066, -0.000035, 0.000034, 0.000015]
    oirf_01 = [0.001595, 0.002601, 0.002093, -0.000247, 0.000383, 0.000211,
               0.00002, 0.000025, 0.000029, 4.30E-06, 2.60E-06]
    oirf_10 = [0, 0.007787, 0.004037, -0.00072, 0.000925, 0.000472, 5.80E-06,
               0.000063, 0.000066, 6.00E-06, 5.80E-06]
    oirf_11 = [0.010963, 0.00023, 0.001384, 0.000728, 0.000315, 0.000082,
               0.000108, 0.000036, 0.000014, 0.000011, 5.50E-06]

    mod = varmax.VARMAX([[0, 0]], order=(2, 0), trend='c')

    # IRFs
    actual = mod.impulse_responses(params, steps, impulse=0)
    assert_allclose(actual, np.c_[irf_00, irf_01], atol=1e-6)

    actual = mod.impulse_responses(params, steps, impulse=1)
    assert_allclose(actual, np.c_[irf_10, irf_11], atol=1e-6)

    # Orthogonalized IRFs
    actual = mod.impulse_responses(params, steps, impulse=0,
                                   orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_00, oirf_01], atol=1e-6)

    actual = mod.impulse_responses(params, steps, impulse=1,
                                   orthogonalized=True)
    assert_allclose(actual, np.c_[oirf_10, oirf_11], atol=1e-6)

    # Impulse response passing column name
    data = get_macrodata().view((float, 3), type=np.ndarray)

    df = pd.DataFrame({
        "a": data[:, 0],
        "b": data[:, 1],
        "c": data[:, 2]})

    mod1 = varmax.VARMAX(df, order=(1, 0), trend='c')
    mod1_result = mod1.fit()
    mod2 = varmax.VARMAX(data, order=(1, 0), trend='c')
    mod2_result = mod2.fit()

    with pytest.raises(ValueError, match='Endog must be pd.DataFrame.'):
        mod2_result.impulse_responses(6, impulse="b")

    response1 = mod1_result.impulse_responses(6, impulse="b")
    response2 = mod1_result.impulse_responses(6, impulse=[0, 1, 0])
    assert_allclose(response1, response2)

    # VARMA(2, 2) + trend + exog
    # TODO: This is just a smoke test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = varmax.VARMAX(
            np.random.normal(size=(steps, 2)), order=(2, 2), trend='c',
            exog=np.ones(steps), enforce_stationarity=False,
            enforce_invertibility=False)
    mod.impulse_responses(mod.start_params, steps)


def test_dynamic_factor():
    steps = 10
    exog = np.random.normal(size=steps)

    # DFM: 2 series, AR(2) factor
    mod1 = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=2)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses([-0.9, 0.8, 1., 1., 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM: 2 series, AR(2) factor, exog
    mod1 = dynamic_factor.DynamicFactor(np.zeros((steps, 2)), k_factors=1,
                                        factor_order=2, exog=exog)
    mod2 = sarimax.SARIMAX([0], order=(2, 0, 0))
    actual = mod1.impulse_responses(
        [-0.9, 0.8, 5, -2, 1., 1., 0.5, 0.2], steps)
    desired = mod2.impulse_responses([0.5, 0.2, 1], steps)
    assert_allclose(actual[:, 0], -0.9 * desired)
    assert_allclose(actual[:, 1], 0.8 * desired)

    # DFM, 3 series, VAR(2) factor, exog, error VAR
    # TODO: This is just a smoke test
    mod = dynamic_factor.DynamicFactor(np.random.normal(size=(steps, 3)),
                                       k_factors=2, factor_order=2, exog=exog,
                                       error_order=2, error_var=True,
                                       enforce_stationarity=False)
    mod.impulse_responses(mod.start_params, steps)


def test_time_varying_ssm():
    # Create an ad-hoc time-varying model
    mod = sarimax.SARIMAX([0] * 11, order=(1, 0, 0))
    mod.update([0.5, 1.0])
    T = np.zeros((1, 1, 11))
    T[..., :5] = 0.5
    T[..., 5:] = 0.2
    mod['transition'] = T

    irfs = mod.ssm.impulse_responses()
    desired = np.cumprod(np.r_[1, [0.5] * 4, [0.2] * 5]).reshape(10, 1)
    assert_allclose(irfs, desired)


class TVSS(mlemodel.MLEModel):
    """
    Time-varying state space model for testing

    This creates a state space model with randomly generated time-varying
    system matrices. When used in a test, that test should use
    `reset_randomstate` to ensure consistent test runs.
    """
    def __init__(self, endog, _k_states=None):
        k_states = 2
        k_posdef = 2
        # Allow subcasses to add additional states
        if _k_states is None:
            _k_states = k_states
        super(TVSS, self).__init__(endog, k_states=_k_states,
                                   k_posdef=k_posdef, initialization='diffuse')

        self['obs_intercept'] = np.random.normal(
            size=(self.k_endog, self.nobs))
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['transition'] = np.zeros(
            (self.k_states, self.k_states, self.nobs))
        self['selection'] = np.zeros(
            (self.k_states, self.ssm.k_posdef, self.nobs))
        self['design', :, :k_states, :] = np.random.normal(
            size=(self.k_endog, k_states, self.nobs))
        # For the transition matrices, enforce eigenvalues not too far outside
        # unit circle. Otherwise, the random draws will often lead to large
        # eigenvalues that cause the covariance matrices to blow up to huge
        # values during long periods of missing data, which leads to numerical
        # problems and essentially spurious test failures
        D = [np.diag(d)
             for d in np.random.uniform(-1.1, 1.1, size=(self.nobs, k_states))]
        Q = ortho_group.rvs(k_states, size=self.nobs)
        self['transition', :k_states, :k_states, :] = (
            Q @ D @ Q.transpose(0, 2, 1)).transpose(1, 2, 0)
        self['selection', :k_states, :, :] = np.random.normal(
            size=(k_states, self.ssm.k_posdef, self.nobs))

        # Need to make sure the covariances are positive definite
        H05 = np.random.normal(size=(self.k_endog, self.k_endog, self.nobs))
        Q05 = np.random.normal(
            size=(self.ssm.k_posdef, self.ssm.k_posdef, self.nobs))
        H = np.zeros_like(H05)
        Q = np.zeros_like(Q05)
        for t in range(self.nobs):
            H[..., t] = np.dot(H05[..., t], H05[..., t].T)
            Q[..., t] = np.dot(Q05[..., t], Q05[..., t].T)
        self['obs_cov'] = H
        self['state_cov'] = Q

    def clone(self, endog, exog=None, **kwargs):
        mod = self.__class__(endog, **kwargs)

        for key in self.ssm.shapes.keys():
            if key in ['obs', 'state_intercept']:
                continue
            n = min(self.nobs, mod.nobs)
            mod[key, ..., :n] = self.ssm[key, ..., :n]

        return mod


def test_time_varying_in_sample(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))

    # Compute the max number of in-sample IRFs
    irfs = mod.impulse_responses([], steps=mod.nobs - 1)
    # Compute the same thing, but now with explicit anchor argument
    irfs_anchor = mod.impulse_responses([], steps=mod.nobs - 1, anchor=0)

    # Cumulative IRFs
    cirfs = mod.impulse_responses([], steps=mod.nobs - 1, cumulative=True)
    # Orthogonalized IRFs
    oirfs = mod.impulse_responses([], steps=mod.nobs - 1, orthogonalized=True)
    # Cumulative, orthogonalized IRFs
    coirfs = mod.impulse_responses([], steps=mod.nobs - 1, cumulative=True,
                                   orthogonalized=True)

    # Compute IRFs manually
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., 0]
    L = np.linalg.cholesky(Q)

    desired_irfs = np.zeros((mod.nobs - 1, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - 1, 2)) * np.nan
    tmp = R[..., 0]
    for i in range(1, mod.nobs):
        desired_irfs[i - 1] = Z[:, :, i].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i].dot(tmp)

    assert_allclose(irfs, desired_irfs)
    assert_allclose(irfs_anchor, desired_irfs)

    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))


def test_time_varying_out_of_sample(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))

    # Compute all in-sample IRFs and also one out-of-sample IRF
    new_Z = np.random.normal(size=mod['design', :, :, -1].shape)
    new_T = np.random.normal(size=mod['transition', :, :, -1].shape)
    irfs = mod.impulse_responses(
        [], steps=mod.nobs, design=new_Z[:, :, None],
        transition=new_T[:, :, None])
    # Compute the same thing, but now with explicit anchor argument
    irfs_anchor = mod.impulse_responses(
        [], steps=mod.nobs, anchor=0, design=new_Z[:, :, None],
        transition=new_T[:, :, None])

    # Cumulative IRFs
    cirfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None],
                                  transition=new_T[:, :, None],
                                  cumulative=True)
    # Orthogonalized IRFs
    oirfs = mod.impulse_responses([], steps=mod.nobs, design=new_Z[:, :, None],
                                  transition=new_T[:, :, None],
                                  orthogonalized=True)
    # Cumulative, orthogonalized IRFs
    coirfs = mod.impulse_responses(
        [], steps=mod.nobs, design=new_Z[:, :, None],
        transition=new_T[:, :, None], cumulative=True, orthogonalized=True)

    # Compute IRFs manually
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., 0]
    L = np.linalg.cholesky(Q)

    desired_irfs = np.zeros((mod.nobs, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs, 2)) * np.nan
    tmp = R[..., 0]
    for i in range(1, mod.nobs):
        desired_irfs[i - 1] = Z[:, :, i].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i].dot(tmp)
    desired_irfs[mod.nobs - 1] = new_Z.dot(tmp)[:, 0]
    desired_oirfs[mod.nobs - 1] = new_Z.dot(tmp).dot(L)[:, 0]

    assert_allclose(irfs, desired_irfs)
    assert_allclose(irfs_anchor, desired_irfs)

    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))


def test_time_varying_in_sample_anchored(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))

    # Compute the max number of in-sample IRFs
    anchor = 2
    irfs = mod.impulse_responses(
        [], steps=mod.nobs - 1 - anchor, anchor=anchor)

    # Cumulative IRFs
    cirfs = mod.impulse_responses(
        [], steps=mod.nobs - 1 - anchor, anchor=anchor,
        cumulative=True)
    # Orthogonalized IRFs
    oirfs = mod.impulse_responses(
        [], steps=mod.nobs - 1 - anchor, anchor=anchor,
        orthogonalized=True)
    # Cumulative, orthogonalized IRFs
    coirfs = mod.impulse_responses(
        [], steps=mod.nobs - 1 - anchor, anchor=anchor,
        cumulative=True, orthogonalized=True)

    # Compute IRFs manually
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., anchor]
    L = np.linalg.cholesky(Q)

    desired_irfs = np.zeros((mod.nobs - anchor - 1, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - anchor - 1, 2)) * np.nan
    tmp = R[..., anchor]
    for i in range(1, mod.nobs - anchor):
        desired_irfs[i - 1] = Z[:, :, i + anchor].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i + anchor].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i + anchor].dot(tmp)

    assert_allclose(irfs, desired_irfs)

    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))


def test_time_varying_out_of_sample_anchored(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))

    # Compute all in-sample IRFs and also one out-of-sample IRF
    anchor = 2

    new_Z = mod['design', :, :, -1]
    new_T = mod['transition', :, :, -1]
    irfs = mod.impulse_responses(
        [], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None],
        transition=new_T[:, :, None])

    # Cumulative IRFs
    cirfs = mod.impulse_responses(
        [], steps=mod.nobs - anchor, anchor=anchor,
        design=new_Z[:, :, None], transition=new_T[:, :, None],
        cumulative=True)
    # Orthogonalized IRFs
    oirfs = mod.impulse_responses(
        [], steps=mod.nobs - anchor, anchor=anchor,
        design=new_Z[:, :, None], transition=new_T[:, :, None],
        orthogonalized=True)
    # Cumulative, orthogonalized IRFs
    coirfs = mod.impulse_responses(
        [], steps=mod.nobs - anchor, anchor=anchor,
        design=new_Z[:, :, None], transition=new_T[:, :, None],
        cumulative=True, orthogonalized=True)

    # Compute IRFs manually
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., anchor]
    L = np.linalg.cholesky(Q)

    desired_irfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    tmp = R[..., anchor]
    for i in range(1, mod.nobs - anchor):
        desired_irfs[i - 1] = Z[:, :, i + anchor].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i + anchor].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i + anchor].dot(tmp)
    desired_irfs[mod.nobs - anchor - 1] = new_Z.dot(tmp)[:, 0]
    desired_oirfs[mod.nobs - anchor - 1] = new_Z.dot(tmp).dot(L)[:, 0]

    assert_allclose(irfs, desired_irfs)

    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))


def test_time_varying_out_of_sample_anchored_end(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))

    # Cannot compute the any in-sample IRFs when anchoring at the end
    with pytest.raises(ValueError, match='Model has time-varying'):
        mod.impulse_responses([], steps=2, anchor='end')

    # Compute two out-of-sample IRFs
    new_Z = np.random.normal(size=mod['design', :, :, -2:].shape)
    new_T = np.random.normal(size=mod['transition', :, :, -2:].shape)
    irfs = mod.impulse_responses([], steps=2, anchor='end',
                                 design=new_Z, transition=new_T)

    # Cumulative IRFs
    cirfs = mod.impulse_responses(
        [], steps=2, anchor='end', design=new_Z, transition=new_T,
        cumulative=True)
    # Orthogonalized IRFs
    oirfs = mod.impulse_responses(
        [], steps=2, anchor='end', design=new_Z, transition=new_T,
        orthogonalized=True)
    # Cumulative, orthogonalized IRFs
    coirfs = mod.impulse_responses(
        [], steps=2, anchor='end', design=new_Z, transition=new_T,
        cumulative=True, orthogonalized=True)

    # Compute IRFs manually
    R = mod['selection']
    Q = mod['state_cov', ..., -1]
    L = np.linalg.cholesky(Q)

    desired_irfs = np.zeros((2, 2)) * np.nan
    desired_oirfs = np.zeros((2, 2)) * np.nan
    # desired[0] = 0
    # Z_{T+1} R_T
    tmp = R[..., -1]
    desired_irfs[0] = new_Z[:, :, 0].dot(tmp)[:, 0]
    desired_oirfs[0] = new_Z[:, :, 0].dot(tmp).dot(L)[:, 0]
    # T_{T+1} R_T
    tmp = new_T[..., 0].dot(tmp)
    # Z_{T+2} T_{T+1} R_T
    desired_irfs[1] = new_Z[:, :, 1].dot(tmp)[:, 0]
    desired_oirfs[1] = new_Z[:, :, 1].dot(tmp).dot(L)[:, 0]

    assert_allclose(irfs, desired_irfs)

    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))


def test_pandas_univariate_rangeindex():
    # Impulse responses have RangeIndex
    endog = pd.Series(np.zeros(1))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    actual = res.impulse_responses(2)
    desired = pd.Series([1., 0.5, 0.25])
    assert_allclose(res.impulse_responses(2), desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_univariate_dateindex():
    # Impulse responses still have RangeIndex (i.e. aren't wrapped with dates)
    ix = pd.date_range(start='2000', periods=1, freq='M')
    endog = pd.Series(np.zeros(1), index=ix)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.])

    actual = res.impulse_responses(2)
    desired = pd.Series([1., 0.5, 0.25])
    assert_allclose(res.impulse_responses(2), desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_rangeindex():
    # Impulse responses have RangeIndex
    endog = pd.DataFrame(np.zeros((1, 2)))
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    actual = res.impulse_responses(2)
    desired = pd.DataFrame([[1., 0.5, 0.25], [0., 0., 0.]]).T
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_multivariate_dateindex():
    # Impulse responses still have RangeIndex (i.e. aren't wrapped with dates)
    ix = pd.date_range(start='2000', periods=1, freq='M')
    endog = pd.DataFrame(np.zeros((1, 2)), index=ix)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0., 0., 0.2, 1., 0., 1.])

    actual = res.impulse_responses(2)
    desired = pd.DataFrame([[1., 0.5, 0.25], [0., 0., 0.]]).T
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))


def test_pandas_anchor():
    # Test that anchor with dates works
    ix = pd.date_range(start='2000', periods=10, freq='M')
    endog = pd.DataFrame(np.zeros((10, 2)), index=ix)
    mod = TVSS(endog)
    res = mod.filter([])

    desired = res.impulse_responses(2, anchor=1)

    # Anchor to date
    actual = res.impulse_responses(2, anchor=ix[1])
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))

    # Anchor to negative index
    actual = res.impulse_responses(2, anchor=-9)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
