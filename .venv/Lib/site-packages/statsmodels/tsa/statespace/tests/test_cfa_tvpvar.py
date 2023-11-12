"""
Tests for CFA simulation smoothing in a TVP-VAR model

See "results/cfa_tvpvar_test.m" for Matlab file that generates the results.

Based on "TVPVAR.m", found at http://joshuachan.org/code/code_TVPVAR.html.
See [1]_ for details on the TVP-VAR model and the CFA method.

References
----------
.. [1] Chan, Joshua CC, and Ivan Jeliazkov.
       "Efficient simulation and integrated likelihood estimation in
       state space models."
       International Journal of Mathematical Modelling and Numerical
       Optimisation 1, no. 1-2 (2009): 101-120.

Author: Chad Fulton
License: BSD-3
"""
import os

import numpy as np
import pandas as pd

from numpy.testing import assert_allclose

from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel

dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
endog = np.log(dta[['realcons', 'realgdp']]).diff().iloc[1:13] * 400

current_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_path, 'results')
results = {
    'invP': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_invP.csv'), header=None),
    'posterior_mean': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_posterior_mean.csv'),
        header=None),
    'state_variates': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_state_variates.csv'),
        header=None),
    'beta': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_beta.csv'), header=None),
    'v10': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_v10.csv'), header=None),
    'S10': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_S10.csv'), header=None),
    'Omega_11': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_Omega_11.csv'), header=None),
    'vi0': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_vi0.csv'), header=None),
    'Si0': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_Si0.csv'), header=None),
    'Omega_22': pd.read_csv(
        os.path.join(results_path, 'cfa_tvpvar_Omega_22.csv'), header=None)}


class TVPVAR(mlemodel.MLEModel):
    def __init__(self, endog):
        if not isinstance(endog, pd.DataFrame):
            endog = pd.DataFrame(endog)

        k = endog.shape[1]
        augmented = lagmat(endog, 1, trim='both', original='in',
                           use_pandas=True)
        endog = augmented.iloc[:, :k]
        exog = add_constant(augmented.iloc[:, k:])

        k_states = k * (k + 1)
        super().__init__(endog, k_states=k_states)

        self.ssm.initialize('known', stationary_cov=np.eye(self.k_states) * 5)

        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = exog.T
        self['transition'] = np.eye(k_states)
        self['selection'] = np.eye(k_states)

        self._obs_cov_slice = np.s_[:self.k_endog * (self.k_endog + 1) // 2]
        self._obs_cov_tril = np.tril_indices(self.k_endog)
        self._state_cov_slice = np.s_[-self.k_states:]
        self._state_cov_ix = ('state_cov',) + np.diag_indices(self.k_states)

    @property
    def state_names(self):
        state_names = []
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names += ['intercept.%s' % endog_name]
            state_names += ['L1.%s->%s' % (other_name, endog_name)
                            for other_name in self.endog_names]
        return state_names

    def update_direct(self, obs_cov, state_cov_diag):
        self['obs_cov'] = obs_cov
        self[self._state_cov_ix] = state_cov_diag


def test_tvpvar():
    # This tests two MCMC iterations of the TVP-VAR model using the CFA
    # simulation smoother.

    # Create the model and the CFA simulation smoother
    mod = TVPVAR(endog.iloc[2:])
    sim = mod.simulation_smoother(method='cfa')

    # Prior hyperparameters
    v10 = mod.k_endog + 3
    S10 = np.eye(mod.k_endog)

    vi0 = np.ones(mod.k_states) * 6 / 2
    Si0 = np.ones(mod.k_states) * 0.01 / 2

    # - First iteration ------------------------------------------------------

    # Update the model with initial parameterization
    initial_obs_cov = np.cov(endog.T)
    initial_state_cov_vars = np.ones(mod.k_states) * 0.01
    mod.update_direct(initial_obs_cov, initial_state_cov_vars)
    res = mod.ssm.smooth()

    # Simulate the state using the given variates
    # (this also computes posterior moments)
    variates_1 = results['state_variates'].iloc[:6]
    sim.simulate(variates_1)

    # Check state posterior moments
    posterior_mean_1 = results['posterior_mean'].iloc[:6]
    assert_allclose(sim.posterior_mean, posterior_mean_1)
    assert_allclose(sim.posterior_mean, res.smoothed_state)

    posterior_cov_1 = np.linalg.inv(results['invP'].iloc[:54])
    assert_allclose(sim.posterior_cov, posterior_cov_1)

    # Check simulated state
    simulated_state_1 = results['beta'].iloc[:6]
    assert_allclose(sim.simulated_state, simulated_state_1)

    # Posterior for obs cov
    fitted = np.matmul(mod['design'].transpose(2, 0, 1),
                       sim.simulated_state.T[..., None])[..., 0]
    resid = mod.endog - fitted
    df = v10 + mod.nobs
    scale = S10 + np.dot(resid.T, resid)
    assert_allclose(df, results['v10'].iloc[:2])
    assert_allclose(scale, results['S10'].iloc[:, :2])

    # Posterior for state cov
    resid = sim.simulated_state.T[1:] - sim.simulated_state.T[:-1]
    sse = np.sum(resid**2, axis=0)
    shapes = vi0 + (mod.nobs - 1) / 2
    scales = Si0 + sse / 2
    assert_allclose(shapes, results['vi0'].values[0, 0])
    assert_allclose(scales, results['Si0'].iloc[:, 0])

    # - Second iteration -----------------------------------------------------

    # Update the model with variates drawn in the previous iteration (here we
    # use the saved test case variates)
    mod.update_direct(results['Omega_11'].iloc[:, :2],
                      results['Omega_22'].iloc[:, 0])
    res = mod.ssm.smooth()

    # Simulate the state using the given variates
    # (this also computes posterior moments)
    variates_2 = results['state_variates'].iloc[6:]
    sim.simulate(variates_2)

    # Check state posterior moments
    posterior_mean_2 = results['posterior_mean'].iloc[6:]
    assert_allclose(sim.posterior_mean, posterior_mean_2)
    assert_allclose(sim.posterior_mean, res.smoothed_state)

    posterior_cov_2 = np.linalg.inv(results['invP'].iloc[54:])
    assert_allclose(sim.posterior_cov, posterior_cov_2)

    # Check simulated state
    simulated_state_2 = results['beta'].iloc[6:]
    assert_allclose(sim.simulated_state, simulated_state_2)

    # Posterior for obs cov
    fitted = np.matmul(mod['design'].transpose(2, 0, 1),
                       sim.simulated_state.T[..., None])[..., 0]
    resid = mod.endog - fitted
    df = v10 + mod.nobs
    scale = S10 + np.dot(resid.T, resid)
    assert_allclose(df, results['v10'].iloc[2:])
    assert_allclose(scale, results['S10'].iloc[:, 2:])

    # Posterior for state cov
    resid = sim.simulated_state.T[1:] - sim.simulated_state.T[:-1]
    sse = np.sum(resid**2, axis=0)
    shapes = vi0 + (mod.nobs - 1) / 2
    scales = Si0 + sse / 2
    assert_allclose(shapes, results['vi0'].values[0, 1])
    assert_allclose(scales, results['Si0'].iloc[:, 1])
