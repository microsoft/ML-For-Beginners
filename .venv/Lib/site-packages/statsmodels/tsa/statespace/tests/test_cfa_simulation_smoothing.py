"""
Tests for CFA simulation smoothing

Author: Chad Fulton
License: BSD-3
"""
import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded

from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
                                        varmax)

current_path = os.path.dirname(os.path.abspath(__file__))
dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
dta = np.log(dta[['realcons', 'realgdp', 'cpi']]).diff().iloc[1:] * 400


class CheckPosteriorMoments:
    @classmethod
    def setup_class(cls, model_class, missing=None, mean_atol=0, cov_atol=0,
                    use_complex=False, *args, **kwargs):
        cls.mean_atol = mean_atol
        cls.cov_atol = cov_atol

        endog = dta.copy()

        if missing == 'all':
            endog.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            endog.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            endog.iloc[0:50, 0] = np.nan
            endog.iloc[19:70, 1] = np.nan
            endog.iloc[39:90, 2] = np.nan
            endog.iloc[119:130, 0] = np.nan
            endog.iloc[119:130, 2] = np.nan
            endog.iloc[-10:, :] = np.nan

        if model_class in [sarimax.SARIMAX, structural.UnobservedComponents]:
            endog = endog.iloc[:, 2]

        cls.mod = model_class(endog, *args, **kwargs)
        params = cls.mod.start_params
        if use_complex:
            params = params + 0j
        cls.res = cls.mod.smooth(params)

        cls.sim_cfa = cls.mod.simulation_smoother(method='cfa')
        cls.sim_cfa.simulate()
        prefix = 'z' if use_complex else 'd'
        cls._sim_cfa = cls.sim_cfa._simulation_smoothers[prefix]

    def test_posterior_mean(self):
        # Test the values from the Cython results
        actual = np.array(self._sim_cfa.posterior_mean, copy=True)
        assert_allclose(actual, self.res.smoothed_state, atol=self.mean_atol)

        # Test the values from the CFASimulationSmoother wrapper results
        assert_allclose(self.sim_cfa.posterior_mean, self.res.smoothed_state,
                        atol=self.mean_atol)

    def test_posterior_cov(self):
        # Test the values from the Cython results
        inv_chol = np.array(self._sim_cfa.posterior_cov_inv_chol, copy=True)
        actual = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))

        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m],
                            self.res.smoothed_state_cov[..., t],
                            atol=self.cov_atol)

        # Test the values from the CFASimulationSmoother wrapper results
        actual = self.sim_cfa.posterior_cov

        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m],
                            self.res.smoothed_state_cov[..., t],
                            atol=self.cov_atol)


class TestDFM(CheckPosteriorMoments):
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['k_factors'] = 1
        kwargs['factor_order'] = 1
        super().setup_class(dynamic_factor.DynamicFactor, missing=missing,
                            *args, **kwargs)


class TestDFMComplex(CheckPosteriorMoments):
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['k_factors'] = 1
        kwargs['factor_order'] = 1
        super().setup_class(dynamic_factor.DynamicFactor, missing=missing,
                            use_complex=True, *args, **kwargs)


class TestDFMAllMissing(TestDFM):
    def setup_class(cls, missing='all', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestDFMPartialMissing(TestDFM):
    def setup_class(cls, missing='partial', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestDFMMixedMissing(TestDFM):
    def setup_class(cls, missing='mixed', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestVARME(CheckPosteriorMoments):
    # Test VAR model with Measurement Error
    # Note: this includes a trend
    # Note: have to use measurement error, due to the restriction that all
    # shocks must be non-degenerate for the CFA algorithm
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['order'] = (1, 0)
        kwargs['measurement_error'] = True
        super().setup_class(varmax.VARMAX, missing=missing, *args, **kwargs)


class TestVARMEAllMissing(TestVARME):
    def setup_class(cls, missing='all', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestVARMEPartialMissing(TestVARME):
    def setup_class(cls, missing='partial', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestVARMEMixedMissing(TestVARME):
    def setup_class(cls, missing='mixed', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestSARIMAXME(CheckPosteriorMoments):
    # Test SARIMAX model with Measurement Error
    # Note: have to use measurement error, due to the restriction that all
    # shocks must be non-degenerate for the CFA algorithm
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['order'] = (1, 0, 0)
        kwargs['measurement_error'] = True
        super().setup_class(sarimax.SARIMAX, missing=missing, *args, **kwargs)


class TestSARIMAXMEMissing(TestSARIMAXME):
    def setup_class(cls, missing='mixed', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


class TestUnobservedComponents(CheckPosteriorMoments):
    # Test UC model, with exog
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['level'] = 'llevel'
        kwargs['exog'] = np.arange(dta.shape[0])
        kwargs['autoregressive'] = 1
        super().setup_class(structural.UnobservedComponents, missing=missing,
                            *args, **kwargs)


class TestUnobservedComponentsMissing(TestUnobservedComponents):
    def setup_class(cls, missing='mixed', *args, **kwargs):
        super().setup_class(missing=missing, *args, **kwargs)


def test_dfm(missing=None):
    mod = dynamic_factor.DynamicFactor(dta, k_factors=2, factor_order=1)
    mod.update(mod.start_params)
    sim_cfa = mod.simulation_smoother(method='cfa')
    res = mod.ssm.smooth()

    # Test zero variates
    sim_cfa.simulate(np.zeros((mod.k_states, mod.nobs)))
    assert_allclose(sim_cfa.simulated_state, res.smoothed_state)
