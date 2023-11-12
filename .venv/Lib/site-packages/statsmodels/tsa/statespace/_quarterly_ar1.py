# -*- coding: utf-8 -*-
"""
(Internal) AR(1) model for monthly growth rates aggregated to quarterly freq.

Author: Chad Fulton
License: BSD-3
"""
import warnings
import numpy as np

from statsmodels.tools.tools import Bunch
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.kalman_smoother import (
    SMOOTHER_STATE, SMOOTHER_STATE_COV, SMOOTHER_STATE_AUTOCOV)
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate)


class QuarterlyAR1(mlemodel.MLEModel):
    r"""
    AR(1) model for monthly growth rates aggregated to quarterly frequency

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`

    Notes
    -----
    This model is internal, used to estimate starting parameters for the
    DynamicFactorMQ class. The model is:

    .. math::

        y_t & = \begin{bmatrix} 1 & 2 & 3 & 2 & 1 \end{bmatrix} \alpha_t \\
        \alpha_t & = \begin{bmatrix}
            \phi & 0 & 0 & 0 & 0 \\
               1 & 0 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 & 0 \\
               0 & 0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 1 & 0 \\
        \end{bmatrix} +
        \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} \varepsilon_t

    The two parameters to be estimated are :math:`\phi` and :math:`\sigma^2`.

    It supports fitting via the usual quasi-Newton methods, as well as using
    the EM algorithm.

    """
    def __init__(self, endog):
        super().__init__(endog, k_states=5, k_posdef=1,
                         initialization='stationary')
        self['design'] = [1, 2, 3, 2, 1]
        self['transition', 1:, :-1] = np.eye(4)
        self['selection', 0, 0] = 1.

    @property
    def param_names(self):
        return ['phi', 'sigma2']

    @property
    def start_params(self):
        return np.array([0, np.nanvar(self.endog) / 19])

    def fit(self, *args, **kwargs):
        # Don't show warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = super().fit(*args, **kwargs)
        return out

    def fit_em(self, start_params=None, transformed=True, cov_type='none',
               cov_kwds=None, maxiter=500, tolerance=1e-6,
               em_initialization=True, mstep_method=None, full_output=True,
               return_params=False, low_memory=False):
        if self._has_fixed_params:
            raise NotImplementedError('Cannot fit using the EM algorithm while'
                                      ' holding some parameters fixed.')
        if low_memory:
            raise ValueError('Cannot fit using the EM algorithm when using'
                             ' low_memory option.')

        if start_params is None:
            start_params = self.start_params
            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        if not transformed:
            start_params = self.transform_params(start_params)

        # Perform expectation-maximization
        llf = []
        params = [start_params]
        init = None
        i = 0
        delta = 0
        while i < maxiter and (i < 2 or (delta > tolerance)):
            out = self._em_iteration(params[-1], init=init,
                                     mstep_method=mstep_method)
            llf.append(out[0].llf_obs.sum())
            params.append(out[1])
            if em_initialization:
                init = initialization.Initialization(
                    self.k_states, 'known',
                    constant=out[0].smoothed_state[..., 0],
                    stationary_cov=out[0].smoothed_state_cov[..., 0])
            if i > 0:
                delta = (2 * (llf[-1] - llf[-2]) /
                         (np.abs(llf[-1]) + np.abs(llf[-2])))
            i += 1

        # Just return the fitted parameters if requested
        if return_params:
            result = params[-1]
        # Otherwise construct the results class if desired
        else:
            if em_initialization:
                base_init = self.ssm.initialization
                self.ssm.initialization = init
            result = self.smooth(params[-1], transformed=True,
                                 cov_type=cov_type, cov_kwds=cov_kwds)
            if em_initialization:
                self.ssm.initialization = base_init

            # Save the output
            if full_output:
                em_retvals = Bunch(**{'params': np.array(params),
                                      'llf': np.array(llf),
                                      'iter': i})
                em_settings = Bunch(**{'tolerance': tolerance,
                                       'maxiter': maxiter})
            else:
                em_retvals = None
                em_settings = None

            result.mle_retvals = em_retvals
            result.mle_settings = em_settings

        return result

    def _em_iteration(self, params0, init=None, mstep_method=None):
        # (E)xpectation step
        res = self._em_expectation_step(params0, init=init)

        # (M)aximization step
        params1 = self._em_maximization_step(res, params0,
                                             mstep_method=mstep_method)

        return res, params1

    def _em_expectation_step(self, params0, init=None):
        # (E)xpectation step
        self.update(params0)
        # Re-initialize state, if new initialization is given
        if init is not None:
            base_init = self.ssm.initialization
            self.ssm.initialization = init
        # Perform smoothing, only saving what is required
        res = self.ssm.smooth(
            SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_STATE_AUTOCOV,
            update_filter=False)
        res.llf_obs = np.array(
            self.ssm._kalman_filter.loglikelihood, copy=True)
        # Reset initialization
        if init is not None:
            self.ssm.initialization = base_init

        return res

    def _em_maximization_step(self, res, params0, mstep_method=None):
        a = res.smoothed_state.T[..., None]
        cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
        acov_a = res.smoothed_state_autocov.transpose(2, 0, 1)

        # E[a_t a_t'], t = 0, ..., T
        Eaa = cov_a.copy() + np.matmul(a, a.transpose(0, 2, 1))
        # E[a_t a_{t-1}'], t = 1, ..., T
        Eaa1 = acov_a[:-1] + np.matmul(a[1:], a[:-1].transpose(0, 2, 1))

        # Factor VAR and covariance
        A = Eaa[:-1, :1, :1].sum(axis=0)
        B = Eaa1[:, :1, :1].sum(axis=0)
        C = Eaa[1:, :1, :1].sum(axis=0)
        nobs = Eaa.shape[0] - 1

        f_A = B / A
        f_Q = (C - f_A @ B.T) / nobs
        params1 = np.zeros_like(params0)
        params1[0] = f_A[0, 0]
        params1[1] = f_Q[0, 0]

        return params1

    def transform_params(self, unconstrained):
        # array no longer accepts inhomogeneous inputs
        return np.hstack([
            constrain_stationary_univariate(unconstrained[:1]),
            unconstrained[1]**2])

    def untransform_params(self, constrained):
        # array no longer accepts inhomogeneous inputs
        return np.hstack([
            unconstrain_stationary_univariate(constrained[:1]),
            constrained[1] ** 0.5])

    def update(self, params, **kwargs):
        super().update(params, **kwargs)

        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]
