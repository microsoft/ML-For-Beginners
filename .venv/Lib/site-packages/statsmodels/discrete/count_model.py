__all__ = ["ZeroInflatedPoisson", "ZeroInflatedGeneralizedPoisson",
           "ZeroInflatedNegativeBinomialP"]

import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
                                                 Poisson, Logit, CountResults,
                                                 L1CountResults, Probit,
                                                 _discrete_results_docs,
                                                 _validate_l1_method,
                                                 GeneralizedPoisson,
                                                 NegativeBinomialP)
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender


_doc_zi_params = """
    exog_infl : array_like or None
        Explanatory variables for the binary inflation model, i.e. for
        mixing probability model. If None, then a constant is used.
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    exposure : array_like
        Log(exposure) is added to the linear prediction with coefficient
        equal to 1.
    inflation : {'logit', 'probit'}
        The model for the zero inflation, either Logit (default) or Probit
    """


class GenericZeroInflated(CountModel):
    __doc__ = """
    Generic Zero Inflated Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    exog_infl : ndarray
        A reference to the zero-inflated exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' : _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None,
                 inflation='logit', exposure=None, missing='none', **kwargs):
        super(GenericZeroInflated, self).__init__(endog, exog, offset=offset,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)

        if exog_infl is None:
            self.k_inflate = 1
            self._no_exog_infl = True
            self.exog_infl = np.ones((endog.size, self.k_inflate),
                                     dtype=np.float64)
        else:
            self.exog_infl = exog_infl
            self.k_inflate = exog_infl.shape[1]
            self._no_exog_infl = False

        if len(exog.shape) == 1:
            self.k_exog = 1
        else:
            self.k_exog = exog.shape[1]

        self.infl = inflation
        if inflation == 'logit':
            self.model_infl = Logit(np.zeros(self.exog_infl.shape[0]),
                                    self.exog_infl)
            self._hessian_inflate = self._hessian_logit
        elif inflation == 'probit':
            self.model_infl = Probit(np.zeros(self.exog_infl.shape[0]),
                                    self.exog_infl)
            self._hessian_inflate = self._hessian_probit

        else:
            raise ValueError("inflation == %s, which is not handled"
                             % inflation)

        self.inflation = inflation
        self.k_extra = self.k_inflate

        if len(self.exog) != len(self.exog_infl):
            raise ValueError('exog and exog_infl have different number of'
                             'observation. `missing` handling is not supported')

        infl_names = ['inflate_%s' % i for i in self.model_infl.data.param_names]
        self.exog_names[:] = infl_names + list(self.exog_names)
        self.exog_infl = np.asarray(self.exog_infl, dtype=np.float64)

        self._init_keys.extend(['exog_infl', 'inflation'])
        self._null_drop_keys = ['exog_infl']

    def _get_exogs(self):
        """list of exogs, for internal use in post-estimation
        """
        return (self.exog, self.exog_infl)

    def loglike(self, params):
        """
        Loglikelihood of Generic Zero Inflated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math:: \\ln L=\\sum_{y_{i}=0}\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\sum_{y_{i}>0}(\\ln(1-w_{i})+L_{main\\_model})
            where P - pdf of main model, L - loglike function of main model.
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Zero Inflated model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes for definition.

        Notes
        -----
        .. math:: \\ln L=\\ln(w_{i}+(1-w_{i})*P_{main\\_model})+
            \\ln(1-w_{i})+L_{main\\_model}
            where P - pdf of main model, L - loglike function of main model.

        for observations :math:`i=1,...,n`
        """
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)

        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        llf_main = self.model_main.loglikeobs(params_main)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        llf = np.zeros_like(y, dtype=np.float64)
        llf[zero_idx] = (np.log(w[zero_idx] +
            (1 - w[zero_idx]) * np.exp(llf_main[zero_idx])))
        llf[nonzero_idx] = np.log(1 - w[nonzero_idx]) + llf_main[nonzero_idx]

        return llf

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35,
            full_output=1, disp=1, callback=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            start_params = self._get_start_params()

        if callback is None:
            # work around perfect separation callback #3895
            callback = lambda *x: x

        mlefit = super(GenericZeroInflated, self).fit(start_params=start_params,
                       maxiter=maxiter, disp=disp, method=method,
                       full_output=full_output, callback=callback,
                       **kwargs)

        zipfit = self.result_class(self, mlefit._results)
        result = self.result_class_wrapper(zipfit)

        if cov_kwds is None:
            cov_kwds = {}

        result._get_robustcov_results(cov_type=cov_type,
                                      use_self=True, use_t=use_t, **cov_kwds)
        return result

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1',
            maxiter='defined_by_method', full_output=1, disp=1, callback=None,
            alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=1e-4,
            qc_tol=0.03, **kwargs):

        _validate_l1_method(method)

        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.k_exog + self.k_inflate
            alpha = alpha * np.ones(k_params)

        extra = self.k_extra - self.k_inflate
        alpha_p = alpha[:-(self.k_extra - extra)] if (self.k_extra
            and np.size(alpha) > 1) else alpha
        if start_params is None:
            offset = getattr(self, "offset", 0) + getattr(self, "exposure", 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            start_params = self.model_main.fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=0, callback=callback,
                alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(np.ones(self.k_inflate), start_params)
        cntfit = super(CountModel, self).fit_regularized(
                start_params=start_params, method=method, maxiter=maxiter,
                full_output=full_output, disp=disp, callback=callback,
                alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol,
                size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)

        discretefit = self.result_class_reg(self, cntfit)
        return self.result_class_reg_wrapper(discretefit)

    def score_obs(self, params):
        """
        Generic Zero Inflated model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score_main = self.model_main.score_obs(params_main)
        llf_main = self.model_main.loglikeobs(params_main)
        llf = self.loglikeobs(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        mu = self.model_main.predict(params_main)

        # TODO: need to allow for complex to use CS numerical derivatives
        dldp = np.zeros((self.exog.shape[0], self.k_exog), dtype=np.float64)
        dldw = np.zeros_like(self.exog_infl, dtype=np.float64)

        dldp[zero_idx,:] = (score_main[zero_idx].T *
                     (1 - (w[zero_idx]) / np.exp(llf[zero_idx]))).T
        dldp[nonzero_idx,:] = score_main[nonzero_idx]

        if self.inflation == 'logit':
            dldw[zero_idx,:] =  (self.exog_infl[zero_idx].T * w[zero_idx] *
                                 (1 - w[zero_idx]) *
                                 (1 - np.exp(llf_main[zero_idx])) /
                                  np.exp(llf[zero_idx])).T
            dldw[nonzero_idx,:] = -(self.exog_infl[nonzero_idx].T *
                                    w[nonzero_idx]).T
        elif self.inflation == 'probit':
            return approx_fprime(params, self.loglikeobs)

        return np.hstack((dldw, dldp))

    def score(self, params):
        return self.score_obs(params).sum(0)

    def _hessian_main(self, params):
        pass

    def _hessian_logit(self, params):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score_main = self.model_main.score_obs(params_main)
        llf_main = self.model_main.loglikeobs(params_main)
        llf = self.loglikeobs(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        hess_arr = np.zeros((self.k_inflate, self.k_exog + self.k_inflate))

        pmf = np.exp(llf)

        #d2l/dw2
        for i in range(self.k_inflate):
            for j in range(i, -1, -1):
                hess_arr[i, j] = ((
                    self.exog_infl[zero_idx, i] * self.exog_infl[zero_idx, j] *
                    (w[zero_idx] * (1 - w[zero_idx]) * ((1 -
                    np.exp(llf_main[zero_idx])) * (1 - 2 * w[zero_idx]) *
                    np.exp(llf[zero_idx]) - (w[zero_idx] - w[zero_idx]**2) *
                    (1 - np.exp(llf_main[zero_idx]))**2) /
                    pmf[zero_idx]**2)).sum() -
                    (self.exog_infl[nonzero_idx, i] * self.exog_infl[nonzero_idx, j] *
                    w[nonzero_idx] * (1 - w[nonzero_idx])).sum())

        #d2l/dpdw
        for i in range(self.k_inflate):
            for j in range(self.k_exog):
                hess_arr[i, j + self.k_inflate] = -(score_main[zero_idx, j] *
                    w[zero_idx] * (1 - w[zero_idx]) *
                    self.exog_infl[zero_idx, i] / pmf[zero_idx]).sum()

        return hess_arr

    def _hessian_probit(self, params):
        pass

    def hessian(self, params):
        """
        Generic Zero Inflated model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        hess_arr_main = self._hessian_main(params)
        hess_arr_infl = self._hessian_inflate(params)

        if hess_arr_main is None or hess_arr_infl is None:
            return approx_hess(params, self.loglike)

        dim = self.k_exog + self.k_inflate

        hess_arr = np.zeros((dim, dim))

        hess_arr[:self.k_inflate,:] = hess_arr_infl
        hess_arr[self.k_inflate:,self.k_inflate:] = hess_arr_main

        tri_idx = np.triu_indices(self.k_exog + self.k_inflate, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]

        return hess_arr

    def predict(self, params, exog=None, exog_infl=None, exposure=None,
                offset=None, which='mean', y_values=None):
        """
        Predict expected response or other statistic given exogenous variables.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor with coefficient
            equal to 1. If exposure is specified, then it will be logged by
            the method. The user does not need to log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.
        which : str (optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' : the conditional expectation of endog E(y | x). This
              takes inflated zeros into account.
            - 'linear' : the linear predictor of the mean function.
            - 'var' : returns the estimated variance of endog implied by the
              model.
            - 'mean-main' : mean of the main count model
            - 'prob-main' : probability of selecting the main model.
                The probability of zero inflation is ``1 - prob-main``.
            - 'mean-nonzero' : expected value conditional on having observation
              larger than zero, E(y | X, y>0)
            - 'prob-zero' : probability of observing a zero count. P(y=0 | x)
            - 'prob' : probabilities of each count from 0 to max(endog), or
              for y_values if those are provided. This is a multivariate
              return (2-dim when predicting for several observations).

        y_values : array_like
            Values of the random variable endog at which pmf is evaluated.
            Only used if ``which="prob"``
        """
        no_exog = False
        if exog is None:
            no_exog = True
            exog = self.exog

        if exog_infl is None:
            if no_exog:
                exog_infl = self.exog_infl
            else:
                if self._no_exog_infl:
                    exog_infl = np.ones((len(exog), 1))
        else:
            exog_infl = np.asarray(exog_infl)
            if exog_infl.ndim == 1 and self.k_inflate == 1:
                exog_infl = exog_infl[:, None]

        if exposure is None:
            if no_exog:
                exposure = getattr(self, 'exposure', 0)
            else:
                exposure = 0
        else:
            exposure = np.log(exposure)

        if offset is None:
            if no_exog:
                offset = getattr(self, 'offset', 0)
            else:
                offset = 0

        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        prob_main = 1 - self.model_infl.predict(params_infl, exog_infl)

        lin_pred = np.dot(exog, params_main[:self.exog.shape[1]]) + exposure + offset

        # Refactor: This is pretty hacky,
        # there should be an appropriate predict method in model_main
        # this is just prob(y=0 | model_main)
        tmp_exog = self.model_main.exog
        tmp_endog = self.model_main.endog
        tmp_offset = getattr(self.model_main, 'offset', False)
        tmp_exposure = getattr(self.model_main, 'exposure', False)
        self.model_main.exog = exog
        self.model_main.endog = np.zeros((exog.shape[0]))
        self.model_main.offset = offset
        self.model_main.exposure = exposure
        llf = self.model_main.loglikeobs(params_main)
        self.model_main.exog = tmp_exog
        self.model_main.endog = tmp_endog
        # tmp_offset might be an array with elementwise equality testing
        #if np.size(tmp_offset) == 1 and tmp_offset[0] == 'no':
        if tmp_offset is False:
            del self.model_main.offset
        else:
            self.model_main.offset = tmp_offset
        #if np.size(tmp_exposure) == 1 and tmp_exposure[0] == 'no':
        if tmp_exposure is False:
            del self.model_main.exposure
        else:
            self.model_main.exposure = tmp_exposure
        # end hack

        prob_zero = (1 - prob_main) + prob_main * np.exp(llf)

        if which == 'mean':
            return prob_main * np.exp(lin_pred)
        elif which == 'mean-main':
            return np.exp(lin_pred)
        elif which == 'linear':
            return lin_pred
        elif which == 'mean-nonzero':
            return prob_main * np.exp(lin_pred) / (1 - prob_zero)
        elif which == 'prob-zero':
            return prob_zero
        elif which == 'prob-main':
            return prob_main
        elif which == 'var':
            mu = np.exp(lin_pred)
            return self._predict_var(params, mu, 1 - prob_main)
        elif which == 'prob':
            return self._predict_prob(params, exog, exog_infl, exposure,
                                      offset, y_values=y_values)
        else:
            raise ValueError('which = %s is not available' % which)

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """NotImplemented
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, transform="dydx",
                         dummy_idx=None, count_idx=None):
        """NotImplemented
        """
        raise NotImplementedError

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main)

        score_infl = self.model_infl._deriv_mean_dparams(params_infl)
        score_main = self.model_main._deriv_mean_dparams(params_main)

        dmat_infl = - mu[:, None] * score_infl
        dmat_main = (1 - w[:, None]) * score_main

        dmat = np.column_stack((dmat_infl, dmat_main))
        return dmat

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        raise NotImplementedError

        # The below currently does not work, discontinuity at zero
        # see https://github.com/statsmodels/statsmodels/pull/7951#issuecomment-996355875  # noqa
        from statsmodels.tools.numdiff import _approx_fprime_scalar
        endog_original = self.endog

        def f(y):
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            self.endog = y
            self.model_main.endog = y
            sf = self.score_obs(params)
            self.endog = endog_original
            self.model_main.endog = endog_original
            return sf

        ds = _approx_fprime_scalar(self.endog[:, None], f, epsilon=1e-2)

        return ds


class ZeroInflatedPoisson(GenericZeroInflated):
    __doc__ = """
    Poisson Zero Inflated Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    exog_infl : ndarray
        A reference to the zero-inflated exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' : _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', missing='none', **kwargs):
        super(ZeroInflatedPoisson, self).__init__(endog, exog, offset=offset,
                                                  inflation=inflation,
                                                  exog_infl=exog_infl,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, offset=offset,
                                  exposure=exposure)
        self.distribution = zipoisson
        self.result_class = ZeroInflatedPoissonResults
        self.result_class_wrapper = ZeroInflatedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedPoissonResultsWrapper

    def _hessian_main(self, params):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score = self.score(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]

        mu = self.model_main.predict(params_main)

        hess_arr = np.zeros((self.k_exog, self.k_exog))

        coeff = (1 + w[zero_idx] * (np.exp(mu[zero_idx]) - 1))

        #d2l/dp2
        for i in range(self.k_exog):
            for j in range(i, -1, -1):
                hess_arr[i, j] = ((
                    self.exog[zero_idx, i] * self.exog[zero_idx, j] *
                    mu[zero_idx] * (w[zero_idx] - 1) * (1 / coeff -
                    w[zero_idx] * mu[zero_idx] * np.exp(mu[zero_idx]) /
                    coeff**2)).sum() - (mu[nonzero_idx] * self.exog[nonzero_idx, i] *
                    self.exog[nonzero_idx, j]).sum())

        return hess_arr

    def _predict_prob(self, params, exog, exog_infl, exposure, offset,
                      y_values=None):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        if y_values is None:
            y_values = np.atleast_2d(np.arange(0, np.max(self.endog)+1))

        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(
                self.model_infl.predict(params_infl, exog_infl))[:, None]
        else:
            transform = False
            w = self.model_infl.predict(params_infl, exog_infl)[:, None]

        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main, exog,
            offset=offset)[:, None]
        result = self.distribution.pmf(y_values, mu, w)
        return result[0] if transform else result

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        w = prob_infl
        var_ = (1 - w) * mu * (1 + w * mu)
        return var_

    def _get_start_params(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            start_params = self.model_main.fit(disp=0, method="nm").params
        start_params = np.append(np.ones(self.k_inflate) * 0.1, start_params)
        return start_params

    def get_distribution(self, params, exog=None, exog_infl=None,
                         exposure=None, offset=None):
        """Get frozen instance of distribution based on predicted parameters.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor  of the mean
            function with coefficient equal to 1. If exposure is specified,
            then it will be logged by the method. The user does not need to
            log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.

        Returns
        -------
        Instance of frozen scipy distribution subclass.
        """
        mu = self.predict(params, exog=exog, exog_infl=exog_infl,
                          exposure=exposure, offset=offset, which="mean-main")
        w = self.predict(params, exog=exog, exog_infl=exog_infl,
                         exposure=exposure, offset=offset, which="prob-main")

        # distr = self.distribution(mu[:, None], 1 - w[:, None])
        distr = self.distribution(mu, 1 - w)
        return distr


class ZeroInflatedGeneralizedPoisson(GenericZeroInflated):
    __doc__ = """
    Zero Inflated Generalized Poisson Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    exog_infl : ndarray
        A reference to the zero-inflated exogenous design.
    p : scalar
        P denotes parametrizations for ZIGP regression.
    """ % {'params' : base._model_params_doc,
           'extra_params' : _doc_zi_params +
           """p : float
        dispersion power parameter for the GeneralizedPoisson model.  p=1 for
        ZIGP-1 and p=2 for ZIGP-2. Default is p=2
    """ + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', p=2, missing='none', **kwargs):
        super(ZeroInflatedGeneralizedPoisson, self).__init__(endog, exog,
                                                  offset=offset,
                                                  inflation=inflation,
                                                  exog_infl=exog_infl,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog,
            offset=offset, exposure=exposure, p=p)
        self.distribution = zigenpoisson
        self.k_exog += 1
        self.k_extra += 1
        self.exog_names.append("alpha")
        self.result_class = ZeroInflatedGeneralizedPoissonResults
        self.result_class_wrapper = ZeroInflatedGeneralizedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedGeneralizedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedGeneralizedPoissonResultsWrapper

    def _get_init_kwds(self):
        kwds = super(ZeroInflatedGeneralizedPoisson, self)._get_init_kwds()
        kwds['p'] = self.model_main.parameterization + 1
        return kwds

    def _predict_prob(self, params, exog, exog_infl, exposure, offset,
                      y_values=None):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        p = self.model_main.parameterization + 1
        if y_values is None:
            y_values = np.atleast_2d(np.arange(0, np.max(self.endog)+1))

        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(
                self.model_infl.predict(params_infl, exog_infl))[:, None]
        else:
            transform = False
            w = self.model_infl.predict(params_infl, exog_infl)[:, None]

        w[w == 1.] = np.nextafter(1, 0)
        mu = self.model_main.predict(params_main, exog,
            exposure=exposure, offset=offset)[:, None]
        result = self.distribution.pmf(y_values, mu, params_main[-1], p, w)
        return result[0] if transform else result

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        alpha = params[-1]
        w = prob_infl
        p = self.model_main.parameterization
        var_ = (1 - w) * mu * ((1 + alpha * mu**p)**2 + w * mu)
        return var_

    def _get_start_params(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            start_params = ZeroInflatedPoisson(self.endog, self.exog,
                exog_infl=self.exog_infl).fit(disp=0).params
        start_params = np.append(start_params, 0.1)
        return start_params

    @Appender(ZeroInflatedPoisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exog_infl=None,
                         exposure=None, offset=None):

        p = self.model_main.parameterization + 1
        mu = self.predict(params, exog=exog, exog_infl=exog_infl,
                          exposure=exposure, offset=offset, which="mean-main")
        w = self.predict(params, exog=exog, exog_infl=exog_infl,
                         exposure=exposure, offset=offset, which="prob-main")
        # distr = self.distribution(mu[:, None], params[-1], p, 1 - w[:, None])
        distr = self.distribution(mu, params[-1], p, 1 - w)
        return distr


class ZeroInflatedNegativeBinomialP(GenericZeroInflated):
    __doc__ = """
    Zero Inflated Generalized Negative Binomial Model

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    endog : ndarray
        A reference to the endogenous response variable
    exog : ndarray
        A reference to the exogenous design.
    exog_infl : ndarray
        A reference to the zero-inflated exogenous design.
    p : scalar
        P denotes parametrizations for ZINB regression. p=1 for ZINB-1 and
    p=2 for ZINB-2. Default is p=2
    """ % {'params' : base._model_params_doc,
           'extra_params' : _doc_zi_params +
           """p : float
        dispersion power parameter for the NegativeBinomialP model.  p=1 for
        ZINB-1 and p=2 for ZINM-2. Default is p=2
    """ + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None,
                 inflation='logit', p=2, missing='none', **kwargs):
        super(ZeroInflatedNegativeBinomialP, self).__init__(endog, exog,
                                                  offset=offset,
                                                  inflation=inflation,
                                                  exog_infl=exog_infl,
                                                  exposure=exposure,
                                                  missing=missing, **kwargs)
        self.model_main = NegativeBinomialP(self.endog, self.exog,
            offset=offset, exposure=exposure, p=p)
        self.distribution = zinegbin
        self.k_exog += 1
        self.k_extra += 1
        self.exog_names.append("alpha")
        self.result_class = ZeroInflatedNegativeBinomialResults
        self.result_class_wrapper = ZeroInflatedNegativeBinomialResultsWrapper
        self.result_class_reg = L1ZeroInflatedNegativeBinomialResults
        self.result_class_reg_wrapper = L1ZeroInflatedNegativeBinomialResultsWrapper

    def _get_init_kwds(self):
        kwds = super(ZeroInflatedNegativeBinomialP, self)._get_init_kwds()
        kwds['p'] = self.model_main.parameterization
        return kwds

    def _predict_prob(self, params, exog, exog_infl, exposure, offset,
                      y_values=None):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]

        p = self.model_main.parameterization
        if y_values is None:
            y_values = np.arange(0, np.max(self.endog)+1)

        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(
                self.model_infl.predict(params_infl, exog_infl))[:, None]
        else:
            transform = False
            w = self.model_infl.predict(params_infl, exog_infl)[:, None]

        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main, exog,
            exposure=exposure, offset=offset)[:, None]
        result = self.distribution.pmf(y_values, mu, params_main[-1], p, w)
        return result[0] if transform else result

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        alpha = params[-1]
        w = prob_infl
        p = self.model_main.parameterization
        var_ = (1 - w) * mu * (1 + alpha * mu**(p - 1) + w * mu)
        return var_

    def _get_start_params(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            start_params = self.model_main.fit(disp=0, method='nm').params
        start_params = np.append(np.zeros(self.k_inflate), start_params)
        return start_params

    @Appender(ZeroInflatedPoisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exog_infl=None,
                         exposure=None, offset=None):

        p = self.model_main.parameterization
        mu = self.predict(params, exog=exog, exog_infl=exog_infl,
                          exposure=exposure, offset=offset, which="mean-main")
        w = self.predict(params, exog=exog, exog_infl=exog_infl,
                         exposure=exposure, offset=offset, which="prob-main")

        # distr = self.distribution(mu[:, None], params[-1], p, 1 - w[:, None])
        distr = self.distribution(mu, params[-1], p, 1 - w)
        return distr


class ZeroInflatedResults(CountResults):

    def get_prediction(self, exog=None, exog_infl=None, exposure=None,
                       offset=None, which='mean', average=False,
                       agg_weights=None, y_values=None,
                       transform=True, row_labels=None):

        import statsmodels.base._prediction_inference as pred

        pred_kwds = {
            'exog_infl': exog_infl,
            'exposure': exposure,
            'offset': offset,
            'y_values': y_values,
            }

        res = pred.get_prediction_delta(self, exog=exog, which=which,
                                        average=average,
                                        agg_weights=agg_weights,
                                        pred_kwds=pred_kwds)
        return res

    def get_influence(self):
        """
        Influence and outlier measures

        See notes section for influence measures that do not apply for
        zero inflated models.

        Returns
        -------
        MLEInfluence
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        ZeroInflated models have functions that are not differentiable
        with respect to sample endog if endog=0. This means that generalized
        leverage cannot be computed in the usual definition.

        Currently, both the generalized leverage, in `hat_matrix_diag`
        attribute and studetized residuals are not available. In the influence
        plot generalized leverage is replaced by a hat matrix diagonal that
        only takes combined exog into account, computed in the same way as
        for OLS. This is a measure for exog outliers but does not take
        specific features of the model into account.
        """
        # same as sumper in DiscreteResults, only added for docstring
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)


class ZeroInflatedPoissonResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {
    "one_line_description": "A results class for Zero Inflated Poisson",
    "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        mu = self.predict(which='linear')
        w = 1 - self.predict() / np.exp(self.predict(which='linear'))
        return (1 + w * np.exp(mu))

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedPoissonResults(L1CountResults, ZeroInflatedPoissonResults):
    pass


class ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedPoissonResultsWrapper,
                      ZeroInflatedPoissonResults)


class L1ZeroInflatedPoissonResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedPoissonResultsWrapper,
                      L1ZeroInflatedPoissonResults)


class ZeroInflatedGeneralizedPoissonResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Zero Inflated Generalized Poisson",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        p = self.model.model_main.parameterization
        alpha = self.params[self.model.k_inflate:][-1]
        mu = np.exp(self.predict(which='linear'))
        w = 1 - self.predict() / mu
        return ((1 + alpha * mu**p)**2 + w * mu)

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedGeneralizedPoissonResults(L1CountResults,
        ZeroInflatedGeneralizedPoissonResults):
    pass


class ZeroInflatedGeneralizedPoissonResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedGeneralizedPoissonResultsWrapper,
                      ZeroInflatedGeneralizedPoissonResults)


class L1ZeroInflatedGeneralizedPoissonResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedGeneralizedPoissonResultsWrapper,
                      L1ZeroInflatedGeneralizedPoissonResults)


class ZeroInflatedNegativeBinomialResults(ZeroInflatedResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description": "A results class for Zero Inflated Generalized Negative Binomial",
        "extra_attr": ""}

    @cache_readonly
    def _dispersion_factor(self):
        p = self.model.model_main.parameterization
        alpha = self.params[self.model.k_inflate:][-1]
        mu = np.exp(self.predict(which='linear'))
        w = 1 - self.predict() / mu
        return (1 + alpha * mu**(p-1) + w * mu)

    def get_margeff(self, at='overall', method='dydx', atexog=None,
            dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Not yet implemented for Zero Inflated Models
        """
        raise NotImplementedError("not yet implemented for zero inflation")


class L1ZeroInflatedNegativeBinomialResults(L1CountResults,
        ZeroInflatedNegativeBinomialResults):
    pass


class ZeroInflatedNegativeBinomialResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(ZeroInflatedNegativeBinomialResultsWrapper,
                      ZeroInflatedNegativeBinomialResults)


class L1ZeroInflatedNegativeBinomialResultsWrapper(
        lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(L1ZeroInflatedNegativeBinomialResultsWrapper,
                      L1ZeroInflatedNegativeBinomialResults)
