# -*- coding: utf-8 -*-
"""
Generalized Additive Models

Author: Luca Puggini
Author: Josef Perktold

created on 08/07/2015
"""

from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7

import numpy as np
from scipy import optimize
import pandas as pd

import statsmodels.base.wrapper as wrap

from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
    GLM, GLMResults, GLMResultsWrapper, _check_convergence)
import statsmodels.regression.linear_model as lm
# import statsmodels.regression._tools as reg_tools  # TODO: use this for pirls
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
                                             ValueWarning)
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt

from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
    MultivariateGAMCVPath)
from statsmodels.gam.gam_cross_validation.cross_validators import KFold


def _transform_predict_exog(model, exog, design_info=None):
    """transform exog for predict using design_info

    Note: this is copied from base.model.Results.predict and converted to
    standalone function with additional options.
    """

    is_pandas = _is_using_pandas(exog, None)

    exog_index = exog.index if is_pandas else None

    if design_info is None:
        design_info = getattr(model.data, 'design_info', None)

    if design_info is not None and (exog is not None):
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            # we are guessing whether it should be column or row
            if (hasattr(exog, 'name') and isinstance(exog.name, str) and
                    exog.name in design_info.describe()):
                # assume we need one column
                exog = pd.DataFrame(exog)
            else:
                # assume we need a row
                exog = pd.DataFrame(exog).T
        orig_exog_len = len(exog)
        is_dict = isinstance(exog, dict)
        exog = dmatrix(design_info, exog, return_type="dataframe")
        if orig_exog_len > len(exog) and not is_dict:
            import warnings
            if exog_index is None:
                warnings.warn('nan values have been dropped', ValueWarning)
            else:
                exog = exog.reindex(exog_index)
        exog_index = exog.index

    if exog is not None:
        exog = np.asarray(exog)
        if exog.ndim == 1 and (model.exog.ndim == 1 or
                               model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]

    return exog, exog_index


class GLMGamResults(GLMResults):
    """Results class for generalized additive models, GAM.

    This inherits from GLMResults.

    Warning: some inherited methods might not correctly take account of the
    penalization

    GLMGamResults inherits from GLMResults
    All methods related to the loglikelihood function return the penalized
    values.

    Attributes
    ----------

    edf
        list of effective degrees of freedom for each column of the design
        matrix.
    hat_matrix_diag
        diagonal of hat matrix
    gcv
        generalized cross-validation criterion computed as
        ``gcv = scale / (1. - hat_matrix_trace / self.nobs)**2``
    cv
        cross-validation criterion computed as
        ``cv = ((resid_pearson / (1 - hat_matrix_diag))**2).sum() / nobs``

    Notes
    -----
    status: experimental
    """

    def __init__(self, model, params, normalized_cov_params, scale, **kwds):

        # this is a messy way to compute edf and update scale
        # need several attributes to compute edf
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        edf = self.edf.sum()
        self.df_model = edf - 1  # assume constant
        # need to use nobs or wnobs attribute
        self.df_resid = self.model.endog.shape[0] - edf

        # we are setting the model df for the case when super is using it
        # df in model will be incorrect state when alpha/pen_weight changes
        self.model.df_model = self.df_model
        self.model.df_resid = self.df_resid
        mu = self.fittedvalues
        self.scale = scale = self.model.estimate_scale(mu)
        super(GLMGamResults, self).__init__(model, params,
                                            normalized_cov_params, scale,
                                            **kwds)

    def _tranform_predict_exog(self, exog=None, exog_smooth=None,
                               transform=True):
        """Transform original explanatory variables for prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is False, then ``exog`` is returned unchanged and
            ``x`` is ignored. It is assumed that exog contains the full
            design matrix for the predict observations.
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.

        Returns
        -------
        exog_transformed : ndarray
            design matrix for the prediction
        """
        if exog_smooth is not None:
            exog_smooth = np.asarray(exog_smooth)
        exog_index = None
        if transform is False:
            # the following allows that either or both exog are not None
            if exog_smooth is None:
                # exog could be None or array
                ex = exog
            else:
                if exog is None:
                    ex = exog_smooth
                else:
                    ex = np.column_stack((exog, exog_smooth))
        else:
            # transform exog_linear if needed
            if exog is not None and hasattr(self.model, 'design_info_linear'):
                exog, exog_index = _transform_predict_exog(
                    self.model, exog, self.model.design_info_linear)

            # create smooth basis
            if exog_smooth is not None:
                ex_smooth = self.model.smoother.transform(exog_smooth)
                if exog is None:
                    ex = ex_smooth
                else:
                    # TODO: there might be problems is exog_smooth is 1-D
                    ex = np.column_stack((exog, ex_smooth))
            else:
                ex = exog

        return ex, exog_index

    def predict(self, exog=None, exog_smooth=None, transform=True, **kwargs):
        """"
        compute prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``exog``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction : ndarray, pandas.Series or pandas.DataFrame
            predicted values
        """
        ex, exog_index = self._tranform_predict_exog(exog=exog,
                                                     exog_smooth=exog_smooth,
                                                     transform=transform)
        predict_results = super(GLMGamResults, self).predict(ex,
                                                             transform=False,
                                                             **kwargs)
        if exog_index is not None and not hasattr(
                predict_results, 'predicted_values'):
            if predict_results.ndim == 1:
                return pd.Series(predict_results, index=exog_index)
            else:
                return pd.DataFrame(predict_results, index=exog_index)
        else:
            return predict_results

    def get_prediction(self, exog=None, exog_smooth=None, transform=True,
                       **kwargs):
        """compute prediction results

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction_results : generalized_linear_model.PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary tables for the prediction of the mean and of new
            observations.
        """
        ex, exog_index = self._tranform_predict_exog(exog=exog,
                                                     exog_smooth=exog_smooth,
                                                     transform=transform)
        return super(GLMGamResults, self).get_prediction(ex, transform=False,
                                                         **kwargs)

    def partial_values(self, smooth_index, include_constant=True):
        """contribution of a smooth term to the linear prediction

        Warning: This will be replaced by a predict method

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.

        Returns
        -------
        predicted : nd_array
            predicted value of linear term.
            This is not the expected response if the link function is not
            linear.
        se_pred : nd_array
            standard error of linear prediction
        """
        variable = smooth_index
        smoother = self.model.smoother
        mask = smoother.mask[variable]

        start_idx = self.model.k_exog_linear
        idx = start_idx + np.nonzero(mask)[0]

        # smoother has only smooth parts, not exog_linear
        exog_part = smoother.basis[:, mask]

        const_idx = self.model.data.const_idx
        if include_constant and const_idx is not None:
            idx = np.concatenate(([const_idx], idx))
            exog_part = self.model.exog[:, idx]

        linpred = np.dot(exog_part, self.params[idx])
        # select the submatrix corresponding to a single variable
        partial_cov_params = self.cov_params(column=idx)

        covb = partial_cov_params
        var = (exog_part * np.dot(covb, exog_part.T).T).sum(1)
        se = np.sqrt(var)

        return linpred, se

    def plot_partial(self, smooth_index, plot_se=True, cpr=False,
                     include_constant=True, ax=None):
        """plot the contribution of a smooth term to the linear prediction

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        plot_se : bool
            If plot_se is true, then the confidence interval for the linear
            prediction will be added to the plot.
        cpr : bool
            If cpr (component plus residual) is true, the a scatter plot of
            the partial working residuals will be added to the plot.
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.
        ax : None or matplotlib axis instance
           If ax is not None, then the plot will be added to it.

        Returns
        -------
        Figure
            If `ax` is None, the created figure. Otherwise the Figure to which
            `ax` is connected.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_ax
        _import_mpl()

        variable = smooth_index
        y_est, se = self.partial_values(variable,
                                        include_constant=include_constant)
        smoother = self.model.smoother
        x = smoother.smoothers[variable].x
        sort_index = np.argsort(x)
        x = x[sort_index]
        y_est = y_est[sort_index]
        se = se[sort_index]

        fig, ax = create_mpl_ax(ax)
        ax.plot(x, y_est, c='blue', lw=2)
        if plot_se:
            ax.plot(x, y_est + 1.96 * se, '-', c='blue')
            ax.plot(x, y_est - 1.96 * se, '-', c='blue')
        if cpr:
            # TODO: resid_response does not make sense with nonlinear link
            # use resid_working ?
            residual = self.resid_working[sort_index]
            cpr_ = y_est + residual
            ax.plot(x, cpr_, '.', lw=2)

        ax.set_xlabel(smoother.smoothers[variable].variable_name)

        return fig

    def test_significance(self, smooth_index):
        """hypothesis test that a smooth component is zero.

        This calls `wald_test` to compute the hypothesis test, but uses
        effective degrees of freedom.

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms

        Returns
        -------
        wald_test : ContrastResults instance
            the results instance created by `wald_test`
        """

        variable = smooth_index
        smoother = self.model.smoother
        start_idx = self.model.k_exog_linear

        k_params = len(self.params)
        # a bit messy, we need first index plus length of smooth term
        mask = smoother.mask[variable]
        k_constraints = mask.sum()
        idx = start_idx + np.nonzero(mask)[0][0]
        constraints = np.eye(k_constraints, k_params, idx)
        df_constraints = self.edf[idx: idx + k_constraints].sum()

        return self.wald_test(constraints, df_constraints=df_constraints)

    def get_hat_matrix_diag(self, observed=True, _axis=1):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.
            This is only relevant for models that implement both observed
            and expected Hessian, which is currently only GLM. Other
            models only use the observed Hessian.
        _axis : int
            This is mainly for internal use. By default it returns the usual
            diagonal of the hat matrix. If _axis is zero, then the result
            corresponds to the effective degrees of freedom, ``edf`` for each
            column of exog.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        weights = self.model.hessian_factor(self.params, scale=self.scale,
                                            observed=observed)
        wexog = np.sqrt(weights)[:, None] * self.model.exog

        # we can use inverse hessian directly instead of computing it from
        # WLS/IRLS as in GLM

        # TODO: does `normalized_cov_params * scale` work in all cases?
        # this avoids recomputing hessian, check when used for other models.
        hess_inv = self.normalized_cov_params * self.scale
        # this is in GLM equivalent to the more generic and direct
        # hess_inv = np.linalg.inv(-self.model.hessian(self.params))
        hd = (wexog * hess_inv.dot(wexog.T).T).sum(axis=_axis)
        return hd

    @cache_readonly
    def edf(self):
        return self.get_hat_matrix_diag(_axis=0)

    @cache_readonly
    def hat_matrix_trace(self):
        return self.hat_matrix_diag.sum()

    @cache_readonly
    def hat_matrix_diag(self):
        return self.get_hat_matrix_diag(observed=True)

    @cache_readonly
    def gcv(self):
        return self.scale / (1. - self.hat_matrix_trace / self.nobs)**2

    @cache_readonly
    def cv(self):
        cv_ = ((self.resid_pearson / (1. - self.hat_matrix_diag))**2).sum()
        cv_ /= self.nobs
        return cv_


class GLMGamResultsWrapper(GLMResultsWrapper):
    pass


wrap.populate_wrapper(GLMGamResultsWrapper, GLMGamResults)


class GLMGam(PenalizedMixin, GLM):
    """
    Generalized Additive Models (GAM)

    This inherits from `GLM`.

    Warning: Not all inherited methods might take correctly account of the
    penalization. Not all options including offset and exposure have been
    verified yet.

    Parameters
    ----------
    endog : array_like
        The response variable.
    exog : array_like or None
        This explanatory variables are treated as linear. The model in this
        case is a partial linear model.
    smoother : instance of additive smoother class
        Examples of smoother instances include Bsplines or CyclicCubicSplines.
    alpha : float or list of floats
        Penalization weights for smooth terms. The length of the list needs
        to be the same as the number of smooth terms in the ``smoother``.
    family : instance of GLM family
        See GLM.
    offset : None or array_like
        See GLM.
    exposure : None or array_like
        See GLM.
    missing : 'none'
        Missing value handling is not supported in this class.
    **kwargs
        Extra keywords are used in call to the super classes.

    Notes
    -----
    Status: experimental. This has full unit test coverage for the core
    results with Gaussian and Poisson (without offset and exposure). Other
    options and additional results might not be correctly supported yet.
    (Binomial with counts, i.e. with n_trials, is most likely wrong in pirls.
    User specified var or freq weights are most likely also not correct for
    all results.)
    """

    _results_class = GLMGamResults
    _results_class_wrapper = GLMGamResultsWrapper

    def __init__(self, endog, exog=None, smoother=None, alpha=0, family=None,
                 offset=None, exposure=None, missing='none', **kwargs):

        # TODO: check usage of hasconst
        hasconst = kwargs.get('hasconst', None)
        xnames_linear = None
        if hasattr(exog, 'design_info'):
            self.design_info_linear = exog.design_info
            xnames_linear = self.design_info_linear.column_names

        is_pandas = _is_using_pandas(exog, None)

        # TODO: handle data is experimental, see #5469
        # This is a bit wasteful because we need to `handle_data twice`
        self.data_linear = self._handle_data(endog, exog, missing, hasconst)
        if xnames_linear is None:
            xnames_linear = self.data_linear.xnames
        if exog is not None:
            exog_linear = self.data_linear.exog
            k_exog_linear = exog_linear.shape[1]
        else:
            exog_linear = None
            k_exog_linear = 0
        self.k_exog_linear = k_exog_linear
        # We need exog_linear for k-fold cross validation
        # TODO: alternative is to take columns from combined exog
        self.exog_linear = exog_linear

        self.smoother = smoother
        self.k_smooths = smoother.k_variables
        self.alpha = self._check_alpha(alpha)
        penal = MultivariateGamPenalty(smoother, alpha=self.alpha,
                                       start_idx=k_exog_linear)
        kwargs.pop('penal', None)
        if exog_linear is not None:
            exog = np.column_stack((exog_linear, smoother.basis))
        else:
            exog = smoother.basis

        # TODO: check: xnames_linear will be None instead of empty list
        #       if no exog_linear
        # can smoother be empty ? I guess not allowed.
        if xnames_linear is None:
            xnames_linear = []
        xnames = xnames_linear + self.smoother.col_names

        if is_pandas and exog_linear is not None:
            # we a dataframe so we can get a PandasData instance for wrapping
            exog = pd.DataFrame(exog, index=self.data_linear.row_labels,
                                columns=xnames)

        super(GLMGam, self).__init__(endog, exog=exog, family=family,
                                     offset=offset, exposure=exposure,
                                     penal=penal, missing=missing, **kwargs)

        if not is_pandas:
            # set exog nanmes if not given by pandas DataFrame
            self.exog_names[:] = xnames

        # TODO: the generic data handling might attach the design_info from the
        #       linear part, but this is incorrect for the full model and
        #       causes problems in wald_test_terms

        if hasattr(self.data, 'design_info'):
            del self.data.design_info
        # formula also might be attached which causes problems in predict
        if hasattr(self, 'formula'):
            self.formula_linear = self.formula
            self.formula = None
            del self.formula

    def _check_alpha(self, alpha):
        """check and convert alpha to required list format

        Parameters
        ----------
        alpha : scalar, list or array_like
            penalization weight

        Returns
        -------
        alpha : list
            penalization weight, list with length equal to the number of
            smooth terms
        """
        if not isinstance(alpha, Iterable):
            alpha = [alpha] * len(self.smoother.smoothers)
        elif not isinstance(alpha, list):
            # we want alpha to be a list
            alpha = list(alpha)
        return alpha

    def fit(self, start_params=None, maxiter=1000, method='pirls', tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, **kwargs):
        """estimate parameters and create instance of GLMGamResults class

        Parameters
        ----------
        most parameters are the same as for GLM
        method : optimization method
            The special optimization method is "pirls" which uses a penalized
            version of IRLS. Other methods are gradient optimizers as used in
            base.model.LikelihoodModel.

        Returns
        -------
        res : instance of wrapped GLMGamResults
        """
        # TODO: temporary hack to remove attribute
        # formula also might be attached which in inherited from_formula
        # causes problems in predict
        if hasattr(self, 'formula'):
            self.formula_linear = self.formula
            del self.formula

        # TODO: alpha not allowed yet, but is in `_fit_pirls`
        # alpha = self._check_alpha()

        if method.lower() in ['pirls', 'irls']:
            res = self._fit_pirls(self.alpha, start_params=start_params,
                                  maxiter=maxiter, tol=tol, scale=scale,
                                  cov_type=cov_type, cov_kwds=cov_kwds,
                                  use_t=use_t, **kwargs)
        else:
            if max_start_irls > 0 and (start_params is None):
                res = self._fit_pirls(self.alpha, start_params=start_params,
                                      maxiter=max_start_irls, tol=tol,
                                      scale=scale,
                                      cov_type=cov_type, cov_kwds=cov_kwds,
                                      use_t=use_t, **kwargs)
                start_params = res.params
                del res
            res = super(GLMGam, self).fit(start_params=start_params,
                                          maxiter=maxiter, method=method,
                                          tol=tol, scale=scale,
                                          cov_type=cov_type, cov_kwds=cov_kwds,
                                          use_t=use_t,
                                          full_output=full_output, disp=disp,
                                          max_start_irls=0,
                                          **kwargs)
        return res

    # pag 165 4.3 # pag 136 PIRLS
    def _fit_pirls(self, alpha, start_params=None, maxiter=100, tol=1e-8,
                   scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
                   weights=None):
        """fit model with penalized reweighted least squares
        """
        # TODO: this currently modifies several attributes
        # self.scale, self.scaletype, self.mu, self.weights
        # self.data_weights,
        # and possibly self._offset_exposure
        # several of those might not be necessary, e.g. mu and weights

        # alpha = alpha * len(y) * self.scale / 100
        # TODO: we need to rescale alpha
        endog = self.endog
        wlsexog = self.exog  # smoother.basis
        spl_s = self.penal.penalty_matrix(alpha=alpha)

        nobs, n_columns = wlsexog.shape

        # TODO what are these values?
        if weights is None:
            self.data_weights = np.array([1.] * nobs)
        else:
            self.data_weights = weights

        if not hasattr(self, '_offset_exposure'):
            self._offset_exposure = 0

        self.scaletype = scale
        # TODO: check default scale types
        # self.scaletype = 'dev'
        # during iteration
        self.scale = 1

        if start_params is None:
            mu = self.family.starting_mu(endog)
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        dev = self.family.deviance(endog, mu)

        history = dict(params=[None, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0

        for iteration in range(maxiter):

            # TODO: is this equivalent to point 1 of page 136:
            # w = 1 / (V(mu) * g'(mu))  ?
            self.weights = self.data_weights * self.family.weights(mu)

            # TODO: is this equivalent to point 1 of page 136:
            # z = g(mu)(y - mu) + X beta  ?
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (endog - mu)
                        - self._offset_exposure)

            # this defines the augmented matrix point 2a on page 136
            wls_results = penalized_wls(wlsendog, wlsexog, spl_s, self.weights)
            lin_pred = np.dot(wlsexog, wls_results.params).ravel()
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)

            # We do not need to update scale in GLM/LEF models
            # We might need it in dispersion models.
            # self.scale = self.estimate_scale(mu)
            history = self._update_history(wls_results, mu, history)

            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)

            # TODO need atol, rtol
            # args of _check_convergence: (criterion, iteration, atol, rtol)
            converged = _check_convergence(criterion, iteration, tol, 0)
            if converged:
                break
        self.mu = mu
        self.scale = self.estimate_scale(mu)
        glm_results = GLMGamResults(self, wls_results.params,
                                    wls_results.normalized_cov_params,
                                    self.scale,
                                    cov_type=cov_type, cov_kwds=cov_kwds,
                                    use_t=use_t)

        glm_results.method = "PIRLS"
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged

        return GLMGamResultsWrapper(glm_results)

    def select_penweight(self, criterion='aic', start_params=None,
                         start_model_params=None,
                         method='basinhopping', **fit_kwds):
        """find alpha by minimizing results criterion

        The objective for the minimization can be results attributes like
        ``gcv``, ``aic`` or ``bic`` where the latter are based on effective
        degrees of freedom.

        Warning: In many case the optimization might converge to a local
        optimum or near optimum. Different start_params or using a global
        optimizer is recommended, default is basinhopping.

        Parameters
        ----------
        criterion='aic'
            name of results attribute to be minimized.
            Default is 'aic', other options are 'gcv', 'cv' or 'bic'.
        start_params : None or array
            starting parameters for alpha in the penalization weight
            minimization. The parameters are internally exponentiated and
            the minimization is with respect to ``exp(alpha)``
        start_model_params : None or array
            starting parameter for the ``model._fit_pirls``.
        method : 'basinhopping', 'nm' or 'minimize'
            'basinhopping' and 'nm' directly use the underlying scipy.optimize
            functions `basinhopping` and `fmin`. 'minimize' provides access
            to the high level interface, `scipy.optimize.minimize`.
        fit_kwds : keyword arguments
            additional keyword arguments will be used in the call to the
            scipy optimizer. Which keywords are supported depends on the
            scipy optimization function.

        Returns
        -------
        alpha : ndarray
            penalization parameter found by minimizing the criterion.
            Note that this can be only a local (near) optimum.
        fit_res : tuple
            results returned by the scipy optimization routine. The
            parameters in the optimization problem are `log(alpha)`
        history : dict
            history of calls to pirls and contains alpha, the fit
            criterion and the parameters to which pirls converged to for the
            given alpha.

        Notes
        -----
        In the test cases Nelder-Mead and bfgs often converge to local optima,
        see also https://github.com/statsmodels/statsmodels/issues/5381.

        This does not use any analytical derivatives for the criterion
        minimization.

        Status: experimental, It is possible that defaults change if there
        is a better way to find a global optimum. API (e.g. type of return)
        might also change.
        """
        # copy attributes that are changed, so we can reset them
        scale_keep = self.scale
        scaletype_keep = self.scaletype
        # TODO: use .copy() method when available for all types
        alpha_keep = copy.copy(self.alpha)

        if start_params is None:
            start_params = np.zeros(self.k_smooths)
        else:
            start_params = np.log(1e-20 + start_params)

        history = {}
        history['alpha'] = []
        history['params'] = [start_model_params]
        history['criterion'] = []

        def fun(p):
            a = np.exp(p)
            res_ = self._fit_pirls(start_params=history['params'][-1],
                                   alpha=a)
            history['alpha'].append(a)
            history['params'].append(np.asarray(res_.params))
            return getattr(res_, criterion)

        if method == 'nm':
            kwds = dict(full_output=True, maxiter=1000, maxfun=2000)
            kwds.update(fit_kwds)
            fit_res = optimize.fmin(fun, start_params, **kwds)
            opt = fit_res[0]
        elif method == 'basinhopping':
            kwds = dict(minimizer_kwargs={'method': 'Nelder-Mead',
                        'options': {'maxiter': 100, 'maxfev': 500}},
                        niter=10)
            kwds.update(fit_kwds)
            fit_res = optimize.basinhopping(fun, start_params, **kwds)
            opt = fit_res.x
        elif method == 'minimize':
            fit_res = optimize.minimize(fun, start_params, **fit_kwds)
            opt = fit_res.x
        else:
            raise ValueError('method not recognized')

        del history['params'][0]  # remove the model start_params

        alpha = np.exp(opt)

        # reset attributes that have or might have changed
        self.scale = scale_keep
        self.scaletype = scaletype_keep
        self.alpha = alpha_keep

        return alpha, fit_res, history

    def select_penweight_kfold(self, alphas=None, cv_iterator=None, cost=None,
                               k_folds=5, k_grid=11):
        """find alphas by k-fold cross-validation

        Warning: This estimates ``k_folds`` models for each point in the
            grid of alphas.

        Parameters
        ----------
        alphas : None or list of arrays
        cv_iterator : instance
            instance of a cross-validation iterator, by default this is a
            KFold instance
        cost : function
            default is mean squared error. The cost function to evaluate the
            prediction error for the left out sample. This should take two
            arrays as argument and return one float.
        k_folds : int
            number of folds if default Kfold iterator is used.
            This is ignored if ``cv_iterator`` is not None.

        Returns
        -------
        alpha_cv : list of float
            Best alpha in grid according to cross-validation
        res_cv : instance of MultivariateGAMCVPath
            The instance was used for cross-validation and holds the results

        Notes
        -----
        The default alphas are defined as
        ``alphas = [np.logspace(0, 7, k_grid) for _ in range(k_smooths)]``
        """

        if cost is None:
            def cost(x1, x2):
                return np.linalg.norm(x1 - x2) / len(x1)

        if alphas is None:
            alphas = [np.logspace(0, 7, k_grid) for _ in range(self.k_smooths)]

        if cv_iterator is None:
            cv_iterator = KFold(k_folds=k_folds, shuffle=True)

        gam_cv = MultivariateGAMCVPath(smoother=self.smoother, alphas=alphas,
                                       gam=GLMGam, cost=cost, endog=self.endog,
                                       exog=self.exog_linear,
                                       cv_iterator=cv_iterator)
        gam_cv_res = gam_cv.fit()

        return gam_cv_res.alpha_cv, gam_cv_res


class LogitGam(PenalizedMixin, Logit):
    """Generalized Additive model for discrete Logit

    This subclasses discrete_model Logit.

    Warning: not all inherited methods might take correctly account of the
    penalization

    not verified yet.
    """
    def __init__(self, endog, smoother, alpha, *args, **kwargs):
        if not isinstance(alpha, Iterable):
            alpha = np.array([alpha] * len(smoother.smoothers))

        self.smoother = smoother
        self.alpha = alpha
        self.pen_weight = 1  # TODO: pen weight should not be defined here!!
        penal = MultivariateGamPenalty(smoother, alpha=alpha)

        super(LogitGam, self).__init__(endog, smoother.basis, penal=penal,
                                       *args, **kwargs)


def penalized_wls(endog, exog, penalty_matrix, weights):
    """weighted least squares with quadratic penalty

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization. Note, the penalty_matrix
        is multiplied by two to match non-pirls fitting methods.
    weights : ndarray
        weights for WLS

    Returns
    -------
    results : Results instance of WLS
    """
    y, x, s = endog, exog, penalty_matrix
    # TODO: I do not understand why I need 2 * s
    aug_y, aug_x, aug_weights = make_augmented_matrix(y, x, 2 * s, weights)
    wls_results = lm.WLS(aug_y, aug_x, aug_weights).fit()
    # TODO: use MinimalWLS during iterations, less overhead
    # However, MinimalWLS does not return normalized_cov_params
    #   which we need at the end of the iterations
    # call would be
    # wls_results = reg_tools._MinimalWLS(aug_y, aug_x, aug_weights).fit()
    wls_results.params = wls_results.params.ravel()

    return wls_results


def make_augmented_matrix(endog, exog, penalty_matrix, weights):
    """augment endog, exog and weights with stochastic restriction matrix

    Parameters
    ----------
    endog : ndarray
        response or endogenous variable
    exog : ndarray
        design matrix, matrix of exogenous or explanatory variables
    penalty_matrix : ndarray, 2-Dim square
        penality matrix for quadratic penalization
    weights : ndarray
        weights for WLS

    Returns
    -------
    endog_aug : ndarray
        augmented response variable
    exog_aug : ndarray
        augmented design matrix
    weights_aug : ndarray
        augmented weights for WLS
    """
    y, x, s, = endog, exog, penalty_matrix
    nobs = x.shape[0]

    # TODO: needs full because of broadcasting with weights
    # check what weights should be doing
    rs = matrix_sqrt(s)
    x1 = np.vstack([x, rs])  # augmented x
    n_samp1es_x1 = x1.shape[0]

    y1 = np.array([0.] * n_samp1es_x1)  # augmented y
    y1[:nobs] = y

    id1 = np.array([1.] * rs.shape[0])
    w1 = np.concatenate([weights, id1])

    return y1, x1, w1
