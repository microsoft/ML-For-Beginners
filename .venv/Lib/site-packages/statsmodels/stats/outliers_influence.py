# -*- coding: utf-8 -*-
"""Influence and Outlier Measures

Created on Sun Jan 29 11:16:09 2012

Author: Josef Perktold
License: BSD-3
"""

import warnings

from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip

from collections import defaultdict

import numpy as np

from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results

# outliers test convenience wrapper

def outlier_test(model_results, method='bonf', alpha=.05, labels=None,
                 order=False, cutoff=None):
    """
    Outlier Tests for RegressionResults instances.

    Parameters
    ----------
    model_results : RegressionResults
        Linear model results
    method : str
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` :
        - `holm` :
        - `simes-hochberg` :
        - `hommel` :
        - `fdr_bh` : Benjamini/Hochberg
        - `fdr_by` : Benjamini/Yekutieli
        See `statsmodels.stats.multitest.multipletests` for details.
    alpha : float
        familywise error rate
    labels : None or array_like
        If `labels` is not None, then it will be used as index to the
        returned pandas DataFrame. See also Returns below
    order : bool
        Whether or not to order the results by the absolute value of the
        studentized residuals. If labels are provided they will also be sorted.
    cutoff : None or float in [0, 1]
        If cutoff is not None, then the return only includes observations with
        multiple testing corrected p-values strictly below the cutoff. The
        returned array or dataframe can be empty if there are no outlier
        candidates at the specified cutoff.

    Returns
    -------
    table : ndarray or DataFrame
        Returns either an ndarray or a DataFrame if labels is not None.
        Will attempt to get labels from model_results if available. The
        columns are the Studentized residuals, the unadjusted p-value,
        and the corrected p-value according to method.

    Notes
    -----
    The unadjusted p-value is stats.t.sf(abs(resid), df) where
    df = df_resid - 1.
    """
    from scipy import stats  # lazy import
    if labels is None:
        labels = getattr(model_results.model.data, 'row_labels', None)
    infl = getattr(model_results, 'get_influence', None)
    if infl is None:
        results = maybe_unwrap_results(model_results)
        raise AttributeError("model_results object %s does not have a "
                             "get_influence "
                             "method." % results.__class__.__name__)
    resid = infl().resid_studentized_external
    if order:
        idx = np.abs(resid).argsort()[::-1]
        resid = resid[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]
    df = model_results.df_resid - 1
    unadj_p = stats.t.sf(np.abs(resid), df) * 2
    adj_p = multipletests(unadj_p, alpha=alpha, method=method)

    data = np.c_[resid, unadj_p, adj_p[1]]
    if cutoff is not None:
        mask = data[:, -1] < cutoff
        data = data[mask]
    else:
        mask = slice(None)

    if labels is not None:
        from pandas import DataFrame
        return DataFrame(data,
                         columns=['student_resid', 'unadj_p', method + "(p)"],
                         index=np.asarray(labels)[mask])
    return data


# influence measures

def reset_ramsey(res, degree=5):
    """Ramsey's RESET specification test for linear models

    This is a general specification test, for additional non-linear effects
    in a model.

    Parameters
    ----------
    degree : int
        Maximum power to include in the RESET test.  Powers 0 and 1 are
        excluded, so that degree tests powers 2, ..., degree of the fitted
        values.

    Notes
    -----
    The test fits an auxiliary OLS regression where the design matrix, exog,
    is augmented by powers 2 to degree of the fitted values. Then it performs
    an F-test whether these additional terms are significant.

    If the p-value of the f-test is below a threshold, e.g. 0.1, then this
    indicates that there might be additional non-linear effects in the model
    and that the linear model is mis-specified.

    References
    ----------
    https://en.wikipedia.org/wiki/Ramsey_RESET_test
    """
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    # vander without constant and x, and drop constant
    norm_values = np.asarray(res.fittedvalues)
    norm_values = norm_values / np.sqrt((norm_values ** 2).mean())
    y_fitted_vander = np.vander(norm_values, order)[:, :-2]
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    exog /= np.sqrt((exog ** 2).mean(0))
    endog = res.model.endog / (res.model.endog ** 2).mean()
    res_aux = OLS(endog, exog).fit()
    # r_matrix = np.eye(degree, exog.shape[1], k_vars)
    r_matrix = np.eye(degree - 1, exog.shape[1], k_vars)
    # df1 = degree - 1
    # df2 = exog.shape[0] - degree - res.df_model  (without constant)
    return res_aux.f_test(r_matrix)  # , r_matrix, res_aux


def variance_inflation_factor(exog, exog_idx):
    """
    Variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : {ndarray, DataFrame}
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: does not exist yet

    References
    ----------
    https://en.wikipedia.org/wiki/Variance_inflation_factor
    """
    k_vars = exog.shape[1]
    exog = np.asarray(exog)
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


class _BaseInfluenceMixin:
    """common methods between OLSInfluence and MLE/GLMInfluence
    """

    @Appender(_plot_influence_doc.format(**{'extra_params_doc': ""}))
    def plot_influence(self, external=None, alpha=.05, criterion="cooks",
                       size=48, plot_alpha=.75, ax=None, **kwargs):

        if external is None:
            external = hasattr(self, '_cache') and 'res_looo' in self._cache
        from statsmodels.graphics.regressionplots import _influence_plot
        if self.hat_matrix_diag is not None:
            res = _influence_plot(self.results, self, external=external,
                                  alpha=alpha,
                                  criterion=criterion, size=size,
                                  plot_alpha=plot_alpha, ax=ax, **kwargs)
        else:
            warnings.warn("Plot uses pearson residuals and exog hat matrix.")
            res = _influence_plot(self.results, self, external=external,
                                  alpha=alpha,
                                  criterion=criterion, size=size,
                                  leverage=self.hat_matrix_exog_diag,
                                  resid=self.resid,
                                  plot_alpha=plot_alpha, ax=ax, **kwargs)
        return res

    def _plot_index(self, y, ylabel, threshold=None, title=None, ax=None,
                    **kwds):
        from statsmodels.graphics import utils
        fig, ax = utils.create_mpl_ax(ax)
        if title is None:
            title = "Index Plot"
        nobs = len(self.endog)
        index = np.arange(nobs)
        ax.scatter(index, y, **kwds)

        if threshold == 'all':
            large_points = np.ones(nobs, np.bool_)
        else:
            large_points = np.abs(y) > threshold
        psize = 3 * np.ones(nobs)
        # add point labels
        labels = self.results.model.data.row_labels
        if labels is None:
            labels = np.arange(nobs)
        ax = utils.annotate_axes(np.where(large_points)[0], labels,
                                 lzip(index, y),
                                 lzip(-psize, psize), "large",
                                 ax)

        font = {"fontsize": 16, "color": "black"}
        ax.set_ylabel(ylabel, **font)
        ax.set_xlabel("Observation", **font)
        ax.set_title(title, **font)
        return fig

    def plot_index(self, y_var='cooks', threshold=None, title=None, ax=None,
                   idx=None, **kwds):
        """index plot for influence attributes

        Parameters
        ----------
        y_var : str
            Name of attribute or shortcut for predefined attributes that will
            be plotted on the y-axis.
        threshold : None or float
            Threshold for adding annotation with observation labels.
            Observations for which the absolute value of the y_var is larger
            than the threshold will be annotated. Set to a negative number to
            label all observations or to a large number to have no annotation.
        title : str
            If provided, the title will replace the default "Index Plot" title.
        ax : matplolib axis instance
            The plot will be added to the `ax` if provided, otherwise a new
            figure is created.
        idx : {None, int}
            Some attributes require an additional index to select the y-var.
            In dfbetas this refers to the column indes.
        kwds : optional keywords
            Keywords will be used in the call to matplotlib scatter function.
        """
        criterion = y_var  # alias
        if threshold is None:
            # TODO: criterion specific defaults
            threshold = 'all'

        if criterion == 'dfbeta':
            y = self.dfbetas[:, idx]
            ylabel = 'DFBETA for ' + self.results.model.exog_names[idx]
        elif criterion.startswith('cook'):
            y = self.cooks_distance[0]
            ylabel = "Cook's distance"
        elif criterion.startswith('hat') or criterion.startswith('lever'):
            y = self.hat_matrix_diag
            ylabel = "Leverage (diagonal of hat matrix)"
        elif criterion.startswith('cook'):
            y = self.cooks_distance[0]
            ylabel = "Cook's distance"
        elif criterion.startswith('resid_stu'):
            y = self.resid_studentized
            ylabel = "Internally Studentized Residuals"
        else:
            # assume we have the name of an attribute
            y = getattr(self, y_var)
            if idx is not None:
                y = y[idx]
            ylabel = y_var

        fig = self._plot_index(y, ylabel, threshold=threshold, title=title,
                               ax=ax, **kwds)
        return fig


class MLEInfluence(_BaseInfluenceMixin):
    """Global Influence and outlier measures (experimental)

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments :
        Those are only available to override default behavior and are used
        instead of the corresponding attribute of the results class.
        By default resid_pearson is used as resid.

    Attributes
    ----------
    hat_matrix_diag (hii) : This is the generalized leverage computed as the
        local derivative of fittedvalues (predicted mean) with respect to the
        observed response for each observation.
        Not available for ZeroInflated models because of nondifferentiability.
    d_params : Change in parameters computed with one Newton step using the
        full Hessian corrected by division by (1 - hii).
        If hat_matrix_diag is not available, then the division by (1 - hii) is
        not included.
    dbetas : change in parameters divided by the standard error of parameters
        from the full model results, ``bse``.
    cooks_distance : quadratic form for change in parameters weighted by
        ``cov_params`` from the full model divided by the number of variables.
        It includes p-values based on the F-distribution which are only
        approximate outside of linear Gaussian models.
    resid_studentized : In the general MLE case resid_studentized are
        computed from the score residuals scaled by hessian factor and
        leverage. This does not use ``cov_params``.
    d_fittedvalues : local change of expected mean given the change in the
        parameters as computed in ``d_params``.
    d_fittedvalues_scaled : same as d_fittedvalues but scaled by the standard
        errors of a predicted mean of the response.
    params_one : is the one step parameter estimate computed as ``params``
        from the full sample minus ``d_params``.

    Notes
    -----
    MLEInfluence uses generic definitions based on maximum likelihood models.

    MLEInfluence produces the same results as GLMInfluence for canonical
    links (verified for GLM Binomial, Poisson and Gaussian). There will be
    some differences for non-canonical links or if a robust cov_type is used.
    For example, the generalized leverage differs from the definition of the
    GLM hat matrix in the case of Probit, which corresponds to family
    Binomial with a non-canonical link.

    The extension to non-standard models, e.g. multi-link model like
    BetaModel and the ZeroInflated models is still experimental and might still
    change.
    Additonally, ZeroInflated and some threshold models have a
    nondifferentiability in the generalized leverage. How this case is treated
    might also change.

    Warning: This does currently not work for constrained or penalized models,
    e.g. models estimated with fit_constrained or fit_regularized.

    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    status: experimental,
    This class will need changes to support different kinds of models, e.g.
    extra parameters in discrete.NegativeBinomial or two-part models like
    ZeroInflatedPoisson.
    """

    def __init__(self, results, resid=None, endog=None, exog=None,
                 hat_matrix_diag=None, cov_params=None, scale=None):
        # this __init__ attaches attributes that we don't really need
        self.results = results = maybe_unwrap_results(results)
        # TODO: check for extra params in e.g. NegBin
        self.nobs, self.k_vars = results.model.exog.shape
        self.k_params = np.size(results.params)
        self.endog = endog if endog is not None else results.model.endog
        self.exog = exog if exog is not None else results.model.exog
        self.scale = scale if scale is not None else results.scale
        if resid is not None:
            self.resid = resid
        else:
            self.resid = getattr(results, "resid_pearson", None)
            if self.resid is not None: # and scale != 1:
                # GLM and similar does not divide resid_pearson by scale
                self.resid = self.resid / np.sqrt(self.scale)

        self.cov_params = (cov_params if cov_params is not None
                           else results.cov_params())
        self.model_class = results.model.__class__

        self.hessian = self.results.model.hessian(self.results.params)
        self.score_obs = self.results.model.score_obs(self.results.params)
        if hat_matrix_diag is not None:
            self._hat_matrix_diag = hat_matrix_diag

    @cache_readonly
    def hat_matrix_diag(self):
        """Diagonal of the generalized leverage

        This is the analogue of the hat matrix diagonal for general MLE.
        """
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag

        try:
            dsdy = self.results.model._deriv_score_obs_dendog(
                self.results.params)
        except NotImplementedError:
            dsdy = None

        if dsdy is None:
            warnings.warn("hat matrix is not available, missing derivatives",
                          UserWarning)
            return None

        dmu_dp = self.results.model._deriv_mean_dparams(self.results.params)

        # dmu_dp = 1 /
        #      self.results.model.family.link.deriv(self.results.fittedvalues)
        h = (dmu_dp * np.linalg.solve(-self.hessian, dsdy.T).T).sum(1)
        return h

    @cache_readonly
    def hat_matrix_exog_diag(self):
        """Diagonal of the hat_matrix using only exog as in OLS

        """
        get_exogs = getattr(self.results.model, "_get_exogs", None)
        if get_exogs is not None:
            exog = np.column_stack(get_exogs())
        else:
            exog = self.exog
        return (exog * np.linalg.pinv(exog).T).sum(1)

    @cache_readonly
    def d_params(self):
        """Approximate change in parameter estimates when dropping observation.

        This uses one-step approximation of the parameter change to deleting
        one observation.
        """
        so_noti = self.score_obs.sum(0) - self.score_obs
        beta_i = np.linalg.solve(self.hessian, so_noti.T).T
        if self.hat_matrix_diag is not None:
            beta_i /= (1 - self.hat_matrix_diag)[:, None]

        return beta_i

    @cache_readonly
    def dfbetas(self):
        """Scaled change in parameter estimates.

        The one-step change of parameters in d_params is rescaled by dividing
        by the standard error of the parameter estimate given by results.bse.
        """

        beta_i = self.d_params / self.results.bse
        return beta_i

    @cache_readonly
    def params_one(self):
        """Parameter estimate based on one-step approximation.

        This the one step parameter estimate computed as
        ``params`` from the full sample minus ``d_params``.
        """
        return self.results.params - self.d_params

    @cache_readonly
    def cooks_distance(self):
        """Cook's distance and p-values.

        Based on one step approximation d_params and on results.cov_params
        Cook's distance divides by the number of explanatory variables.

        p-values are based on the F-distribution which are only approximate
        outside of linear Gaussian models.

        Warning: The definition of p-values might change if we switch to using
        chi-square distribution instead of F-distribution, or if we make it
        dependent on the fit keyword use_t.
        """
        cooks_d2 = (self.d_params * np.linalg.solve(self.cov_params,
                                                    self.d_params.T).T).sum(1)
        cooks_d2 /= self.k_params
        from scipy import stats

        # alpha = 0.1
        # print stats.f.isf(1-alpha, n_params, res.df_modelwc)
        # TODO use chi2   # use_f option
        pvals = stats.f.sf(cooks_d2, self.k_params, self.results.df_resid)

        return cooks_d2, pvals

    @cache_readonly
    def resid_studentized(self):
        """studentized default residuals.

        This uses the residual in `resid` attribute, which is by default
        resid_pearson and studentizes is using the generalized leverage.

        self.resid / np.sqrt(1 - self.hat_matrix_diag)

        Studentized residuals are not available if hat_matrix_diag is None.

        """
        return self.resid / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score_factor(self):
        """Score residual divided by sqrt of hessian factor.

        experimental, agrees with GLMInfluence for Binomial and Gaussian.
        This corresponds to considering the linear predictors as parameters
        of the model.

        Note: Nhis might have nan values if second derivative, hessian_factor,
        is positive, i.e. loglikelihood is not globally concave w.r.t. linear
        predictor. (This occured in an example for GeneralizedPoisson)
        """
        from statsmodels.genmod.generalized_linear_model import GLM
        sf = self.results.model.score_factor(self.results.params)
        hf = self.results.model.hessian_factor(self.results.params)
        if isinstance(sf, tuple):
            sf = sf[0]
        if isinstance(hf, tuple):
            hf = hf[0]
        if not isinstance(self.results.model, GLM):
            # hessian_factor in GLM has wrong sign, is already positive
            hf = -hf

        return sf / np.sqrt(hf) / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score(self, joint=True, index=None, studentize=False):
        """Score observations scaled by inverse hessian.

        Score residual in resid_score are defined in analogy to a score test
        statistic for each observation.

        Parameters
        ----------
        joint : bool
            If joint is true, then a quadratic form similar to score_test is
            returned for each observation.
            If joint is false, then standardized score_obs are returned. The
            returned array is two-dimensional
        index : ndarray (optional)
            Optional index to select a subset of score_obs columns.
            By default, all columns of score_obs will be used.
        studentize : bool
            If studentize is true, the the scaled residuals are also
            studentized using the generalized leverage.

        Returns
        -------
        array :  1-D or 2-D residuals

        Notes
        -----
        Status: experimental

        Because of the one srep approacimation of d_params, score residuals
        are identical to cooks_distance, except for

        - cooks_distance is normalized by the number of parameters
        - cooks_distance uses cov_params, resid_score is based on Hessian.
          This will make them differ in the case of robust cov_params.

        """
        # currently no caching
        score_obs = self.results.model.score_obs(self.results.params)
        hess = self.results.model.hessian(self.results.params)
        if index is not None:
            score_obs = score_obs[:, index]
            hess = hess[index[:, None], index]

        if joint:
            resid = (score_obs.T * np.linalg.solve(-hess, score_obs.T)).sum(0)
        else:
            resid = score_obs / np.sqrt(np.diag(-hess))

        if studentize:
            if joint:
                resid /= np.sqrt(1 - self.hat_matrix_diag)
            else:
                # 2-dim resid
                resid /= np.sqrt(1 - self.hat_matrix_diag[:, None])

        return resid

    @cache_readonly
    def _get_prediction(self):
        # TODO: do we cache this or does it need to be a method
        # we only need unchanging parts, alpha for confint could change
        with warnings.catch_warnings():
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.filterwarnings("ignore", message=msg,
                                    category=FutureWarning)
            pred = self.results.get_prediction()
        return pred

    @cache_readonly
    def d_fittedvalues(self):
        """Change in expected response, fittedvalues.

        Local change of expected mean given the change in the parameters as
        computed in d_params.

        Notes
        -----
        This uses the one-step approximation of the parameter change to
        deleting one observation ``d_params``.
        """
        # results.params might be a pandas.Series
        params = np.asarray(self.results.params)
        deriv = self.results.model._deriv_mean_dparams(params)
        return (deriv * self.d_params).sum(1)

    @property
    def d_fittedvalues_scaled(self):
        """
        Change in fittedvalues scaled by standard errors.

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``, and divides by the standard errors
        for the predicted mean provided by results.get_prediction.
        """
        # Note: this and the previous methods are for the response
        # and not for a weighted response, i.e. not the self.exog, self.endog
        # this will be relevant for WLS comparing fitted endog versus wendog
        return self.d_fittedvalues / self._get_prediction.se

    def summary_frame(self):
        """
        Creates a DataFrame with influence results.

        Returns
        -------
        frame : pandas DataFrame
            A DataFrame with selected results for each observation.
            The index will be the same as provided to the model.

        Notes
        -----
        The resultant DataFrame contains six variables in addition to the
        ``dfbetas``. These are:

        * cooks_d : Cook's Distance defined in ``cooks_distance``
        * standard_resid : Standardized residuals defined in
          `resid_studentizedl`
        * hat_diag : The diagonal of the projection, or hat, matrix defined in
          `hat_matrix_diag`. Not included if None.
        * dffits_internal : DFFITS statistics using internally Studentized
          residuals defined in `d_fittedvalues_scaled`
        """
        from pandas import DataFrame

        # row and column labels
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]

        # grab the results
        if self.hat_matrix_diag is not None:
            summary_data = DataFrame(dict(
                cooks_d=self.cooks_distance[0],
                standard_resid=self.resid_studentized,
                hat_diag=self.hat_matrix_diag,
                dffits_internal=self.d_fittedvalues_scaled),
                index=row_labels)
        else:
            summary_data = DataFrame(dict(
                cooks_d=self.cooks_distance[0],
                # standard_resid=self.resid_studentized,
                # hat_diag=self.hat_matrix_diag,
                dffits_internal=self.d_fittedvalues_scaled),
                index=row_labels)

        # NOTE: if we do not give columns, order of above will be arbitrary
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels,
                           index=row_labels)

        return dfbeta.join(summary_data)


class OLSInfluence(_BaseInfluenceMixin):
    """class to calculate outlier and influence measures for OLS result

    Parameters
    ----------
    results : RegressionResults
        currently assumes the results are from an OLS regression

    Notes
    -----
    One part of the results can be calculated without any auxiliary regression
    (some of which have the `_internal` postfix in the name. Other statistics
    require leave-one-observation-out (LOOO) auxiliary regression, and will be
    slower (mainly results with `_external` postfix in the name).
    The auxiliary LOOO regression only the required results are stored.

    Using the LOO measures is currently only recommended if the data set
    is not too large. One possible approach for LOOO measures would be to
    identify possible problem observations with the _internal measures, and
    then run the leave-one-observation-out only with observations that are
    possible outliers. (However, this is not yet available in an automated way.)

    This should be extended to general least squares.

    The leave-one-variable-out (LOVO) auxiliary regression are currently not
    used.
    """

    def __init__(self, results):
        # check which model is allowed
        self.results = maybe_unwrap_results(results)
        self.nobs, self.k_vars = results.model.exog.shape
        self.endog = results.model.endog
        self.exog = results.model.exog
        self.resid = results.resid
        self.model_class = results.model.__class__

        # self.sigma_est = np.sqrt(results.mse_resid)
        self.scale = results.mse_resid

        self.aux_regression_exog = {}
        self.aux_regression_endog = {}

    @cache_readonly
    def hat_matrix_diag(self):
        """Diagonal of the hat_matrix for OLS

        Notes
        -----
        temporarily calculated here, this should go to model class
        """
        return (self.exog * self.results.model.pinv_wexog.T).sum(1)

    @cache_readonly
    def resid_press(self):
        """PRESS residuals
        """
        hii = self.hat_matrix_diag
        return self.resid / (1 - hii)

    @cache_readonly
    def influence(self):
        """Influence measure

        matches the influence measure that gretl reports
        u * h / (1 - h)
        where u are the residuals and h is the diagonal of the hat_matrix
        """
        hii = self.hat_matrix_diag
        return self.resid * hii / (1 - hii)

    @cache_readonly
    def hat_diag_factor(self):
        """Factor of diagonal of hat_matrix used in influence

        this might be useful for internal reuse
        h / (1 - h)
        """
        hii = self.hat_matrix_diag
        return hii / (1 - hii)

    @cache_readonly
    def ess_press(self):
        """Error sum of squares of PRESS residuals
        """
        return np.dot(self.resid_press, self.resid_press)

    @cache_readonly
    def resid_studentized(self):
        """Studentized residuals using variance from OLS

        alias for resid_studentized_internal for compatibility with
        MLEInfluence this uses sigma from original estimate and does
        not require leave one out loop
        """
        return self.resid_studentized_internal

    @cache_readonly
    def resid_studentized_internal(self):
        """Studentized residuals using variance from OLS

        this uses sigma from original estimate
        does not require leave one out loop
        """
        return self.get_resid_studentized_external(sigma=None)
        # return self.results.resid / self.sigma_est

    @cache_readonly
    def resid_studentized_external(self):
        """Studentized residuals using LOOO variance

        this uses sigma from leave-one-out estimates

        requires leave one out loop for observations
        """
        sigma_looo = np.sqrt(self.sigma2_not_obsi)
        return self.get_resid_studentized_external(sigma=sigma_looo)

    def get_resid_studentized_external(self, sigma=None):
        """calculate studentized residuals

        Parameters
        ----------
        sigma : None or float
            estimate of the standard deviation of the residuals. If None, then
            the estimate from the regression results is used.

        Returns
        -------
        stzd_resid : ndarray
            studentized residuals

        Notes
        -----
        studentized residuals are defined as ::

           resid / sigma / np.sqrt(1 - hii)

        where resid are the residuals from the regression, sigma is an
        estimate of the standard deviation of the residuals, and hii is the
        diagonal of the hat_matrix.
        """
        hii = self.hat_matrix_diag
        if sigma is None:
            sigma2_est = self.scale
            # can be replace by different estimators of sigma
            sigma = np.sqrt(sigma2_est)

        return self.resid / sigma / np.sqrt(1 - hii)

    # same computation as GLMInfluence
    @cache_readonly
    def cooks_distance(self):
        """
        Cooks distance

        Uses original results, no nobs loop

        References
        ----------
        .. [*] Eubank, R. L. (1999). Nonparametric regression and spline
            smoothing. CRC press.
        .. [*] Cook's distance. (n.d.). In Wikipedia. July 2019, from
            https://en.wikipedia.org/wiki/Cook%27s_distance
        """
        hii = self.hat_matrix_diag
        # Eubank p.93, 94
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)

        from scipy import stats

        # alpha = 0.1
        # print stats.f.isf(1-alpha, n_params, res.df_modelwc)
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)

        return cooks_d2, pvals

    @cache_readonly
    def dffits_internal(self):
        """dffits measure for influence of an observation

        based on resid_studentized_internal
        uses original results, no nobs loop
        """
        # TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    @cache_readonly
    def dffits(self):
        """
        dffits measure for influence of an observation

        based on resid_studentized_external,
        uses results from leave-one-observation-out loop

        It is recommended that observations with dffits large than a
        threshold of 2 sqrt{k / n} where k is the number of parameters, should
        be investigated.

        Returns
        -------
        dffits : float
        dffits_threshold : float

        References
        ----------
        `Wikipedia <https://en.wikipedia.org/wiki/DFFITS>`_
        """
        # TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    @cache_readonly
    def dfbetas(self):
        """dfbetas

        uses results from leave-one-observation-out loop
        """
        dfbetas = self.results.params - self.params_not_obsi  # [None,:]
        dfbetas /= np.sqrt(self.sigma2_not_obsi[:, None])
        dfbetas /= np.sqrt(np.diag(self.results.normalized_cov_params))
        return dfbetas

    @cache_readonly
    def dfbeta(self):
        """dfbetas

        uses results from leave-one-observation-out loop
        """
        dfbeta = self.results.params - self.params_not_obsi
        return dfbeta

    @cache_readonly
    def sigma2_not_obsi(self):
        """error variance for all LOOO regressions

        This is 'mse_resid' from each auxiliary regression.

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['mse_resid'])

    @property
    def params_not_obsi(self):
        """parameter estimates for all LOOO regressions

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['params'])

    @property
    def det_cov_params_not_obsi(self):
        """determinant of cov_params of all LOOO regressions

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['det_cov_params'])

    @cache_readonly
    def cov_ratio(self):
        """covariance ratio between LOOO and original

        This uses determinant of the estimate of the parameter covariance
        from leave-one-out estimates.
        requires leave one out loop for observations
        """
        # do not use inplace division / because then we change original
        cov_ratio = (self.det_cov_params_not_obsi
                     / np.linalg.det(self.results.cov_params()))
        return cov_ratio

    @cache_readonly
    def resid_var(self):
        """estimate of variance of the residuals

        ::

           sigma2 = sigma2_OLS * (1 - hii)

        where hii is the diagonal of the hat matrix
        """
        # TODO:check if correct outside of ols
        return self.scale * (1 - self.hat_matrix_diag)

    @cache_readonly
    def resid_std(self):
        """estimate of standard deviation of the residuals

        See Also
        --------
        resid_var
        """
        return np.sqrt(self.resid_var)

    def _ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
        """regression results from LOVO auxiliary regression with cache


        The result instances are stored, which could use a large amount of
        memory if the datasets are large. There are too many combinations to
        store them all, except for small problems.

        Parameters
        ----------
        drop_idx : int
            index of exog that is dropped from the regression
        endog_idx : 'endog' or int
            If 'endog', then the endogenous variable of the result instance
            is regressed on the exogenous variables, excluding the one at
            drop_idx. If endog_idx is an integer, then the exog with that
            index is regressed with OLS on all other exogenous variables.
            (The latter is the auxiliary regression for the variance inflation
            factor.)

        this needs more thought, memory versus speed
        not yet used in any other parts, not sufficiently tested
        """
        # reverse the structure, access store, if fail calculate ?
        # this creates keys in store even if store = false ! bug
        if endog_idx == 'endog':
            stored = self.aux_regression_endog
            if hasattr(stored, drop_idx):
                return stored[drop_idx]
            x_i = self.results.model.endog

        else:
            # nested dictionary
            try:
                self.aux_regression_exog[endog_idx][drop_idx]
            except KeyError:
                pass

            stored = self.aux_regression_exog[endog_idx]
            stored = {}

            x_i = self.exog[:, endog_idx]

        k_vars = self.exog.shape[1]
        mask = np.arange(k_vars) != drop_idx
        x_noti = self.exog[:, mask]
        res = OLS(x_i, x_noti).fit()
        if store:
            stored[drop_idx] = res

        return res

    def _get_drop_vari(self, attributes):
        """
        regress endog on exog without one of the variables

        This uses a k_vars loop, only attributes of the OLS instance are
        stored.

        Parameters
        ----------
        attributes : list[str]
           These are the names of the attributes of the auxiliary OLS results
           instance that are stored and returned.

        not yet used
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        endog = self.results.model.endog
        exog = self.exog

        cv_iter = LeaveOneOut(self.k_vars)
        res_loo = defaultdict(list)
        for inidx, outidx in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog, exog[:, inidx]).fit()
                res_loo[att].append(getattr(res_i, att))

        return res_loo

    @cache_readonly
    def _res_looo(self):
        """collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        regresses endog on exog dropping one observation at a time

        this uses a nobs loop, only attributes of the OLS instance are stored.
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        def get_det_cov_params(res):
            return np.linalg.det(res.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog

        params = np.zeros(exog.shape, dtype=float)
        mse_resid = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)

        cv_iter = LeaveOneOut(self.nobs)
        for inidx, outidx in cv_iter:
            res_i = self.model_class(endog[inidx], exog[inidx]).fit()
            params[outidx] = res_i.params
            mse_resid[outidx] = res_i.mse_resid
            det_cov_params[outidx] = get_det_cov_params(res_i)

        return dict(params=params, mse_resid=mse_resid,
                    det_cov_params=det_cov_params)

    def summary_frame(self):
        """
        Creates a DataFrame with all available influence results.

        Returns
        -------
        frame : DataFrame
            A DataFrame with all results.

        Notes
        -----
        The resultant DataFrame contains six variables in addition to the
        DFBETAS. These are:

        * cooks_d : Cook's Distance defined in `Influence.cooks_distance`
        * standard_resid : Standardized residuals defined in
          `Influence.resid_studentized_internal`
        * hat_diag : The diagonal of the projection, or hat, matrix defined in
          `Influence.hat_matrix_diag`
        * dffits_internal : DFFITS statistics using internally Studentized
          residuals defined in `Influence.dffits_internal`
        * dffits : DFFITS statistics using externally Studentized residuals
          defined in `Influence.dffits`
        * student_resid : Externally Studentized residuals defined in
          `Influence.resid_studentized_external`
        """
        from pandas import DataFrame

        # row and column labels
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]

        # grab the results
        summary_data = DataFrame(dict(
            cooks_d=self.cooks_distance[0],
            standard_resid=self.resid_studentized_internal,
            hat_diag=self.hat_matrix_diag,
            dffits_internal=self.dffits_internal[0],
            student_resid=self.resid_studentized_external,
            dffits=self.dffits[0],
        ),
            index=row_labels)
        # NOTE: if we do not give columns, order of above will be arbitrary
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels,
                           index=row_labels)

        return dfbeta.join(summary_data)

    def summary_table(self, float_fmt="%6.3f"):
        """create a summary table with all influence and outlier measures

        This does currently not distinguish between statistics that can be
        calculated from the original regression results and for which a
        leave-one-observation-out loop is needed

        Returns
        -------
        res : SimpleTable
           SimpleTable instance with the results, can be printed

        Notes
        -----
        This also attaches table_data to the instance.
        """
        # print self.dfbetas

        #        table_raw = [ np.arange(self.nobs),
        #                      self.endog,
        #                      self.fittedvalues,
        #                      self.cooks_distance(),
        #                      self.resid_studentized_internal,
        #                      self.hat_matrix_diag,
        #                      self.dffits_internal,
        #                      self.resid_studentized_external,
        #                      self.dffits,
        #                      self.dfbetas
        #                      ]
        table_raw = [('obs', np.arange(self.nobs)),
                     ('endog', self.endog),
                     ('fitted\nvalue', self.results.fittedvalues),
                     ("Cook's\nd", self.cooks_distance[0]),
                     ("student.\nresidual", self.resid_studentized_internal),
                     ('hat diag', self.hat_matrix_diag),
                     ('dffits \ninternal', self.dffits_internal[0]),
                     ("ext.stud.\nresidual", self.resid_studentized_external),
                     ('dffits', self.dffits[0])
                     ]
        colnames, data = lzip(*table_raw)  # unzip
        data = np.column_stack(data)
        self.table_data = data
        from copy import deepcopy

        from statsmodels.iolib.table import SimpleTable, default_html_fmt
        from statsmodels.iolib.tableformatting import fmt_base
        fmt = deepcopy(fmt_base)
        fmt_html = deepcopy(default_html_fmt)
        fmt['data_fmts'] = ["%4d"] + [float_fmt] * (data.shape[1] - 1)
        # fmt_html['data_fmts'] = fmt['data_fmts']
        return SimpleTable(data, headers=colnames, txt_fmt=fmt,
                           html_fmt=fmt_html)


def summary_table(res, alpha=0.05):
    """
    Generate summary table of outlier and influence similar to SAS

    Parameters
    ----------
    alpha : float
       significance level for confidence interval

    Returns
    -------
    st : SimpleTable
       table with results that can be printed
    data : ndarray
       calculated measures and statistics for the table
    ss2 : list[str]
       column_names for table (Note: rows of table are observations)
    """

    from scipy import stats

    from statsmodels.sandbox.regression.predstd import wls_prediction_std

    infl = OLSInfluence(res)

    # standard error for predicted mean
    # Note: using hat_matrix only works for fitted values
    predict_mean_se = np.sqrt(infl.hat_matrix_diag * res.mse_resid)

    tppf = stats.t.isf(alpha / 2., res.df_resid)
    predict_mean_ci = np.column_stack([
        res.fittedvalues - tppf * predict_mean_se,
        res.fittedvalues + tppf * predict_mean_se])

    # standard error for predicted observation
    tmp = wls_prediction_std(res, alpha=alpha)
    predict_se, predict_ci_low, predict_ci_upp = tmp

    predict_ci = np.column_stack((predict_ci_low, predict_ci_upp))

    # standard deviation of residual
    resid_se = np.sqrt(res.mse_resid * (1 - infl.hat_matrix_diag))

    table_sm = np.column_stack([
        np.arange(res.nobs) + 1,
        res.model.endog,
        res.fittedvalues,
        predict_mean_se,
        predict_mean_ci[:, 0],
        predict_mean_ci[:, 1],
        predict_ci[:, 0],
        predict_ci[:, 1],
        res.resid,
        resid_se,
        infl.resid_studentized_internal,
        infl.cooks_distance[0]
    ])

    # colnames, data = lzip(*table_raw) #unzip
    data = table_sm
    ss2 = ['Obs', 'Dep Var\nPopulation', 'Predicted\nValue',
           'Std Error\nMean Predict', 'Mean ci\n95% low', 'Mean ci\n95% upp',
           'Predict ci\n95% low', 'Predict ci\n95% upp', 'Residual',
           'Std Error\nResidual', 'Student\nResidual', "Cook's\nD"]
    colnames = ss2
    # self.table_data = data
    # data = np.column_stack(data)
    from copy import deepcopy

    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ["%4d"] + ["%6.3f"] * (data.shape[1] - 1)
    # fmt_html['data_fmts'] = fmt['data_fmts']
    st = SimpleTable(data, headers=colnames, txt_fmt=fmt,
                     html_fmt=fmt_html)

    return st, data, ss2


class GLMInfluence(MLEInfluence):
    """Influence and outlier measures (experimental)

    This uses partly formulas specific to GLM, specifically cooks_distance
    is based on the hessian, i.e. observed or expected information matrix and
    not on cov_params, in contrast to MLEInfluence.
    Standardization for changes in parameters, in fittedvalues and in
    the linear predictor are based on cov_params.

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments are only to override default behavior and are used instead
    of the corresponding attribute of the results class.
    By default resid_pearson is used as resid.

    Attributes
    ----------
    dbetas
        change in parameters divided by the standard error of parameters from
        the full model results, ``bse``.
    d_fittedvalues_scaled
        same as d_fittedvalues but scaled by the standard errors of a
        predicted mean of the response.
    d_linpred
        local change in linear prediction.
    d_linpred_scale
        local change in linear prediction scaled by the standard errors for
        the prediction based on cov_params.

    Notes
    -----
    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    Some GLM specific measures like d_deviance are still missing.

    Computing an explicit leave-one-observation-out (LOOO) loop is included
    but no influence measures are currently computed from it.
    """

    @cache_readonly
    def hat_matrix_diag(self):
        """
        Diagonal of the hat_matrix for GLM

        Notes
        -----
        This returns the diagonal of the hat matrix that was provided as
        argument to GLMInfluence or computes it using the results method
        `get_hat_matrix`.
        """
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag
        else:
            return self.results.get_hat_matrix()

    @cache_readonly
    def d_params(self):
        """Change in parameter estimates

        Notes
        -----
        This uses one-step approximation of the parameter change to deleting
        one observation.
        """

        beta_i = np.linalg.pinv(self.exog) * self.resid_studentized
        beta_i /= np.sqrt(1 - self.hat_matrix_diag)
        return beta_i.T

    # same computation as OLS
    @cache_readonly
    def resid_studentized(self):
        """
        Internally studentized pearson residuals

        Notes
        -----
        residuals / sqrt( scale * (1 - hii))

        where residuals are those provided to GLMInfluence which are
        pearson residuals by default, and
        hii is the diagonal of the hat matrix.
        """
        # redundant with scaled resid_pearson, keep for docstring for now
        return super().resid_studentized

    # same computation as OLS
    @cache_readonly
    def cooks_distance(self):
        """Cook's distance

        Notes
        -----
        Based on one step approximation using resid_studentized and
        hat_matrix_diag for the computation.

        Cook's distance divides by the number of explanatory variables.

        Computed using formulas for GLM and does not use results.cov_params.
        It includes p-values based on the F-distribution which are only
        approximate outside of linear Gaussian models.
        """
        hii = self.hat_matrix_diag
        # Eubank p.93, 94
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)

        from scipy import stats

        # alpha = 0.1
        # print stats.f.isf(1-alpha, n_params, res.df_modelwc)
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)

        return cooks_d2, pvals

    @property
    def d_linpred(self):
        """
        Change in linear prediction

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``.
        """
        # TODO: This will need adjustment for extra params in Poisson
        # use original model exog not transformed influence exog
        exog = self.results.model.exog
        return (exog * self.d_params).sum(1)

    @property
    def d_linpred_scaled(self):
        """
        Change in linpred scaled by standard errors

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``, and divides by the standard errors
        for linpred provided by results.get_prediction.
        """
        # Note: this and the previous methods are for the response
        # and not for a weighted response, i.e. not the self.exog, self.endog
        # this will be relevant for WLS comparing fitted endog versus wendog
        return self.d_linpred / self._get_prediction.linpred.se

    @property
    def _fittedvalues_one(self):
        """experimental code
        """
        warnings.warn('this ignores offset and exposure', UserWarning)
        # TODO: we need to handle offset, exposure and weights
        # use original model exog not transformed influence exog
        exog = self.results.model.exog
        fitted = np.array([self.results.model.predict(pi, exog[i])
                           for i, pi in enumerate(self.params_one)])
        return fitted.squeeze()

    @property
    def _diff_fittedvalues_one(self):
        """experimental code
        """
        # in discrete we cannot reuse results.fittedvalues
        return self.results.predict() - self._fittedvalues_one

    @cache_readonly
    def _res_looo(self):
        """collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        Reestimates the model with endog and exog dropping one observation
        at a time

        This uses a nobs loop, only attributes of the results instance are
        stored.

        Warning: This will need refactoring and API changes to be able to
        add options.
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        get_det_cov_params = lambda res: np.linalg.det(res.cov_params())

        endog = self.results.model.endog
        exog = self.results.model.exog

        init_kwds = self.results.model._get_init_kwds()
        # We need to drop obs also from extra arrays
        freq_weights = init_kwds.pop('freq_weights')
        var_weights = init_kwds.pop('var_weights')
        offset = offset_ = init_kwds.pop('offset')
        exposure = exposure_ = init_kwds.pop('exposure')
        n_trials = init_kwds.pop('n_trials', None)
        # family Binomial creates `n` i.e. `n_trials`
        # we need to reset it
        # TODO: figure out how to do this properly
        if hasattr(init_kwds['family'], 'initialize'):
            # assume we have Binomial
            is_binomial = True
        else:
            is_binomial = False

        params = np.zeros(exog.shape, dtype=float)
        scale = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)

        cv_iter = LeaveOneOut(self.nobs)
        for inidx, outidx in cv_iter:
            if offset is not None:
                offset_ = offset[inidx]
            if exposure is not None:
                exposure_ = exposure[inidx]
            if n_trials is not None:
                init_kwds['n_trials'] = n_trials[inidx]

            mod_i = self.model_class(endog[inidx], exog[inidx],
                                     offset=offset_,
                                     exposure=exposure_,
                                     freq_weights=freq_weights[inidx],
                                     var_weights=var_weights[inidx],
                                     **init_kwds)
            if is_binomial:
                mod_i.family.n = init_kwds['n_trials']
            res_i = mod_i.fit(start_params=self.results.params,
                              method='newton')
            params[outidx] = res_i.params.copy()
            scale[outidx] = res_i.scale
            det_cov_params[outidx] = get_det_cov_params(res_i)

        return dict(params=params, scale=scale, mse_resid=scale,
                    # alias for now
                    det_cov_params=det_cov_params)
