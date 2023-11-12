# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
import pandas as pd


# this is similar to ContrastResults after t_test, partially copied, adjusted
class PredictionResultsBase:
    """Based class for get_prediction results
    """

    def __init__(self, predicted, var_pred, func=None, deriv=None,
                 df=None, dist=None, row_labels=None, **kwds):
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        '''z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        '''
        # assumes symmetric distribution
        stat = (self.predicted - value) / self.se

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, *, alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            Ignored in base class, only for compatibility, consistent signature
            with subclasses

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """

        ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                    dist_args=self.dist_args)
        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns 'predicted', 'se', 'ci_lower', 'ci_upper'
        """
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


class PredictionResultsMonotonic(PredictionResultsBase):

    def __init__(self, predicted, var_pred, linpred=None, linpred_se=None,
                 func=None, deriv=None, df=None, dist=None, row_labels=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted = predicted
        self.var_pred = var_pred
        self.linpred = linpred
        self.linpred_se = linpred_se
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()

        q = self.dist.ppf(1 - alpha / 2., *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        # if we want to stack at a new last axis, for lower.ndim > 1
        # np.concatenate((lower[..., None], upper[..., None]), axis=-1)
        return ci

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        # TODO: drop check?
        is_linear = (self.func(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self._conf_int_generic(self.linpred, self.linpred_se,
                                               alpha,
                                               dist_args=self.dist_args)
            ci = self.func(ci_linear)
        elif method == 'delta' or is_linear:
            ci = self._conf_int_generic(self.predicted, self.se, alpha,
                                        dist_args=self.dist_args)

        return ci


class PredictionResultsDelta(PredictionResultsBase):
    """Prediction results based on delta method
    """

    def __init__(self, results_delta, **kwds):

        predicted = results_delta.predicted()
        var_pred = results_delta.var()

        super().__init__(predicted, var_pred, **kwds)


class PredictionResultsMean(PredictionResultsBase):
    """Prediction results for GLM.

    This results class is used for backwards compatibility for
    `get_prediction` with GLM. The new PredictionResults classes dropped the
    `_mean` post fix in the attribute names.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid=None,
                 df=None, dist=None, row_labels=None, linpred=None, link=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link

        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def predicted_mean(self):
        # alias for backwards compatibility
        return self.predicted

    @property
    def var_pred_mean(self):
        # alias for backwards compatibility
        return self.var_pred

    @property
    def se_mean(self):
        # alias for backwards compatibility
        return self.se

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        This is currently only available for t and z tests.

        Parameters
        ----------
        method : {"endpoint", "delta"}
            Method for confidence interval, "m
            If method is "endpoint", then the confidence interval of the
            linear predictor is transformed by the prediction function.
            If method is "delta", then the delta-method is used. The confidence
            interval in this case might reach outside the range of the
            prediction, for example probabilities larger than one or smaller
            than zero.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        kwds : extra keyword arguments
            currently ignored, only for compatibility, consistent signature

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.link.inverse(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self.linpred.conf_int(alpha=alpha, obs=False)
            ci = self.link.inverse(ci_linear)
        elif method == 'delta' or is_linear:
            se = self.se_mean
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.predicted_mean - q * se
            upper = self.predicted_mean + q * se
            ci = np.column_stack((lower, upper))
            # if we want to stack at a new last axis, for lower.ndim > 1
            # np.concatenate((lower[..., None], upper[..., None]), axis=-1)

        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns
        'mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'.
        """
        # TODO: finish and cleanup
        ci_mean = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    """Prepare or transform exog for prediction

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    exog : ndarray
        Prediction exog
    row_labels : list of str
        Labels or pandas index for rows of prediction
    """

    # prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)

    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    return exog, row_labels


def get_prediction_glm(self, exog=None, transform=True,
                       row_labels=None, linpred=None, link=None,
                       pred_kwds=None):
    """
    Compute prediction results for GLM compatible models.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    linpred : linear prediction instance
        Instance of linear prediction results used for confidence intervals
        based on endpoint transformation.
    link : instance of link function
        If no link function is provided, then the `model.family.link` is used.
    pred_kwds : dict
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models. See the predict method of the
        model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if pred_kwds is None:
        pred_kwds = {}

    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)

    covb = self.cov_params()

    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv**2 * (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale  # self.mse_resid / weights

    # TODO: check that we have correct scale, Refactor scale #???
    # special case for now:
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']

    dist = ['norm', 't'][self.use_t]
    return PredictionResultsMean(
        predicted_mean, var_pred_mean, var_resid,
        df=self.df_resid, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)


def get_prediction_linear(self, exog=None, transform=True,
                          row_labels=None, pred_kwds=None, index=None):
    """
    Compute prediction results for linear prediction.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if pred_kwds is None:
        pred_kwds = {}

    k1 = exog.shape[1]
    if len(self.params > k1):
        # TODO: we allow endpoint transformation only for the first link
        index = np.arange(k1)
    else:
        index = None
    # get linear prediction and standard errors
    covb = self.cov_params(column=index)
    var_pred = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear["which"] = "linear"
    predicted = self.model.predict(self.params, exog, **pred_kwds_linear)

    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsBase(predicted, var_pred,
                                df=self.df_resid, dist=dist,
                                row_labels=row_labels
                                )
    return res


def get_prediction_monotonic(self, exog=None, transform=True,
                             row_labels=None, link=None,
                             pred_kwds=None, index=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    link : instance of link function
        If no link function is provided, then the ``mmodel.family.link` is
        used.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )

    if pred_kwds is None:
        pred_kwds = {}

    if link is None:
        link = self.model.family.link

    func_deriv = link.inverse_deriv

    # get linear prediction and standard errors
    covb = self.cov_params(column=index)
    linpred_var = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear["which"] = "linear"
    linpred = self.model.predict(self.params, exog, **pred_kwds_linear)

    predicted = self.model.predict(self.params, exog, **pred_kwds)
    link_deriv = func_deriv(linpred)
    var_pred = link_deriv**2 * linpred_var

    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsMonotonic(predicted, var_pred,
                                     df=self.df_resid, dist=dist,
                                     row_labels=row_labels, linpred=linpred,
                                     linpred_se=np.sqrt(linpred_var),
                                     func=link.inverse, deriv=func_deriv)
    return res


def get_prediction_delta(
        self,
        exog=None,
        which="mean",
        average=False,
        agg_weights=None,
        transform=True,
        row_labels=None,
        pred_kwds=None
        ):
    """
    compute prediction results

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    which : str
        The statistic that is prediction. Which statistics are available
        depends on the model.predict method.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then them mean over
        observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    exog, row_labels = _get_exog_predict(
        self,
        exog=exog,
        transform=transform,
        row_labels=row_labels,
        )
    if agg_weights is None:
        agg_weights = np.array(1.)

    def f_pred(p):
        """Prediction function as function of params
        """
        pred = self.model.predict(p, exog, which=which, **pred_kwds)
        if average:
            # using `.T` which should work if aggweights is 1-dim
            pred = (pred.T * agg_weights.T).mean(-1).T
        return pred

    nlpm = self._get_wald_nonlinear(f_pred)
    # TODO: currently returns NonlinearDeltaCov
    res = PredictionResultsDelta(nlpm)
    return res


def get_prediction(self, exog=None, transform=True, which="mean",
                   row_labels=None, average=False, agg_weights=None,
                   pred_kwds=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    which : str
        Which statistic is to be predicted. Default is "mean".
        The available statistics and options depend on the model.
        see the model.predict docstring
    linear : bool
        Linear has been replaced by the `which` keyword and will be
        deprecated.
        If linear is True, then `which` is ignored and the linear
        prediction is returned.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    average : bool
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then the average
        over observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Aggregation weights, only used if average is True.
        The weights are not normalized.
    **kwargs :
        Some models can take additional keyword arguments, such as offset,
        exposure or additional exog in multi-part models like zero inflated
        models.
        See the predict method of the model for the details.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and
        summary dataframe for the prediction.

    Notes
    -----
    Status: new in 0.14, experimental
    """
    use_endpoint = getattr(self.model, "_use_endpoint", True)

    if which == "linear":
        res = get_prediction_linear(
            self,
            exog=exog,
            transform=transform,
            row_labels=row_labels,
            pred_kwds=pred_kwds,
            )

    elif (which == "mean")and (use_endpoint is True) and (average is False):
        # endpoint transformation
        k1 = self.model.exog.shape[1]
        if len(self.params > k1):
            # TODO: we allow endpoint transformation only for the first link
            index = np.arange(k1)
        else:
            index = None

        pred_kwds["which"] = which
        # TODO: add link or ilink to all link based models (except zi
        link = getattr(self.model, "link", None)
        if link is None:
            # GLM
            if hasattr(self.model, "family"):
                link = getattr(self.model.family, "link", None)
        if link is None:
            # defaulting to log link for count models
            import warnings
            warnings.warn("using default log-link in get_prediction")
            from statsmodels.genmod.families import links
            link = links.Log()
        res = get_prediction_monotonic(
            self,
            exog=exog,
            transform=transform,
            row_labels=row_labels,
            link=link,
            pred_kwds=pred_kwds,
            index=index,
            )

    else:
        # which is not mean or linear, or we need averaging
        res = get_prediction_delta(
            self,
            exog=exog,
            which=which,
            average=average,
            agg_weights=agg_weights,
            pred_kwds=pred_kwds,
            )

    return res


def params_transform_univariate(params, cov_params, link=None, transform=None,
                                row_labels=None):
    """
    results for univariate, nonlinear, monotonicaly transformed parameters

    This provides transformed values, standard errors and confidence interval
    for transformations of parameters, for example in calculating rates with
    `exp(params)` in the case of Poisson or other models with exponential
    mean function.
    """

    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()

    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index

    params = np.asarray(params)

    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv**2 * np.diag(cov_params)
    # TODO: do we want covariance also, or just var/se

    dist = stats.norm

    # TODO: need ci for linear prediction, method of `lin_pred
    linpred = PredictionResultsMean(
        params, np.diag(cov_params), dist=dist,
        row_labels=row_labels, link=links.Identity())

    res = PredictionResultsMean(
        predicted_mean, var_pred_mean, dist=dist,
        row_labels=row_labels, linpred=linpred, link=link)

    return res
