# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
import pandas as pd


# this is similar to ContrastResults after t_test, copied and adjusted
class PredictionResults:
    """
    Results class for predictions.

    Parameters
    ----------
    predicted_mean : ndarray
        The array containing the prediction means.
    var_pred_mean : ndarray
        The array of the variance of the prediction means.
    var_resid : ndarray
        The array of residual variances.
    df : int
        The degree of freedom used if dist is 't'.
    dist : {'norm', 't', object}
        Either a string for the normal or t distribution or another object
        that exposes a `ppf` method.
    row_labels : list[str]
        Row labels used in summary frame.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid,
                 df=None, dist=None, row_labels=None):
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
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

    @property
    def se_obs(self):
        return np.sqrt(self.var_pred_mean + self.var_resid)

    @property
    def se_mean(self):
        return self.se

    @property
    def predicted_mean(self):
        # alias for backwards compatibility
        return self.predicted

    @property
    def var_pred_mean(self):
        # alias for backwards compatibility
        return self.var_pred

    @property
    def se(self):
        # alias for backwards compatibility
        return np.sqrt(self.var_pred_mean)

    def conf_int(self, obs=False, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of the
        constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """

        se = self.se_obs if obs else self.se_mean

        q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
        lower = self.predicted_mean - q * se
        upper = self.predicted_mean + q * se
        return np.column_stack((lower, upper))

    def summary_frame(self, alpha=0.05):
        # TODO: finish and cleanup
        ci_obs = self.conf_int(alpha=alpha, obs=True)  # need to split
        ci_mean = self.conf_int(alpha=alpha, obs=False)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]
        to_include['obs_ci_lower'] = ci_obs[:, 0]
        to_include['obs_ci_upper'] = ci_obs[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        # data = np.column_stack(list(to_include.values()))
        # names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def get_prediction(self, exog=None, transform=True, weights=None,
                   row_labels=None, pred_kwds=None):
    """
    Compute prediction results.

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
    weights : array_like, optional
        Weights interpreted as in WLS, used for the variance of the predicted
        residual.
    row_labels : list
        A list of row labels to use.  If not provided, read `exog` is
        available.
    **kwargs
        Some models can take additional keyword arguments, see the predict
        method of the model for the details.

    Returns
    -------
    linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """

    # prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            # GH-6509
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)

    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1:
            # Params informs whether a row or column vector
            if self.params.shape[0] > 1:
                exog = exog[None, :]
            else:
                exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog
        if weights is None:
            weights = getattr(self.model, 'weights', None)

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)

    # need to handle other arrays, TODO: is delegating to model possible ?
    if weights is not None:
        weights = np.asarray(weights)
        if (weights.size > 1 and
                (weights.ndim != 1 or weights.shape[0] == exog.shape[1])):
            raise ValueError('weights has wrong shape')

    if pred_kwds is None:
        pred_kwds = {}
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)

    covb = self.cov_params()
    var_pred_mean = (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale  # self.mse_resid / weights

    # TODO: check that we have correct scale, Refactor scale #???
    # special case for now:
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']

    if weights is not None:
        var_resid /= weights

    dist = ['norm', 't'][self.use_t]
    return PredictionResults(predicted_mean, var_pred_mean, var_resid,
                             df=self.df_resid, dist=dist,
                             row_labels=row_labels)
