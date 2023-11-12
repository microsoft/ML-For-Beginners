'''Additional functions

prediction standard errors and confidence intervals


A: josef pktd
'''

import numpy as np
from scipy import stats

def atleast_2dcol(x):
    ''' convert array_like to 2d from 1d or 0d

    not tested because not used
    '''
    x = np.asarray(x)
    if (x.ndim == 1):
        x = x[:, None]
    elif (x.ndim == 0):
        x = np.atleast_2d(x)
    elif (x.ndim > 0):
        raise ValueError('too many dimensions')
    return x


def wls_prediction_std(res, exog=None, weights=None, alpha=0.05):
    '''calculate standard deviation and confidence interval for prediction

    applies to WLS and OLS, not to general GLS,
    that is independently but not identically distributed observations

    Parameters
    ----------
    res : regression result instance
        results of WLS or OLS regression required attributes see notes
    exog : array_like (optional)
        exogenous variables for points to predict
    weights : scalar or array_like (optional)
        weights as defined for WLS (inverse of variance of observation)
    alpha : float (default: alpha = 0.05)
        confidence level for two-sided hypothesis

    Returns
    -------
    predstd : array_like, 1d
        standard error of prediction
        same length as rows of exog
    interval_l, interval_u : array_like
        lower und upper confidence bounds

    Notes
    -----
    The result instance needs to have at least the following
    res.model.predict() : predicted values or
    res.fittedvalues : values used in estimation
    res.cov_params() : covariance matrix of parameter estimates

    If exog is 1d, then it is interpreted as one observation,
    i.e. a row vector.

    testing status: not compared with other packages

    References
    ----------

    Greene p.111 for OLS, extended to WLS by analogy

    '''
    # work around current bug:
    #    fit does not attach results to model, predict broken
    #res.model.results

    covb = res.cov_params()
    if exog is None:
        exog = res.model.exog
        predicted = res.fittedvalues
        if weights is None:
            weights = res.model.weights
    else:
        exog = np.atleast_2d(exog)
        if covb.shape[1] != exog.shape[1]:
            raise ValueError('wrong shape of exog')
        predicted = res.model.predict(res.params, exog)
        if weights is None:
            weights = 1.
        else:
            weights = np.asarray(weights)
            if weights.size > 1 and len(weights) != exog.shape[0]:
                raise ValueError('weights and exog do not have matching shape')


    # full covariance:
    #predvar = res3.mse_resid + np.diag(np.dot(X2,np.dot(covb,X2.T)))
    # predication variance only
    predvar = res.mse_resid/weights + (exog * np.dot(covb, exog.T).T).sum(1)
    predstd = np.sqrt(predvar)
    tppf = stats.t.isf(alpha/2., res.df_resid)
    interval_u = predicted + tppf * predstd
    interval_l = predicted - tppf * predstd
    return predstd, interval_l, interval_u
