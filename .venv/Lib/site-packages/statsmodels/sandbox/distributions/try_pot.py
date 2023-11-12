# -*- coding: utf-8 -*-
"""
Created on Wed May 04 06:09:18 2011

@author: josef
"""
import numpy as np


def mean_residual_life(x, frac=None, alpha=0.05):
    '''empirical mean residual life or expected shortfall

    Parameters
    ----------
    x : 1-dimensional array_like
    frac : list[float], optional
        All entries must be between 0 and 1
    alpha : float, default 0.05
        FIXME: not actually used.

    TODO:
        check formula for std of mean
        does not include case for all observations
        last observations std is zero
        vectorize loop using cumsum
        frac does not work yet
    '''

    axis = 0  # searchsorted is 1d only
    x = np.asarray(x)
    nobs = x.shape[axis]
    xsorted = np.sort(x, axis=axis)
    if frac is None:
        xthreshold = xsorted
    else:
        xthreshold = xsorted[np.floor(nobs * frac).astype(int)]
    # use searchsorted instead of simple index in case of ties
    xlargerindex = np.searchsorted(xsorted, xthreshold, side='right')

    # TODO:replace loop with cumsum ?
    result = []
    for i in range(len(xthreshold)-1):
        k_ind = xlargerindex[i]
        rmean = x[k_ind:].mean()
        # this does not work for last observations, nans
        rstd = x[k_ind:].std()
        rmstd = rstd/np.sqrt(nobs-k_ind)  # std error of mean, check formula
        result.append((k_ind, xthreshold[i], rmean, rmstd))

    res = np.array(result)
    crit = 1.96  # TODO: without loading stats, crit = -stats.t.ppf(0.05)
    confint = res[:, 1:2] + crit * res[:, -1:] * np.array([[-1, 1]])
    return np.column_stack((res, confint))


expected_shortfall = mean_residual_life  # alias


if __name__ == "__main__":
    rvs = np.random.standard_t(5, size=10)
    res = mean_residual_life(rvs)
    print(res)
    rmean = [rvs[i:].mean() for i in range(len(rvs))]
    print(res[:, 2] - rmean[1:])

    res_frac = mean_residual_life(rvs, frac=[0.5])
    print(res_frac)
