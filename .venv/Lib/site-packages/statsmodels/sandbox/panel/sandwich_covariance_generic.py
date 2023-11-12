# -*- coding: utf-8 -*-
"""covariance with (nobs,nobs) loop and general kernel

This is a general implementation that is not efficient for any special cases.
kernel is currently only for one continuous variable and any number of
categorical groups.

No spatial example, continuous is interpreted as time

Created on Wed Nov 30 08:20:44 2011

Author: Josef Perktold
License: BSD-3

"""
import numpy as np

def kernel(d1, d2, r=None, weights=None):
    '''general product kernel

    hardcoded split for the example:
        cat1 is continuous (time), other categories are discrete

    weights is e.g. Bartlett for cat1
    r is (0,1) indicator vector for boolean weights 1{d1_i == d2_i}

    returns boolean if no continuous weights are used
    '''

    diff = d1 - d2
    if (weights is None) or (r[0] == 0):
        #time is irrelevant or treated as categorical
        return np.all((r * diff) == 0)   #return bool
    else:
        #time uses continuous kernel, all other categorical
        return weights[diff] * np.all((r[1:] * diff[1:]) == 0)


def aggregate_cov(x, d, r=None, weights=None):
    '''sum of outer procuct over groups and time selected by r

    This is for a generic reference implementation, it uses a nobs-nobs double
    loop.

    Parameters
    ----------
    x : ndarray, (nobs,) or (nobs, k_vars)
        data, for robust standard error calculation, this is array of x_i * u_i
    d : ndarray, (nobs, n_groups)
        integer group labels, each column contains group (or time) indices
    r : ndarray, (n_groups,)
        indicator for which groups to include. If r[i] is zero, then
        this group is ignored. If r[i] is not zero, then the cluster robust
        standard errors include this group.
    weights : ndarray
        weights if the first group dimension uses a HAC kernel

    Returns
    -------
    cov : ndarray (k_vars, k_vars) or scalar
        covariance matrix aggregates over group kernels
    count : int
        number of terms added in sum, mainly returned for cross-checking

    Notes
    -----
    This uses `kernel` to calculate the weighted distance between two
    observations.

    '''

    nobs = x.shape[0]   #either 1d or 2d with obs in rows
    #next is not needed yet
#    if x.ndim == 2:
#        kvars = x.shape[1]
#    else:
#        kvars = 1

    count = 0 #count non-zero pairs for cross checking, not needed
    res = 0 * np.outer(x[0], x[0])  #get output shape

    for ii in range(nobs):
        for jj in range(nobs):
            w = kernel(d[ii], d[jj], r=r, weights=weights)
            if w:  #true or non-zero
                res += w * np.outer(x[0], x[0])
                count *= 1

    return res, count

def weights_bartlett(nlags):
    #with lag zero, nlags is the highest lag included
    return 1 - np.arange(nlags+1)/(nlags+1.)

#------- examples, cases: hardcoded for d is time and two categorical groups
def S_all_hac(x, d, nlags=1):
    '''HAC independent of categorical group membership
    '''
    r = np.zeros(d.shape[1])
    r[0] = 1
    weights = weights_bartlett(nlags)
    return aggregate_cov(x, d, r=r, weights=weights)

def S_within_hac(x, d, nlags=1, groupidx=1):
    '''HAC for observations within a categorical group
    '''
    r = np.zeros(d.shape[1])
    r[0] = 1
    r[groupidx] = 1
    weights = weights_bartlett(nlags)
    return aggregate_cov(x, d, r=r, weights=weights)

def S_cluster(x, d, groupidx=[1]):
    r = np.zeros(d.shape[1])
    r[groupidx] = 1
    return aggregate_cov(x, d, r=r, weights=None)

def S_white(x, d):
    '''simple white heteroscedasticity robust covariance
    note: calculating this way is very inefficient, just for cross-checking
    '''
    r = np.ones(d.shape[1])  #only points on diagonal
    return aggregate_cov(x, d, r=r, weights=None)
