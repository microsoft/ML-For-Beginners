# -*- coding: utf-8 -*-
"""Panel data analysis for short T and large N

Created on Sat Dec 17 19:32:00 2011

Author: Josef Perktold
License: BSD-3


starting from scratch before looking at references again
just a stub to get the basic structure for group handling
target outsource as much as possible for reuse

Notes
-----

this is the basic version using a loop over individuals which will be more
widely applicable. Depending on the special cases, there will be faster
implementations possible (sparse, kroneker, ...)

the only two group specific methods or get_within_cov and whiten

"""

import numpy as np
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.tools.grouputils import GroupSorted


def sum_outer_product_loop(x, group_iter):
    '''sum outerproduct dot(x_i, x_i.T) over individuals

    loop version

    '''

    mom = 0
    for g in group_iter():
        x_g = x[g]
        #print 'x_g.shape', x_g.shape
        mom += np.outer(x_g, x_g)

    return mom

def sum_outer_product_balanced(x, n_groups):
    '''sum outerproduct dot(x_i, x_i.T) over individuals

    where x_i is (nobs_i, 1), and result is (nobs_i, nobs_i)

    reshape-dot version, for x.ndim=1 only

    '''
    xrs = x.reshape(-1, n_groups, order='F')
    return np.dot(xrs, xrs.T)  #should be (nobs_i, nobs_i)

    #x.reshape(n_groups, nobs_i,  k_vars) #, order='F')
    #... ? this is getting 3-dimensional  dot, tensordot?
    #needs (n_groups, k_vars, k_vars) array with sum over groups
    #NOT
    #I only need this for x is 1d, i.e. residual


def whiten_individuals_loop(x, transform, group_iter):
    '''apply linear transform for each individual

    loop version
    '''

    #Note: figure out dimension of transformed variable
    #so we can pre-allocate
    x_new = []
    for g in group_iter():
        x_g = x[g]
        x_new.append(np.dot(transform, x_g))

    return np.concatenate(x_new) #np.vstack(x_new)  #or np.array(x_new) #check shape



class ShortPanelGLS2:
    '''Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    '''

    def __init__(self, endog, exog, group):
        self.endog = endog
        self.exog = exog
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups
        #self.nobs_group =   #list for unbalanced?

    def fit_ols(self):
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled  #return or not

    def get_within_cov(self, resid):
        #central moment or not?
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups   #df correction ?

    def whiten_groups(self, x, cholsigmainv_i):
        #from scipy import sparse #use sparse
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def fit(self):
        res_pooled = self.fit_ols() #get starting estimate
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        #print wendog.shape, wexog.shape
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1

class ShortPanelGLS(GLS):
    '''Short Panel with general intertemporal within correlation

    assumes data is stacked by individuals, panel is balanced and
    within correlation structure is identical across individuals.

    It looks like this can just inherit GLS and overwrite whiten
    '''

    def __init__(self, endog, exog, group, sigma_i=None):
        self.group = GroupSorted(group)
        self.n_groups = self.group.n_groups
        #self.nobs_group =   #list for unbalanced?
        nobs_i = len(endog) / self.n_groups #endog might later not be an ndarray
        #balanced only for now,
        #which is a requirement anyway in this case (full cov)
        #needs to change for parametrized sigma_i

        #
        if sigma_i is None:
            sigma_i = np.eye(int(nobs_i))
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T

        #super is taking care of endog, exog and sigma
        super(self.__class__, self).__init__(endog, exog, sigma=None)

    def get_within_cov(self, resid):
        #central moment or not?
        mom = sum_outer_product_loop(resid, self.group.group_iter)
        return mom / self.n_groups   #df correction ?

    def whiten_groups(self, x, cholsigmainv_i):
        #from scipy import sparse #use sparse
        wx = whiten_individuals_loop(x, cholsigmainv_i, self.group.group_iter)
        return wx

    def _fit_ols(self):
        #used as starting estimate in old explicity version
        self.res_pooled = OLS(self.endog, self.exog).fit()
        return self.res_pooled  #return or not

    def _fit_old(self):
        #old explicit version
        res_pooled = self._fit_ols() #get starting estimate
        sigma_i = self.get_within_cov(res_pooled.resid)
        self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T
        wendog = self.whiten_groups(self.endog, self.cholsigmainv_i)
        wexog = self.whiten_groups(self.exog, self.cholsigmainv_i)
        self.res1 = OLS(wendog, wexog).fit()
        return self.res1

    def whiten(self, x):
        #whiten x by groups, will be applied to endog and exog
        wx = whiten_individuals_loop(x, self.cholsigmainv_i, self.group.group_iter)
        return wx

    #copied from GLSHet and adjusted (boiler plate?)
    def fit_iterative(self, maxiter=3):
        """
        Perform an iterative two-step procedure to estimate the GLS model.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) once does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
        #Note: in contrast to GLSHet, we do not have an auxiliary regression here
        #      might be needed if there is more structure in cov_i

        #because we only have the loop we are not attaching the ols_pooled
        #initial estimate anymore compared to original version

        if maxiter < 1:
            raise ValueError('maxiter needs to be at least 1')

        import collections
        self.history = collections.defaultdict(list) #not really necessary

        for i in range(maxiter):
            #pinv_wexog is cached, delete it to force recalculation
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog

            #fit with current cov, GLS, i.e. OLS on whitened endog, exog
            results = self.fit()
            self.history['self_params'].append(results.params)

            if not i == maxiter-1:  #skip for last iteration, could break instead
                #print 'ols',
                self.results_old = results #store previous results for debugging

                #get cov from residuals of previous regression
                sigma_i = self.get_within_cov(results.resid)
                self.cholsigmainv_i = np.linalg.cholesky(np.linalg.pinv(sigma_i)).T

                #calculate new whitened endog and exog
                self.initialize()

        #note results is the wrapper, results._results is the results instance
        #results._results.results_residual_regression = res_resid
        return results
