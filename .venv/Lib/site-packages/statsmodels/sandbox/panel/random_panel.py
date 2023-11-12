# -*- coding: utf-8 -*-
"""Generate a random process with panel structure

Created on Sat Dec 17 22:15:27 2011

Author: Josef Perktold


Notes
-----
* written with unbalanced panels in mind, but not flexible enough yet
* need more shortcuts and options for balanced panel
* need to add random intercept or coefficients
* only one-way (repeated measures) so far

"""

import numpy as np
from . import correlation_structures as cs


class PanelSample:
    '''data generating process for panel with within correlation

    allows various within correlation structures, but no random intercept yet

    Parameters
    ----------
    nobs : int
        total number of observations
    k_vars : int
        number of explanatory variables to create in exog, including constant
    n_groups int
        number of groups in balanced sample
    exog : None or ndarray
        default is None, in which case a exog is created
    within : bool
        If True (default), then the exog vary within a group. If False, then
        only variation across groups is used.
        TODO: this option needs more work
    corr_structure : ndarray or ??
        Default is np.eye.
    corr_args : tuple
        arguments for the corr_structure
    scale : float
        scale of noise, standard deviation of normal distribution
    seed : None or int
        If seed is given, then this is used to create the random numbers for
        the sample.

    Notes
    -----
    The behavior for panel robust covariance estimators seems to differ by
    a large amount by whether exog have mostly within group or across group
    variation. I do not understand why this should be the case from the theory,
    and this would warrant more investigation.

    This is just used in one example so far and needs more usage to see what
    will be useful to add.

    '''

    def __init__(self, nobs, k_vars, n_groups, exog=None, within=True,
                 corr_structure=np.eye, corr_args=(), scale=1, seed=None):


        nobs_i = nobs//n_groups
        nobs = nobs_i * n_groups  #make balanced
        self.nobs = nobs
        self.nobs_i = nobs_i
        self.n_groups = n_groups
        self.k_vars = k_vars
        self.corr_structure = corr_structure
        self.groups = np.repeat(np.arange(n_groups), nobs_i)

        self.group_indices = np.arange(n_groups+1) * nobs_i #check +1

        if exog is None:
            if within:
                #t = np.tile(np.linspace(-1,1,nobs_i), n_groups)
                t = np.tile(np.linspace(0, 2, nobs_i), n_groups)
                #rs2 = np.random.RandomState(9876)
                #t = 1 + 0.3 * rs2.randn(nobs_i * n_groups)
                #mix within and across variation
                #t += np.repeat(np.linspace(-1,1,nobs_i), n_groups)
            else:
                #no within group variation,
                t = np.repeat(np.linspace(-1,1,nobs_i), n_groups)

            exog = t[:,None]**np.arange(k_vars)

        self.exog = exog
        #self.y_true = exog.sum(1)  #all coefficients equal 1,
        #moved to make random coefficients
        #initialize
        self.y_true = None
        self.beta = None

        if seed is None:
            seed = np.random.randint(0, 999999)

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

        #this makes overwriting difficult, move to method?
        self.std = scale * np.ones(nobs_i)
        corr = self.corr_structure(nobs_i, *corr_args)
        self.cov = cs.corr2cov(corr, self.std)
        self.group_means = np.zeros(n_groups)


    def get_y_true(self):
        if self.beta is None:
            self.y_true = self.exog.sum(1)
        else:
            self.y_true = np.dot(self.exog, self.beta)


    def generate_panel(self):
        '''
        generate endog for a random panel dataset with within correlation

        '''

        random = self.random_state

        if self.y_true is None:
            self.get_y_true()

        nobs_i = self.nobs_i
        n_groups = self.n_groups

        use_balanced = True
        if use_balanced: #much faster for balanced case
            noise = self.random_state.multivariate_normal(np.zeros(nobs_i),
                                                  self.cov,
                                                  size=n_groups).ravel()
            #need to add self.group_means
            noise += np.repeat(self.group_means, nobs_i)
        else:
            noise = np.empty(self.nobs, np.float64)
            noise.fill(np.nan)
            for ii in range(self.n_groups):
                #print ii,
                idx, idxupp = self.group_indices[ii:ii+2]
                #print idx, idxupp
                mean_i = self.group_means[ii]
                noise[idx:idxupp] = self.random_state.multivariate_normal(
                                        mean_i * np.ones(self.nobs_i), self.cov)

        endog = self.y_true + noise
        return endog
