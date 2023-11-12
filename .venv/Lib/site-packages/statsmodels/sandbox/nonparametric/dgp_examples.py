# -*- coding: utf-8 -*-
"""Examples of non-linear functions for non-parametric regression

Created on Sat Jan 05 20:21:22 2013

Author: Josef Perktold
"""

import numpy as np

## Functions

def fg1(x):
    '''Fan and Gijbels example function 1

    '''
    return x + 2 * np.exp(-16 * x**2)

def fg1eu(x):
    '''Eubank similar to Fan and Gijbels example function 1

    '''
    return x + 0.5 * np.exp(-50 * (x - 0.5)**2)

def fg2(x):
    '''Fan and Gijbels example function 2

    '''
    return np.sin(2 * x) + 2 * np.exp(-16 * x**2)

def func1(x):
    '''made up example with sin, square

    '''
    return np.sin(x * 5) / x + 2. * x - 1. * x**2

## Classes with Data Generating Processes

doc = {'description':
'''Base Class for Univariate non-linear example

    Does not work on it's own.
    needs additional at least self.func
''',
'ref': ''}

class _UnivariateFunction:
    #Base Class for Univariate non-linear example.
    #Does not work on it's own. needs additionally at least self.func
    __doc__ = '''%(description)s

    Parameters
    ----------
    nobs : int
        number of observations to simulate
    x : None or 1d array
        If x is given then it is used for the exogenous variable instead of
        creating a random sample
    distr_x : None or distribution instance
        Only used if x is None. The rvs method is used to create a random
        sample of the exogenous (explanatory) variable.
    distr_noise : None or distribution instance
        The rvs method is used to create a random sample of the errors.

    Attributes
    ----------
    x : ndarray, 1-D
        exogenous or explanatory variable. x is sorted.
    y : ndarray, 1-D
        endogenous or response variable
    y_true : ndarray, 1-D
        expected values of endogenous or response variable, i.e. values of y
        without noise
    func : callable
        underlying function (defined by subclass)

    %(ref)s
    ''' #% doc

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):

        if x is None:
            if distr_x is None:
                x = np.random.normal(loc=0, scale=self.s_x, size=nobs)
            else:
                x = distr_x.rvs(size=nobs)
            x.sort()

        self.x = x

        if distr_noise is None:
            noise = np.random.normal(loc=0, scale=self.s_noise, size=nobs)
        else:
            noise = distr_noise.rvs(size=nobs)

        if hasattr(self, 'het_scale'):
            noise *= self.het_scale(self.x)

        #self.func = fg1
        self.y_true = y_true = self.func(x)
        self.y = y_true + noise


    def plot(self, scatter=True, ax=None):
        '''plot the mean function and optionally the scatter of the sample

        Parameters
        ----------
        scatter : bool
            If true, then add scatterpoints of sample to plot.
        ax : None or matplotlib axis instance
            If None, then a matplotlib.pyplot figure is created, otherwise
            the given axis, ax, is used.

        Returns
        -------
        Figure
            This is either the created figure instance or the one associated
            with ax if ax is given.

        '''
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        if scatter:
            ax.plot(self.x, self.y, 'o', alpha=0.5)

        xx = np.linspace(self.x.min(), self.x.max(), 100)
        ax.plot(xx, self.func(xx), lw=2, color='b', label='dgp mean')
        return ax.figure

doc = {'description':
'''Fan and Gijbels example function 1

linear trend plus a hump
''',
'ref':
'''
References
----------
Fan, Jianqing, and Irene Gijbels. 1992. "Variable Bandwidth and Local
Linear Regression Smoothers."
The Annals of Statistics 20 (4) (December): 2008-2036. doi:10.2307/2242378.

'''}

class UnivariateFanGijbels1(_UnivariateFunction):
    __doc__ = _UnivariateFunction.__doc__ % doc


    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.
        self.s_noise = 0.7
        self.func = fg1
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                             distr_x=distr_x,
                                             distr_noise=distr_noise)

doc['description'] =\
'''Fan and Gijbels example function 2

sin plus a hump
'''

class UnivariateFanGijbels2(_UnivariateFunction):
    __doc__ = _UnivariateFunction.__doc__ % doc

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.
        self.s_noise = 0.5
        self.func = fg2
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                             distr_x=distr_x,
                                             distr_noise=distr_noise)

class UnivariateFanGijbels1EU(_UnivariateFunction):
    '''

    Eubank p.179f
    '''

    def __init__(self, nobs=50, x=None, distr_x=None, distr_noise=None):
        if distr_x is None:
            from scipy import stats
            distr_x = stats.uniform
        self.s_noise = 0.15
        self.func = fg1eu
        super(self.__class__, self).__init__(nobs=nobs, x=x,
                                             distr_x=distr_x,
                                             distr_noise=distr_noise)

class UnivariateFunc1(_UnivariateFunction):
    '''

    made up, with sin and quadratic trend
    '''

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        if x is None and distr_x is None:
            from scipy import stats
            distr_x = stats.uniform(-2, 4)
        else:
            nobs = x.shape[0]
        self.s_noise = 2.
        self.func = func1
        super(UnivariateFunc1, self).__init__(nobs=nobs, x=x,
                                             distr_x=distr_x,
                                             distr_noise=distr_noise)

    def het_scale(self, x):
        return np.sqrt(np.abs(3+x))
