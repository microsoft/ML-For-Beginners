"""
Mixed effects models

Author: Jonathan Taylor
Author: Josef Perktold
License: BSD-3


Notes
-----

It's pretty slow if the model is misspecified, in my first example convergence
in loglike is not reached within 2000 iterations. Added stop criteria based
on convergence of parameters instead.

With correctly specified model, convergence is fast, in 6 iterations in
example.

"""
import numpy as np
import numpy.linalg as L

from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly

class Unit:
    """
    Individual experimental unit for
    EM implementation of (repeated measures)
    mixed effects model.

    \'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm\'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    endog : ndarray, (nobs,)
        response, endogenous variable
    exog_fe : ndarray, (nobs, k_vars_fe)
        explanatory variables as regressors or fixed effects,
        should include exog_re to correct mean of random
        coefficients, see Notes
    exog_re : ndarray, (nobs, k_vars_re)
        explanatory variables or random effects or coefficients

    Notes
    -----
    If the exog_re variables are not included in exog_fe, then the
    mean of the random constants or coefficients are not centered.
    The covariance matrix of the random parameter estimates are not
    centered in this case. (That's how it looks to me. JP)
    """


    def __init__(self, endog, exog_fe, exog_re):

        self.Y = endog
        self.X = exog_fe
        self.Z = exog_re
        self.n = endog.shape[0]

    def _compute_S(self, D, sigma):
        """covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.3) from Laird, Lange, Stram (see help(Unit))
        """
        self.S = (np.identity(self.n) * sigma**2 +
                  np.dot(self.Z, np.dot(D, self.Z.T)))

    def _compute_W(self):
        """inverse covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.2) from Laird, Lange, Stram (see help(Unit))
        """
        self.W = L.inv(self.S)

    def compute_P(self, Sinv):
        """projection matrix (nobs_i, nobs_i) (M in regression ?)  (JP check, guessing)
        Display (3.10) from Laird, Lange, Stram (see help(Unit))

        W - W X Sinv X' W'
        """
        t = np.dot(self.W, self.X)
        self.P = self.W - np.dot(np.dot(t, Sinv), t.T)

    def _compute_r(self, alpha):
        """residual after removing fixed effects

        Display (3.5) from Laird, Lange, Stram (see help(Unit))
        """
        self.r = self.Y - np.dot(self.X, alpha)

    def _compute_b(self, D):
        """coefficients for random effects/coefficients
        Display (3.4) from Laird, Lange, Stram (see help(Unit))

        D Z' W r
        """
        self.b = np.dot(D, np.dot(np.dot(self.Z.T, self.W), self.r))

    def fit(self, a, D, sigma):
        """
        Compute unit specific parameters in
        Laird, Lange, Stram (see help(Unit)).

        Displays (3.2)-(3.5).
        """

        self._compute_S(D, sigma)    #random effect plus error covariance
        self._compute_W()            #inv(S)
        self._compute_r(a)           #residual after removing fixed effects/exogs
        self._compute_b(D)           #?  coefficients on random exog, Z ?

    def compute_xtwy(self):
        """
        Utility function to compute X^tWY (transposed ?) for Unit instance.
        """
        return np.dot(np.dot(self.W, self.Y), self.X) #is this transposed ?

    def compute_xtwx(self):
        """
        Utility function to compute X^tWX for Unit instance.
        """
        return np.dot(np.dot(self.X.T, self.W), self.X)

    def cov_random(self, D, Sinv=None):
        """
        Approximate covariance of estimates of random effects. Just after
        Display (3.10) in Laird, Lange, Stram (see help(Unit)).

        D - D' Z' P Z D

        Notes
        -----
        In example where the mean of the random coefficient is not zero, this
        is not a covariance but a non-centered moment. (proof by example)
        """
        if Sinv is not None:
            self.compute_P(Sinv)
        t = np.dot(self.Z, D)
        return D - np.dot(np.dot(t.T, self.P), t)

    def logL(self, a, ML=False):
        """
        Individual contributions to the log-likelihood, tries to return REML
        contribution by default though this requires estimated
        fixed effect a to be passed as an argument.

        no constant with pi included

        a is not used if ML=true  (should be a=None in signature)
        If ML is false, then the residuals are calculated for the given fixed
        effects parameters a.
        """

        if ML:
            return (np.log(L.det(self.W)) - (self.r * np.dot(self.W, self.r)).sum()) / 2.
        else:
            if a is None:
                raise ValueError('need fixed effect a for REML contribution to log-likelihood')
            r = self.Y - np.dot(self.X, a)
            return (np.log(L.det(self.W)) - (r * np.dot(self.W, r)).sum()) / 2.

    def deviance(self, ML=False):
        '''deviance defined as 2 times the negative loglikelihood

        '''
        return - 2 * self.logL(ML=ML)


class OneWayMixed:
    """
    Model for
    EM implementation of (repeated measures)
    mixed effects model.

    \'Maximum Likelihood Computations with Repeated Measures:
    Application of the EM Algorithm\'

    Nan Laird; Nicholas Lange; Daniel Stram

    Journal of the American Statistical Association,
    Vol. 82, No. 397. (Mar., 1987), pp. 97-105.


    Parameters
    ----------
    units : list of units
       the data for the individual units should be attached to the units
    response, fixed and random : formula expression, called as argument to Formula


    *available results and alias*

    (subject to renaming, and coversion to cached attributes)

    params() -> self.a : coefficient for fixed effects or exog
    cov_params() -> self.Sinv : covariance estimate of fixed effects/exog
    bse() : standard deviation of params

    cov_random -> self.D : estimate of random effects covariance
    params_random_units -> [self.units[...].b] : random coefficient for each unit


    *attributes*

    (others)

    self.m : number of units
    self.p : k_vars_fixed
    self.q : k_vars_random
    self.N : nobs (total)


    Notes
    -----
    Fit returns a result instance, but not all results that use the inherited
    methods have been checked.

    Parameters need to change: drop formula and we require a naming convention for
    the units (currently Y,X,Z). - endog, exog_fe, endog_re ?

    logL does not include constant, e.g. sqrt(pi)
    llf is for MLE not for REML


    convergence criteria for iteration
    Currently convergence in the iterative solver is reached if either the loglikelihood
    *or* the fixed effects parameter do not change above tolerance.

    In some examples, the fixed effects parameters converged to 1e-5 within 150 iterations
    while the log likelihood did not converge within 2000 iterations. This might be
    the case if the fixed effects parameters are well estimated, but there are still
    changes in the random effects. If params_rtol and params_atol are set at a higher
    level, then the random effects might not be estimated to a very high precision.

    The above was with a misspecified model, without a constant. With a
    correctly specified model convergence is fast, within a few iterations
    (6 in example).
    """

    def __init__(self, units):
        self.units = units
        self.m = len(self.units)
        self.n_units = self.m

        self.N = sum(unit.X.shape[0] for unit in self.units)
        self.nobs = self.N     #alias for now

        # Determine size of fixed effects
        d = self.units[0].X
        self.p = d.shape[1]  # d.shape = p
        self.k_exog_fe = self.p   #alias for now
        self.a = np.zeros(self.p, np.float64)

        # Determine size of D, and sensible initial estimates
        # of sigma and D
        d = self.units[0].Z
        self.q = d.shape[1]  # Z.shape = q
        self.k_exog_re = self.q   #alias for now
        self.D = np.zeros((self.q,)*2, np.float64)
        self.sigma = 1.

        self.dev = np.inf   #initialize for iterations, move it?

    def _compute_a(self):
        """fixed effects parameters

        Display (3.1) of
        Laird, Lange, Stram (see help(Mixed)).
        """

        for unit in self.units:
            unit.fit(self.a, self.D, self.sigma)

        S = sum([unit.compute_xtwx() for unit in self.units])
        Y = sum([unit.compute_xtwy() for unit in self.units])

        self.Sinv = L.pinv(S)
        self.a = np.dot(self.Sinv, Y)

    def _compute_sigma(self, ML=False):
        """
        Estimate sigma. If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.6) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.8).

        sigma is the standard deviation of the noise (residual)
        """
        sigmasq = 0.
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            t = unit.r - np.dot(unit.Z, unit.b)
            sigmasq += np.power(t, 2).sum()
            sigmasq += self.sigma**2 * np.trace(np.identity(unit.n) -
                                               self.sigma**2 * W)
        self.sigma = np.sqrt(sigmasq / self.N)

    def _compute_D(self, ML=False):
        """
        Estimate random effects covariance D.
        If ML is True, return the ML estimate of sigma,
        else return the REML estimate.

        If ML, this is (3.7) in Laird, Lange, Stram (see help(Mixed)),
        otherwise it corresponds to (3.9).
        """
        D = 0.
        for unit in self.units:
            if ML:
                W = unit.W
            else:
                unit.compute_P(self.Sinv)
                W = unit.P
            D += np.multiply.outer(unit.b, unit.b)
            t = np.dot(unit.Z, self.D)
            D += self.D - np.dot(np.dot(t.T, W), t)

        self.D = D / self.m

    def cov_fixed(self):
        """
        Approximate covariance of estimates of fixed effects.

        Just after Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
        return self.Sinv

    #----------- alias (JP)   move to results class ?

    def cov_random(self):
        """
        Estimate random effects covariance D.

        If ML is True, return the ML estimate of sigma, else return the REML estimate.

        see _compute_D, alias for self.D
        """
        return self.D

    @property
    def params(self):
        '''
        estimated coefficients for exogeneous variables or fixed effects

        see _compute_a, alias for self.a
        '''
        return self.a

    @property
    def params_random_units(self):
        '''random coefficients for each unit

        '''
        return np.array([unit.b for unit in self.units])

    def cov_params(self):
        '''
        estimated covariance for coefficients for exogeneous variables or fixed effects

        see cov_fixed, and Sinv in _compute_a
        '''
        return self.cov_fixed()


    @property
    def bse(self):
        '''
        standard errors of estimated coefficients for exogeneous variables (fixed)

        '''
        return np.sqrt(np.diag(self.cov_params()))

    #----------- end alias

    def deviance(self, ML=False):
        '''deviance defined as 2 times the negative loglikelihood

        '''
        return -2 * self.logL(ML=ML)


    def logL(self, ML=False):
        """
        Return log-likelihood, REML by default.
        """
        #I do not know what the difference between REML and ML is here.
        logL = 0.

        for unit in self.units:
            logL += unit.logL(a=self.a, ML=ML)
        if not ML:
            logL += np.log(L.det(self.Sinv)) / 2
        return logL

    def initialize(self):
        S = sum([np.dot(unit.X.T, unit.X) for unit in self.units])
        Y = sum([np.dot(unit.X.T, unit.Y) for unit in self.units])
        self.a = L.lstsq(S, Y, rcond=-1)[0]

        D = 0
        t = 0
        sigmasq = 0
        for unit in self.units:
            unit.r = unit.Y - np.dot(unit.X, self.a)
            if self.q > 1:
                unit.b = L.lstsq(unit.Z, unit.r, rcond=-1)[0]
            else:
                Z = unit.Z.reshape((unit.Z.shape[0], 1))
                unit.b = L.lstsq(Z, unit.r, rcond=-1)[0]

            sigmasq += (np.power(unit.Y, 2).sum() -
                        (self.a * np.dot(unit.X.T, unit.Y)).sum() -
                        (unit.b * np.dot(unit.Z.T, unit.r)).sum())
            D += np.multiply.outer(unit.b, unit.b)
            t += L.pinv(np.dot(unit.Z.T, unit.Z))

        #TODO: JP added df_resid check
        self.df_resid = (self.N - (self.m - 1) * self.q - self.p)
        sigmasq /= (self.N - (self.m - 1) * self.q - self.p)
        self.sigma = np.sqrt(sigmasq)
        self.D = (D - sigmasq * t) / self.m

    def cont(self, ML=False, rtol=1.0e-05, params_rtol=1e-5, params_atol=1e-4):
        '''convergence check for iterative estimation

        '''

        self.dev, old = self.deviance(ML=ML), self.dev

        #self.history.append(np.hstack((self.dev, self.a)))
        self.history['llf'].append(self.dev)
        self.history['params'].append(self.a.copy())
        self.history['D'].append(self.D.copy())

        if np.fabs((self.dev - old) / self.dev) < rtol:   #why is there times `*`?
            #print np.fabs((self.dev - old)), self.dev, old
            self.termination = 'llf'
            return False

        #break if parameters converged
        #TODO: check termination conditions, OR or AND
        if np.all(np.abs(self.a - self._a_old) < (params_rtol * self.a + params_atol)):
            self.termination = 'params'
            return False

        self._a_old =  self.a.copy()
        return True

    def fit(self, maxiter=100, ML=False, rtol=1.0e-05, params_rtol=1e-6, params_atol=1e-6):

        #initialize for convergence criteria
        self._a_old = np.inf * self.a
        self.history = {'llf':[], 'params':[], 'D':[]}

        for i in range(maxiter):
            self._compute_a()              #a, Sinv :  params, cov_params of fixed exog
            self._compute_sigma(ML=ML)     #sigma   MLE or REML of sigma ?
            self._compute_D(ML=ML)         #D :  covariance of random effects, MLE or REML
            if not self.cont(ML=ML, rtol=rtol, params_rtol=params_rtol,
                                             params_atol=params_atol):
                break
        else: #if end of loop is reached without break
            self.termination = 'maxiter'
            print('Warning: maximum number of iterations reached')

        self.iterations = i

        results = OneWayMixedResults(self)
        #compatibility functions for fixed effects/exog
        results.scale = 1
        results.normalized_cov_params = self.cov_params()

        return results


class OneWayMixedResults(LikelihoodModelResults):
    '''Results class for OneWayMixed models

    '''
    def __init__(self, model):
        #TODO: check, change initialization to more standard pattern
        self.model = model
        self.params = model.params


    #need to overwrite this because we do not have a standard
    #model.loglike yet
    #TODO: what todo about REML loglike, logL is not normalized
    @cache_readonly
    def llf(self):
        return self.model.logL(ML=True)

    @property
    def params_random_units(self):
        return self.model.params_random_units

    def cov_random(self):
        return self.model.cov_random()

    def mean_random(self, idx='lastexog'):
        if idx == 'lastexog':
            meanr = self.params[-self.model.k_exog_re:]
        elif isinstance(idx, list):
            if not len(idx) == self.model.k_exog_re:
                raise ValueError('length of idx different from k_exog_re')
            else:
                meanr = self.params[idx]
        else:
            meanr = np.zeros(self.model.k_exog_re)

        return meanr

    def std_random(self):
        return np.sqrt(np.diag(self.cov_random()))

    def plot_random_univariate(self, bins=None, use_loc=True):
        '''create plot of marginal distribution of random effects

        Parameters
        ----------
        bins : int or bin edges
            option for bins in matplotlibs hist method. Current default is not
            very sophisticated. All distributions use the same setting for
            bins.
        use_loc : bool
            If True, then the distribution with mean given by the fixed
            effect is used.

        Returns
        -------
        Figure
            figure with subplots

        Notes
        -----
        What can make this fancier?

        Bin edges will not make sense if loc or scale differ across random
        effect distributions.

        '''
        #outsource this
        import matplotlib.pyplot as plt
        from scipy.stats import norm as normal
        fig = plt.figure()
        k = self.model.k_exog_re
        if k > 3:
            rows, cols = int(np.ceil(k * 0.5)), 2
        else:
            rows, cols = k, 1
        if bins is None:
            #bins = self.model.n_units // 20    #TODO: just roughly, check
            #bins = np.sqrt(self.model.n_units)
            bins = 5 + 2 * self.model.n_units**(1./3.)

        if use_loc:
            loc = self.mean_random()
        else:
            loc = [0]*k

        scale = self.std_random()

        for ii in range(k):
            ax = fig.add_subplot(rows, cols, ii)

            freq, bins_, _ = ax.hist(loc[ii] + self.params_random_units[:,ii],
                                    bins=bins, normed=True)
            points = np.linspace(bins_[0], bins_[-1], 200)

            #ax.plot(points, normal.pdf(points, loc=loc, scale=scale))
            #loc of sample is approx. zero, with Z appended to X
            #alternative, add fixed  to mean
            ax.set_title('Random Effect %d Marginal Distribution' % ii)
            ax.plot(points,
                    normal.pdf(points, loc=loc[ii], scale=scale[ii]),
                    'r')

        return fig

    def plot_scatter_pairs(self, idx1, idx2, title=None, ax=None):
        '''create scatter plot of two random effects

        Parameters
        ----------
        idx1, idx2 : int
            indices of the two random effects to display, corresponding to
            columns of exog_re
        title : None or string
            If None, then a default title is added
        ax : None or matplotlib axis instance
            If None, then a figure with one axis is created and returned.
            If ax is not None, then the scatter plot is created on it, and
            this axis instance is returned.

        Returns
        -------
        ax_or_fig : axis or figure instance
            see ax parameter

        Notes
        -----
        Still needs ellipse from estimated parameters

        '''
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax_or_fig = fig

        re1 = self.params_random_units[:,idx1]
        re2 = self.params_random_units[:,idx2]
        ax.plot(re1, re2, 'o', alpha=0.75)
        if title is None:
            title = 'Random Effects %d and %d' % (idx1, idx2)
        ax.set_title(title)
        ax_or_fig = ax

        return ax_or_fig

    def plot_scatter_all_pairs(self, title=None):
        from statsmodels.graphics.plot_grids import scatter_ellipse
        if self.model.k_exog_re < 2:
            raise ValueError('less than two variables available')

        return scatter_ellipse(self.params_random_units,
                               #ell_kwds not implemented yet
                               ell_kwds={'color':'r'})

#        #note I have written this already as helper function, get it
#        import matplotlib.pyplot as plt
#        #from scipy.stats import norm as normal
#        fig = plt.figure()
#        k = self.model.k_exog_re
#        n_plots = k * (k - 1) // 2
#        if n_plots > 3:
#            rows, cols = int(np.ceil(n_plots * 0.5)), 2
#        else:
#            rows, cols = n_plots, 1
#
#        count = 1
#        for ii in range(k):
#            for jj in range(ii):
#                ax = fig.add_subplot(rows, cols, count)
#                self.plot_scatter_pairs(ii, jj, title=None, ax=ax)
#                count += 1
#
#        return fig
