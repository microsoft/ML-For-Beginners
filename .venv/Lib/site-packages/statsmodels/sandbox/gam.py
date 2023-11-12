"""
Generalized additive models



Requirements for smoothers
--------------------------

smooth(y, weights=xxx) : ? no return ? alias for fit
predict(x=None) : smoothed values, fittedvalues or for new exog
df_fit() : degress of freedom of fit ?


Notes
-----
- using PolySmoother works for AdditiveModel, and GAM with Poisson and Binomial
- testfailure with Gamma, no other families tested
- there is still an indeterminacy in the split up of the constant across
  components (smoothers) and alpha, sum, i.e. constant, looks good.
  - role of offset, that I have not tried to figure out yet

Refactoring
-----------
currently result is attached to model instead of other way around
split up Result in class for AdditiveModel and for GAM,
subclass GLMResults, needs verification that result statistics are appropriate
how much inheritance, double inheritance?
renamings and cleanup
interface to other smoothers, scipy splines

basic unittests as support for refactoring exist, but we should have a test
case for gamma and the others. Advantage of PolySmoother is that we can
benchmark against the parametric GLM results.

"""

# JP:
# changes: use PolySmoother instead of crashing bspline
# TODO: check/catalogue required interface of a smoother
# TODO: replace default smoother by corresponding function to initialize
#       other smoothers
# TODO: fix iteration, do not define class with iterator methods, use looping;
#       add maximum iteration and other optional stop criteria
# fixed some of the dimension problems in PolySmoother,
#       now graph for example looks good
# NOTE: example script is now in examples folder
#update: I did some of the above, see module docstring

import numpy as np

from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc

import warnings

DEBUG = False

def default_smoother(x, s_arg=None):
    '''

    '''
#    _x = x.copy()
#    _x.sort()
    _x = np.sort(x)
    n = x.shape[0]
    # taken form smooth.spline in R

    #if n < 50:
    if n < 500:
        nknots = n
    else:
        a1 = np.log(50) / np.log(2)
        a2 = np.log(100) / np.log(2)
        a3 = np.log(140) / np.log(2)
        a4 = np.log(200) / np.log(2)
        if n < 200:
            nknots = 2**(a1 + (a2 - a1) * (n - 50)/150.)
        elif n < 800:
            nknots = 2**(a2 + (a3 - a2) * (n - 200)/600.)
        elif n < 3200:
            nknots = 2**(a3 + (a4 - a3) * (n - 800)/2400.)
        else:
            nknots = 200 + (n - 3200.)**0.2
    knots = _x[np.linspace(0, n-1, nknots).astype(np.int32)]

    #s = SmoothingSpline(knots, x=x.copy())
    #when I set order=2, I get nans in the GAM prediction
    if s_arg is None:
        order = 3 #what about knots? need smoother *args or **kwds
    else:
        order = s_arg
    s = PolySmoother(order, x=x.copy())  #TODO: change order, why copy?
#    s.gram(d=2)
#    s.target_df = 5
    return s

class Offset:

    def __init__(self, fn, offset):
        self.fn = fn
        self.offset = offset

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw) + self.offset

class Results:

    def __init__(self, Y, alpha, exog, smoothers, family, offset):
        self.nobs, self.k_vars = exog.shape  #assumes exog is 2d
        #weird: If I put the previous line after the definition of self.mu,
        #    then the attributed do not get added
        self.Y = Y
        self.alpha = alpha
        self.smoothers = smoothers
        self.offset = offset
        self.family = family
        self.exog = exog
        self.offset = offset
        self.mu = self.linkinversepredict(exog)  #TODO: remove __call__



    def __call__(self, exog):
        '''expected value ? check new GLM, same as mu for given exog
        maybe remove this
        '''
        return self.linkinversepredict(exog)

    def linkinversepredict(self, exog):  #TODO what's the name in GLM
        '''expected value ? check new GLM, same as mu for given exog
        '''
        return self.family.link.inverse(self.predict(exog))

    def predict(self, exog):
        '''predict response, sum of smoothed components
        TODO: What's this in the case of GLM, corresponds to X*beta ?
        '''
        #note: sum is here over axis=0,
        #TODO: transpose in smoothed and sum over axis=1

        #BUG: there is some inconsistent orientation somewhere
        #temporary hack, will not work for 1d
        #print dir(self)
        #print 'self.nobs, self.k_vars', self.nobs, self.k_vars
        exog_smoothed = self.smoothed(exog)
        #print 'exog_smoothed.shape', exog_smoothed.shape
        if exog_smoothed.shape[0] == self.k_vars:
            import warnings
            warnings.warn("old orientation, colvars, will go away",
                          FutureWarning)
            return np.sum(self.smoothed(exog), axis=0) + self.alpha
        if exog_smoothed.shape[1] == self.k_vars:
            return np.sum(exog_smoothed, axis=1) + self.alpha
        else:
            raise ValueError('shape mismatch in predict')

    def smoothed(self, exog):
        '''get smoothed prediction for each component

        '''
        #bug: with exog in predict I get a shape error
        #print 'smoothed', exog.shape, self.smoothers[0].predict(exog).shape
        #there was a mistake exog did not have column index i
        return np.array([self.smoothers[i].predict(exog[:,i]) + self.offset[i]
        #should not be a mistake because exog[:,i] is attached to smoother, but
        #it is for different exog
        #return np.array([self.smoothers[i].predict() + self.offset[i]
                         for i in range(exog.shape[1])]).T

    def smoothed_demeaned(self, exog):
        components = self.smoothed(exog)
        means = components.mean(0)
        constant = means.sum() + self.alpha
        components_demeaned = components - means
        return components_demeaned, constant

class AdditiveModel:
    '''additive model with non-parametric, smoothed components

    Parameters
    ----------
    exog : ndarray
    smoothers : None or list of smoother instances
        smoother instances not yet checked
    weights : None or ndarray
    family : None or family instance
        I think only used because of shared results with GAM and subclassing.
        If None, then Gaussian is used.
    '''

    def __init__(self, exog, smoothers=None, weights=None, family=None):
        self.exog = exog
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.exog.shape[0])

        self.smoothers = smoothers or [default_smoother(exog[:,i]) for i in range(exog.shape[1])]

        #TODO: why do we set here df, refactoring temporary?
        for i in range(exog.shape[1]):
            self.smoothers[i].df = 10

        if family is None:
            self.family = families.Gaussian()
        else:
            self.family = family
        #self.family = families.Gaussian()

    def _iter__(self):
        '''initialize iteration ?, should be removed

        '''
        self.iter = 0
        self.dev = np.inf
        return self

    def next(self):
        '''internal calculation for one fit iteration

        BUG: I think this does not improve, what is supposed to improve
            offset does not seem to be used, neither an old alpha
            The smoothers keep coef/params from previous iteration
        '''
        _results = self.results
        Y = self.results.Y
        mu = _results.predict(self.exog)
        #TODO offset is never used ?
        offset = np.zeros(self.exog.shape[1], np.float64)
        alpha = (Y * self.weights).sum() / self.weights.sum()
        for i in range(self.exog.shape[1]):
            tmp = self.smoothers[i].predict()
            #TODO: check what smooth needs to do
            #smooth (alias for fit, fit given x to new y and attach
            #print 'next shape', (Y - alpha - mu + tmp).shape
            bad = np.isnan(Y - alpha - mu + tmp).any()
            if bad: #temporary assert while debugging
                print(Y, alpha, mu, tmp)
                raise ValueError("nan encountered")
            #self.smoothers[i].smooth(Y - alpha - mu + tmp,
            self.smoothers[i].smooth(Y - mu + tmp,
                                     weights=self.weights)
            tmp2 = self.smoothers[i].predict() #fittedvalues of previous smooth/fit
            self.results.offset[i] = -(tmp2*self.weights).sum() / self.weights.sum()
            #self.offset used in smoothed
            if DEBUG:
                print(self.smoothers[i].params)
            mu += tmp2 - tmp
        #change setting offset here: tests still pass, offset equal to constant
        #in component ??? what's the effect of offset
        offset = self.results.offset
        #print self.iter
        #self.iter += 1 #missing incrementing of iter counter NOT
        return Results(Y, alpha, self.exog, self.smoothers, self.family, offset)

    def cont(self):
        '''condition to continue iteration loop

        Parameters
        ----------
        tol

        Returns
        -------
        cont : bool
            If true, then iteration should be continued.

        '''
        self.iter += 1 #moved here to always count, not necessary
        if DEBUG:
            print(self.iter, self.results.Y.shape)
            print(self.results.predict(self.exog).shape, self.weights.shape)
        curdev = (((self.results.Y - self.results.predict(self.exog))**2) * self.weights).sum()

        if self.iter > self.maxiter: #kill it, no max iterationoption
            return False
        if np.fabs((self.dev - curdev) / curdev) < self.rtol:
            self.dev = curdev
            return False

        #self.iter += 1
        self.dev = curdev
        return True

    def df_resid(self):
        '''degrees of freedom of residuals, ddof is sum of all smoothers df
        '''
        return self.results.Y.shape[0] - np.array([self.smoothers[i].df_fit() for i in range(self.exog.shape[1])]).sum()

    def estimate_scale(self):
        '''estimate standard deviation of residuals
        '''
        #TODO: remove use of self.results.__call__
        return ((self.results.Y - self.results(self.exog))**2).sum() / self.df_resid()

    def fit(self, Y, rtol=1.0e-06, maxiter=30):
        '''fit the model to a given endogenous variable Y

        This needs to change for consistency with statsmodels

        '''
        self.rtol = rtol
        self.maxiter = maxiter
        #iter(self)  # what does this do? anything?
        self._iter__()
        mu = 0
        alpha = (Y * self.weights).sum() / self.weights.sum()

        offset = np.zeros(self.exog.shape[1], np.float64)

        for i in range(self.exog.shape[1]):
            self.smoothers[i].smooth(Y - alpha - mu,
                                     weights=self.weights)
            tmp = self.smoothers[i].predict()
            offset[i] = (tmp * self.weights).sum() / self.weights.sum()
            tmp -= tmp.sum()
            mu += tmp

        self.results = Results(Y, alpha, self.exog, self.smoothers, self.family, offset)

        while self.cont():
            self.results = self.next()

        if self.iter >= self.maxiter:
            warnings.warn(iteration_limit_doc, IterationLimitWarning)

        return self.results

class Model(GLM, AdditiveModel):
#class Model(AdditiveModel):
    #TODO: what does GLM do? Is it actually used ?
    #only used in __init__, dropping it does not change results
    #but where gets family attached now? - weird, it's Gaussian in this case now
    #also where is the link defined?
    #AdditiveModel overwrites family and sets it to Gaussian - corrected

    #I think both GLM and AdditiveModel subclassing is only used in __init__

    #niter = 2

#    def __init__(self, exog, smoothers=None, family=family.Gaussian()):
#        GLM.__init__(self, exog, family=family)
#        AdditiveModel.__init__(self, exog, smoothers=smoothers)
#        self.family = family
    def __init__(self, endog, exog, smoothers=None, family=families.Gaussian()):
        #self.family = family
        #TODO: inconsistent super __init__
        AdditiveModel.__init__(self, exog, smoothers=smoothers, family=family)
        GLM.__init__(self, endog, exog, family=family)
        assert self.family is family  #make sure we got the right family

    def next(self):
        _results = self.results
        Y = _results.Y
        if np.isnan(self.weights).all():
            print("nanweights1")

        _results.mu = self.family.link.inverse(_results.predict(self.exog))
        #eta = _results.predict(self.exog)
        #_results.mu = self.family.fitted(eta)
        weights = self.family.weights(_results.mu)
        if np.isnan(weights).all():
            self.weights = weights
            print("nanweights2")
        self.weights = weights
        if DEBUG:
            print('deriv isnan', np.isnan(self.family.link.deriv(_results.mu)).any())

        #Z = _results.predict(self.exog) + \
        Z = _results.predict(self.exog) + \
               self.family.link.deriv(_results.mu) * (Y - _results.mu) #- _results.alpha #?added alpha

        m = AdditiveModel(self.exog, smoothers=self.smoothers,
                          weights=self.weights, family=self.family)

        #TODO: I do not know what the next two lines do, Z, Y ? which is endog?
        #Y is original endog, Z is endog for the next step in the iterative solver

        _results = m.fit(Z)
        self.history.append([Z, _results.predict(self.exog)])
        _results.Y = Y
        _results.mu = self.family.link.inverse(_results.predict(self.exog))
        self.iter += 1
        self.results = _results

        return _results

    def estimate_scale(self, Y=None):
        """
        Return Pearson\'s X^2 estimate of scale.
        """

        if Y is None:
            Y = self.Y
        resid = Y - self.results.mu
        return (np.power(resid, 2) / self.family.variance(self.results.mu)).sum() \
                    / self.df_resid   #TODO check this
                    #/ AdditiveModel.df_resid(self)  #what is the class doing here?


    def fit(self, Y, rtol=1.0e-06, maxiter=30):

        self.rtol = rtol
        self.maxiter = maxiter

        self.Y = np.asarray(Y, np.float64)

        self.history = []

        #iter(self)
        self._iter__()

        #TODO code duplication with next?
        alpha = self.Y.mean()
        mu0 = self.family.starting_mu(Y)
        #Z = self.family.link(alpha) + self.family.link.deriv(alpha) * (Y - alpha)
        Z = self.family.link(alpha) + self.family.link.deriv(alpha) * (Y - mu0)
        m = AdditiveModel(self.exog, smoothers=self.smoothers, family=self.family)
        self.results = m.fit(Z)
        self.results.mu = self.family.link.inverse(self.results.predict(self.exog))
        self.results.Y = Y

        while self.cont():
            self.results = self.next()
            self.scale = self.results.scale = self.estimate_scale()

        if self.iter >= self.maxiter:
            import warnings
            warnings.warn(iteration_limit_doc, IterationLimitWarning)

        return self.results
