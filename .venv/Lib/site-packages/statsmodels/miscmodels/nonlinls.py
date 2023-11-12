'''Non-linear least squares



Author: Josef Perktold based on scipy.optimize.curve_fit

'''
import numpy as np
from scipy import optimize

from statsmodels.base.model import Model


class Results:
    '''just a dummy placeholder for now
    most results from RegressionResults can be used here
    '''
    pass


##def getjaccov(retval, n):
##    '''calculate something and raw covariance matrix from return of optimize.leastsq
##
##    I cannot figure out how to recover the Jacobian, or whether it is even
##    possible
##
##    this is a partial copy of scipy.optimize.leastsq
##    '''
##    info = retval[-1]
##    #n = len(x0)  #nparams, where do I get this
##    cov_x = None
##    if info in [1,2,3,4]:
##        from numpy.dual import inv
##        from numpy.linalg import LinAlgError
##        perm = np.take(np.eye(n), retval[1]['ipvt']-1,0)
##        r = np.triu(np.transpose(retval[1]['fjac'])[:n,:])
##        R = np.dot(r, perm)
##        try:
##            cov_x = inv(np.dot(np.transpose(R),R))
##        except LinAlgError:
##            print 'cov_x not available'
##            pass
##        return r, R, cov_x
##
##def _general_function(params, xdata, ydata, function):
##    return function(xdata, *params) - ydata
##
##def _weighted_general_function(params, xdata, ydata, function, weights):
##    return weights * (function(xdata, *params) - ydata)
##



class NonlinearLS(Model):  #or subclass a model
    r'''Base class for estimation of a non-linear model with least squares

    This class is supposed to be subclassed, and the subclass has to provide a method
    `_predict` that defines the non-linear function `f(params) that is predicting the endogenous
    variable. The model is assumed to be

    :math: y = f(params) + error

    and the estimator minimizes the sum of squares of the estimated error.

    :math: min_parmas \sum (y - f(params))**2

    f has to return the prediction for each observation. Exogenous or explanatory variables
    should be accessed as attributes of the class instance, and can be given as arguments
    when the instance is created.

    Warning:
    Weights are not correctly handled yet in the results statistics,
    but included when estimating the parameters.

    similar to scipy.optimize.curve_fit
    API difference: params are array_like not split up, need n_params information

    includes now weights similar to curve_fit
    no general sigma yet (OLS and WLS, but no GLS)

    This is currently holding on to intermediate results that are not necessary
    but useful for testing.

    Fit returns and instance of RegressionResult, in contrast to the linear
    model, results in this case are based on a local approximation, essentially
    y = f(X, params) is replaced by y = grad * params where grad is the Gradient
    or Jacobian with the shape (nobs, nparams). See for example Greene

    Examples
    --------

    class Myfunc(NonlinearLS):

        def _predict(self, params):
            x = self.exog
            a, b, c = params
            return a*np.exp(-b*x) + c

    Ff we have data (y, x), we can create an instance and fit it with

    mymod = Myfunc(y, x)
    myres = mymod.fit(nparams=3)

    and use the non-linear regression results, for example

    myres.params
    myres.bse
    myres.tvalues


    '''
    #NOTE: This needs to call super for data checking
    def __init__(self, endog=None, exog=None, weights=None, sigma=None,
            missing='none'):
        self.endog = endog
        self.exog = exog
        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.ndim < 2:
                self.sigma = sigma
                self.weights = 1./sigma
            else:
                raise ValueError('correlated errors are not handled yet')
        else:
            self.weights = None

    def predict(self, exog, params=None):
        #copied from GLS, Model has different signature
        return self._predict(params)


    def _predict(self, params):
        pass

    def start_value(self):
        return None

    def geterrors(self, params, weights=None):
        if weights is None:
            if self.weights is None:
                return self.endog - self._predict(params)
            else:
                weights = self.weights
        return weights * (self.endog - self._predict(params))

    def errorsumsquares(self, params):
        return (self.geterrors(params)**2).sum()


    def fit(self, start_value=None, nparams=None, **kw):
        #if hasattr(self, 'start_value'):
        #I added start_value even if it's empty, not sure about it
        #but it makes a visible placeholder

        if start_value is not None:
            p0 = start_value
        else:
            #nesting so that start_value is only calculated if it is needed
            p0 = self.start_value()
            if p0 is not None:
                pass
            elif nparams is not None:
                p0 = 0.1 * np.ones(nparams)
            else:
                raise ValueError('need information about start values for' +
                             'optimization')

        func = self.geterrors
        res = optimize.leastsq(func, p0, full_output=1, **kw)
        (popt, pcov, infodict, errmsg, ier) = res

        if ier not in [1,2,3,4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        err = infodict['fvec']

        ydata = self.endog
        if (len(ydata) > len(p0)) and pcov is not None:
            #this can use the returned errors instead of recalculating

            s_sq = (err**2).sum()/(len(ydata)-len(p0))
            pcov = pcov * s_sq
        else:
            pcov = None

        self.df_resid = len(ydata)-len(p0)
        self.df_model = len(p0)
        fitres = Results()
        fitres.params = popt
        fitres.pcov = pcov
        fitres.rawres = res
        self.wendog = self.endog  #add weights
        self.wexog = self.jac_predict(popt)
        pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(pinv_wexog,
                                         np.transpose(pinv_wexog))

        #TODO: check effect of `weights` on result statistics
        #I think they are correctly included in cov_params
        #maybe not anymore, I'm not using pcov of leastsq
        #direct calculation with jac_predict misses the weights

##        if not weights is None
##            fitres.wexogw = self.weights * self.jacpredict(popt)
        from statsmodels.regression import RegressionResults
        results = RegressionResults

        beta = popt
        lfit = RegressionResults(self, beta,
                       normalized_cov_params=self.normalized_cov_params)

        lfit.fitres = fitres   #mainly for testing
        self._results = lfit
        return lfit

    def fit_minimal(self, start_value, **kwargs):
        '''minimal fitting with no extra calculations'''
        func = self.geterrors
        res = optimize.leastsq(func, start_value, full_output=0, **kwargs)
        return res

    def fit_random(self, ntries=10, rvs_generator=None, nparams=None):
        '''fit with random starting values

        this could be replaced with a global fitter

        '''

        if nparams is None:
            nparams = self.nparams
        if rvs_generator is None:
            rvs = np.random.uniform(low=-10, high=10, size=(ntries, nparams))
        else:
            rvs = rvs_generator(size=(ntries, nparams))

        results = np.array([np.r_[self.fit_minimal(rv),  rv] for rv in rvs])
        #selct best results and check how many solutions are within 1e-6 of best
        #not sure what leastsq returns
        return results

    def jac_predict(self, params):
        '''jacobian of prediction function using complex step derivative

        This assumes that the predict function does not use complex variable
        but is designed to do so.

        '''
        from statsmodels.tools.numdiff import approx_fprime_cs

        jaccs_err = approx_fprime_cs(params, self._predict)
        return jaccs_err


class Myfunc(NonlinearLS):

    #predict model.Model has a different signature
##    def predict(self, params, exog=None):
##        if not exog is None:
##            x = exog
##        else:
##            x = self.exog
##        a, b, c = params
##        return a*np.exp(-b*x) + c

    def _predict(self, params):
        x = self.exog
        a, b, c = params
        return a*np.exp(-b*x) + c





if __name__ == '__main__':
    def func0(x, a, b, c):
        return a*np.exp(-b*x) + c

    def func(params, x):
        a, b, c = params
        return a*np.exp(-b*x) + c

    def error(params, x, y):
        return y - func(params, x)

    def error2(params, x, y):
        return (y - func(params, x))**2




    x = np.linspace(0,4,50)
    params = np.array([2.5, 1.3, 0.5])
    y0 = func(params, x)
    y = y0 + 0.2*np.random.normal(size=len(x))

    res = optimize.leastsq(error, params, args=(x, y), full_output=True)
##    r, R, c = getjaccov(res[1:], 3)

    mod = Myfunc(y, x)
    resmy = mod.fit(nparams=3)

    cf_params, cf_pcov = optimize.curve_fit(func0, x, y)
    cf_bse = np.sqrt(np.diag(cf_pcov))
    print(res[0])
    print(cf_params)
    print(resmy.params)
    print(cf_bse)
    print(resmy.bse)
