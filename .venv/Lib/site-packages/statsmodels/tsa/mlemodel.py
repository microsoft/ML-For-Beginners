"""Base Classes for Likelihood Models in time series analysis

Warning: imports numdifftools



Created on Sun Oct 10 15:00:47 2010

Author: josef-pktd
License: BSD

"""


try:
    import numdifftools as ndt
except ImportError:
    pass

from statsmodels.base.model import LikelihoodModel


#copied from sandbox/regression/mle.py
#TODO: I take it this is only a stub and should be included in another
# model class?
class TSMLEModel(LikelihoodModel):
    """
    univariate time series model for estimation with maximum likelihood

    Note: This is not working yet
    """

    def __init__(self, endog, exog=None):
        #need to override p,q (nar,nma) correctly
        super().__init__(endog, exog)
        #set default arma(1,1)
        self.nar = 1
        self.nma = 1
        #self.initialize()

    def geterrors(self, params):
        raise NotImplementedError

    def loglike(self, params):
        """
        Loglikelihood for timeseries model

        Parameters
        ----------
        params : array_like
            The model parameters

        Notes
        -----
        needs to be overwritten by subclass
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector for Arma model
        """
        #return None
        #print params
        jac = ndt.Jacobian(self.loglike, stepMax=1e-4)
        return jac(params)[-1]

    def hessian(self, params):
        """
        Hessian of arma model.  Currently uses numdifftools
        """
        #return None
        Hfun = ndt.Jacobian(self.score, stepMax=1e-4)
        return Hfun(params)[-1]

    def fit(self, start_params=None, maxiter=5000, method='fmin', tol=1e-08):
        '''estimate model by minimizing negative loglikelihood

        does this need to be overwritten ?
        '''
        if start_params is None and hasattr(self, '_start_params'):
            start_params = self._start_params
        #start_params = np.concatenate((0.05*np.ones(self.nar + self.nma), [1]))
        mlefit = super().fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit
