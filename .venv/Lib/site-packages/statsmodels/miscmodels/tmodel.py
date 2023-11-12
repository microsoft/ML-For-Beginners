"""Linear Model with Student-t distributed errors

Because the t distribution has fatter tails than the normal distribution, it
can be used to model observations with heavier tails and observations that have
some outliers. For the latter case, the t-distribution provides more robust
estimators for mean or mean parameters (what about var?).



References
----------
Kenneth L. Lange, Roderick J. A. Little, Jeremy M. G. Taylor (1989)
Robust Statistical Modeling Using the t Distribution
Journal of the American Statistical Association
Vol. 84, No. 408 (Dec., 1989), pp. 881-896
Published by: American Statistical Association
Stable URL: http://www.jstor.org/stable/2290063

not read yet


Created on 2010-09-24
Author: josef-pktd
License: BSD

TODO
----
* add starting values based on OLS
* bugs: store_params does not seem to be defined, I think this was a module
        global for debugging - commented out
* parameter restriction: check whether version with some fixed parameters works


"""
#mostly copied from the examples directory written for trying out generic mle.

import numpy as np
from scipy import special, stats

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma


#redefine some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln


class TLinearModel(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''

    def initialize(self):
        print("running Tmodel initialize")
        # TODO: here or in __init__
        self.k_vars = self.exog.shape[1]
        if not hasattr(self, 'fix_df'):
            self.fix_df = False

        if self.fix_df is False:
            # df will be estimated, no parameter restrictions
            self.fixed_params = None
            self.fixed_paramsmask = None
            self.k_params = self.exog.shape[1] + 2
            extra_params_names = ['df', 'scale']
        else:
            # df fixed
            self.k_params = self.exog.shape[1] + 1
            fixdf = np.nan * np.zeros(self.exog.shape[1] + 2)
            fixdf[-2] = self.fix_df
            self.fixed_params = fixdf
            self.fixed_paramsmask = np.isnan(fixdf)
            extra_params_names = ['scale']

        super(TLinearModel, self).initialize()

        # Note: this needs to be after super initialize
        # super initialize sets default df_resid,
        #_set_extra_params_names adjusts it
        self._set_extra_params_names(extra_params_names)
        self._set_start_params()


    def _set_start_params(self, start_params=None, use_kurtosis=False):
        if start_params is not None:
            self.start_params = start_params
        else:
            from statsmodels.regression.linear_model import OLS
            res_ols = OLS(self.endog, self.exog).fit()
            start_params = 0.1*np.ones(self.k_params)
            start_params[:self.k_vars] = res_ols.params

            if self.fix_df is False:

                if use_kurtosis:
                    kurt = stats.kurtosis(res_ols.resid)
                    df = 6./kurt + 4
                else:
                    df = 5

                start_params[-2] = df
                #TODO adjust scale for df
                start_params[-1] = np.sqrt(res_ols.scale)

            self.start_params = start_params




    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        The t distribution is the standard t distribution and not a standardized
        t distribution, which means that the scale parameter is not equal to the
        standard deviation.

        self.fixed_params and self.expandparams can be used to fix some
        parameters. (I doubt this has been tested in this model.)
        """
        #print len(params),
        #store_params.append(params)
        if self.fixed_params is not None:
            #print 'using fixed'
            params = self.expandparams(params)

        beta = params[:-2]
        df = params[-2]
        scale = np.abs(params[-1])  #TODO check behavior around zero
        loc = np.dot(self.exog, beta)
        endog = self.endog
        x = (endog - loc)/scale
        #next part is stats.t._logpdf
        lPx = sps_gamln((df+1)/2) - sps_gamln(df/2.)
        lPx -= 0.5*np_log(df*np_pi) + (df+1)/2.*np_log(1+(x**2)/df)
        lPx -= np_log(scale)  # correction for scale
        return -lPx

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])


class TArma(Arma):
    '''Univariate Arma Model with t-distributed errors

    This inherit all methods except loglike from tsa.arma_mle.Arma

    This uses the standard t-distribution, the implied variance of
    the error is not equal to scale, but ::

        error_variance = df/(df-2)*scale**2

    Notes
    -----
    This might be replaced by a standardized t-distribution with scale**2
    equal to variance

    '''

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)


    #add for Jacobian calculation  bsejac in GenericMLE, copied from loglike
    def nloglikeobs(self, params):
        """
        Loglikelihood for arma model for each observation, t-distribute

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """

        errorsest = self.geterrors(params[:-2])
        #sigma2 = np.maximum(params[-1]**2, 1e-6)  #do I need this
        #axis = 0
        #nobs = len(errorsest)

        df = params[-2]
        scale = np.abs(params[-1])
        llike  = - stats.t._logpdf(errorsest/scale, df) + np_log(scale)
        return llike

    #TODO rename fit_mle -> fit, fit -> fit_ls
    def fit_mle(self, order, start_params=None, method='nm', maxiter=5000,
            tol=1e-08, **kwds):
        nar, nma = order
        if start_params is not None:
            if len(start_params) != nar + nma + 2:
                raise ValueError('start_param need sum(order) + 2 elements')
        else:
            start_params = np.concatenate((0.05*np.ones(nar + nma), [5, 1]))


        res = super(TArma, self).fit_mle(order=order,
                                         start_params=start_params,
                                         method=method, maxiter=maxiter,
                                         tol=tol, **kwds)

        return res
