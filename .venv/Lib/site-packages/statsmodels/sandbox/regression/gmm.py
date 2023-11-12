'''Generalized Method of Moments, GMM, and Two-Stage Least Squares for
instrumental variables IV2SLS



Issues
------
* number of parameters, nparams, and starting values for parameters
  Where to put them? start was initially taken from global scope (bug)
* When optimal weighting matrix cannot be calculated numerically
  In DistQuantilesGMM, we only have one row of moment conditions, not a
  moment condition for each observation, calculation for cov of moments
  breaks down. iter=1 works (weights is identity matrix)
  -> need method to do one iteration with an identity matrix or an
     analytical weighting matrix given as parameter.
  -> add result statistics for this case, e.g. cov_params, I have it in the
     standalone function (and in calc_covparams which is a copy of it),
     but not tested yet.
  DONE `fitonce` in DistQuantilesGMM, params are the same as in direct call to fitgmm
      move it to GMM class (once it's clearer for which cases I need this.)
* GMM does not know anything about the underlying model, e.g. y = X beta + u or panel
  data model. It would be good if we can reuse methods from regressions, e.g.
  predict, fitted values, calculating the error term, and some result statistics.
  What's the best way to do this, multiple inheritance, outsourcing the functions,
  mixins or delegation (a model creates a GMM instance just for estimation).


Unclear
-------
* dof in Hausman
  - based on rank
  - differs between IV2SLS method and function used with GMM or (IV2SLS)
  - with GMM, covariance matrix difference has negative eigenvalues in iv example, ???
* jtest/jval
  - I'm not sure about the normalization (multiply or divide by nobs) in jtest.
    need a test case. Scaling of jval is irrelevant for estimation.
    jval in jtest looks to large in example, but I have no idea about the size
* bse for fitonce look too large (no time for checking now)
    formula for calc_cov_params for the case without optimal weighting matrix
    is wrong. I do not have an estimate for omega in that case. And I'm confusing
    between weights and omega, which are *not* the same in this case.



Author: josef-pktd
License: BSD (3-clause)

'''


from statsmodels.compat.python import lrange

import numpy as np
from scipy import optimize, stats

from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
                                    LikelihoodModel, LikelihoodModelResults)
from statsmodels.regression.linear_model import (OLS, RegressionResults,
                                                 RegressionResultsWrapper)
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d

DEBUG = 0


def maxabs(x):
    '''just a shortcut to np.abs(x).max()
    '''
    return np.abs(x).max()


class IV2SLS(LikelihoodModel):
    """
    Instrumental variables estimation using Two-Stage Least-Squares (2SLS)


    Parameters
    ----------
    endog : ndarray
       Endogenous variable, 1-dimensional or 2-dimensional array nobs by 1
    exog : ndarray
       Explanatory variables, 1-dimensional or 2-dimensional array nobs by k
    instrument : ndarray
       Instruments for explanatory variables. Must contain both exog
       variables that are not being instrumented and instruments

    Notes
    -----
    All variables in exog are instrumented in the calculations. If variables
    in exog are not supposed to be instrumented, then these variables
    must also to be included in the instrument array.

    Degrees of freedom in the calculation of the standard errors uses
    `df_resid = (nobs - k_vars)`.
    (This corresponds to the `small` option in Stata's ivreg2.)
    """

    def __init__(self, endog, exog, instrument=None):
        self.instrument, self.instrument_names = _ensure_2d(instrument, True)
        super(IV2SLS, self).__init__(endog, exog)
        # where is this supposed to be handled
        # Note: Greene p.77/78 dof correction is not necessary (because only
        #       asy results), but most packages do it anyway
        self.df_resid = self.exog.shape[0] - self.exog.shape[1]
        #self.df_model = float(self.rank - self.k_constant)
        self.df_model = float(self.exog.shape[1] - self.k_constant)

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog

    def whiten(self, X):
        """Not implemented"""
        pass

    def fit(self):
        '''estimate model using 2SLS IV regression

        Returns
        -------
        results : instance of RegressionResults
           regression result

        Notes
        -----
        This returns a generic RegressioResults instance as defined for the
        linear models.

        Parameter estimates and covariance are correct, but other results
        have not been tested yet, to see whether they apply without changes.

        '''
        #Greene 5th edt., p.78 section 5.4
        #move this maybe
        y,x,z = self.endog, self.exog, self.instrument
        # TODO: this uses "textbook" calculation, improve linalg
        ztz = np.dot(z.T, z)
        ztx = np.dot(z.T, x)
        self.xhatparams = xhatparams = np.linalg.solve(ztz, ztx)
        #print 'x.T.shape, xhatparams.shape', x.shape, xhatparams.shape
        F = xhat = np.dot(z, xhatparams)
        FtF = np.dot(F.T, F)
        self.xhatprod = FtF  #store for Housman specification test
        Ftx = np.dot(F.T, x)
        Fty = np.dot(F.T, y)
        params = np.linalg.solve(FtF, Fty)
        Ftxinv = np.linalg.inv(Ftx)
        self.normalized_cov_params = np.dot(Ftxinv.T, np.dot(FtF, Ftxinv))

        lfit = IVRegressionResults(self, params,
                       normalized_cov_params=self.normalized_cov_params)

        lfit.exog_hat_params = xhatparams
        lfit.exog_hat = xhat  # TODO: do we want to store this, might be large
        self._results_ols2nd = OLS(y, xhat).fit()

        return RegressionResultsWrapper(lfit)

    # copied from GLS, because I subclass currently LikelihoodModel and not GLS
    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array_like
            Design / exogenous data
        params : array_like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        if exog is None:
            exog = self.exog

        return np.dot(exog, params)


class IVRegressionResults(RegressionResults):
    """
    Results class for for an OLS model.

    Most of the methods and attributes are inherited from RegressionResults.
    The special methods that are only available for OLS are:

    - get_influence
    - outlier_test
    - el_test
    - conf_int_el

    See Also
    --------
    RegressionResults
    """

    @cache_readonly
    def fvalue(self):
        const_idx = self.model.data.const_idx
        # if constant is implicit or missing, return nan see #2444, #3544
        if const_idx is None:
            return np.nan
        else:
            k_vars = len(self.params)
            restriction = np.eye(k_vars)
            idx_noconstant = lrange(k_vars)
            del idx_noconstant[const_idx]
            fval = self.f_test(restriction[idx_noconstant]).fvalue # without constant
            return fval


    def spec_hausman(self, dof=None):
        '''Hausman's specification test

        See Also
        --------
        spec_hausman : generic function for Hausman's specification test

        '''
        #use normalized cov_params for OLS

        endog, exog = self.model.endog, self.model.exog
        resols = OLS(endog, exog).fit()
        normalized_cov_params_ols = resols.model.normalized_cov_params
        # Stata `ivendog` does not use df correction for se
        #se2 = resols.mse_resid #* resols.df_resid * 1. / len(endog)
        se2 = resols.ssr / len(endog)

        params_diff = self.params - resols.params

        cov_diff = np.linalg.pinv(self.model.xhatprod) - normalized_cov_params_ols
        #TODO: the following is very inefficient, solves problem (svd) twice
        #use linalg.lstsq or svd directly
        #cov_diff will very often be in-definite (singular)
        if not dof:
            dof = np.linalg.matrix_rank(cov_diff)
        cov_diffpinv = np.linalg.pinv(cov_diff)
        H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))/se2
        pval = stats.chi2.sf(H, dof)

        return H, pval, dof


# copied from regression results with small changes, no llf
    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Default is `var_##` for ## in p the number of regressors
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        #TODO: import where we need it (for now), add as cached attributes
        from statsmodels.stats.stattools import (jarque_bera,
                omni_normtest, durbin_watson)
        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        #TODO: reuse condno from somewhere else ?
        #condno = np.linalg.cond(np.dot(self.wexog.T, self.wexog))
        wexog = self.model.wexog
        eigvals = np.linalg.linalg.eigvalsh(np.dot(wexog.T, wexog))
        eigvals = np.sort(eigvals) #in increasing order
        condno = np.sqrt(eigvals[-1]/eigvals[0])

        # TODO: check what is valid.
        # box-pierce, breusch-pagan, durbin's h are not with endogenous on rhs
        # use Cumby Huizinga 1992 instead
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv, condno=condno,
                          mineigval=eigvals[0])

        #TODO not used yet
        #diagn_left_header = ['Models stats']
        #diagn_right_header = ['Residual stats']

        #TODO: requiring list/iterable is a bit annoying
        #need more control over formatting
        #TODO: default do not work if it's not identically spelled

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Two Stage']),
                    ('', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None), #[self.df_resid]), #TODO: spelling
                    ('Df Model:', None), #[self.df_model])
                    ]

        top_right = [('R-squared:', ["%#8.3f" % self.rsquared]),
                     ('Adj. R-squared:', ["%#8.3f" % self.rsquared_adj]),
                     ('F-statistic:', ["%#8.4g" % self.fvalue] ),
                     ('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     #('Log-Likelihood:', None), #["%#6.4g" % self.llf]),
                     #('AIC:', ["%#8.4g" % self.aic]),
                     #('BIC:', ["%#8.4g" % self.bic])
                     ]

        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])
                      ]

        diagn_right = [('Durbin-Watson:', ["%#8.3f" % durbin_watson(self.wresid)]),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % condno])
                       ]


        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        #create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                             use_t=True)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")



        return smry




############# classes for Generalized Method of Moments GMM

_gmm_options = '''\

Options for GMM
---------------

Type of GMM
~~~~~~~~~~~

 - one-step
 - iterated
 - CUE : not tested yet

weight matrix
~~~~~~~~~~~~~

 - `weights_method` : str, defines method for robust
   Options here are similar to :mod:`statsmodels.stats.robust_covariance`
   default is heteroscedasticity consistent, HC0

   currently available methods are

   - `cov` : HC0, optionally with degrees of freedom correction
   - `hac` :
   - `iid` : untested, only for Z*u case, IV cases with u as error indep of Z
   - `ac` : not available yet
   - `cluster` : not connected yet
   - others from robust_covariance

other arguments:

 - `wargs` : tuple or dict, required arguments for weights_method

   - `centered` : bool,
     indicates whether moments are centered for the calculation of the weights
     and covariance matrix, applies to all weight_methods
   - `ddof` : int
     degrees of freedom correction, applies currently only to `cov`
   - maxlag : int
     number of lags to include in HAC calculation , applies only to `hac`
   - others not yet, e.g. groups for cluster robust

covariance matrix
~~~~~~~~~~~~~~~~~

The same options as for weight matrix also apply to the calculation of the
estimate of the covariance matrix of the parameter estimates.
The additional option is

 - `has_optimal_weights`: If true, then the calculation of the covariance
   matrix assumes that we have optimal GMM with :math:`W = S^{-1}`.
   Default is True.
   TODO: do we want to have a different default after `onestep`?


'''

class GMM(Model):
    '''
    Class for estimation by Generalized Method of Moments

    needs to be subclassed, where the subclass defined the moment conditions
    `momcond`

    Parameters
    ----------
    endog : ndarray
        endogenous variable, see notes
    exog : ndarray
        array of exogenous variables, see notes
    instrument : ndarray
        array of instruments, see notes
    nmoms : None or int
        number of moment conditions, if None then it is set equal to the
        number of columns of instruments. Mainly needed to determine the shape
        or size of start parameters and starting weighting matrix.
    kwds : anything
        this is mainly if additional variables need to be stored for the
        calculations of the moment conditions

    Attributes
    ----------
    results : instance of GMMResults
        currently just a storage class for params and cov_params without it's
        own methods
    bse : property
        return bse



    Notes
    -----
    The GMM class only uses the moment conditions and does not use any data
    directly. endog, exog, instrument and kwds in the creation of the class
    instance are only used to store them for access in the moment conditions.
    Which of this are required and how they are used depends on the moment
    conditions of the subclass.

    Warning:

    Options for various methods have not been fully implemented and
    are still missing in several methods.


    TODO:
    currently onestep (maxiter=0) still produces an updated estimate of bse
    and cov_params.

    '''

    results_class = 'GMMResults'

    def __init__(self, endog, exog, instrument, k_moms=None, k_params=None,
                 missing='none', **kwds):
        '''
        maybe drop and use mixin instead

        TODO: GMM does not really care about the data, just the moment conditions
        '''
        instrument = self._check_inputs(instrument, endog) # attaches if needed
        super(GMM, self).__init__(endog, exog, missing=missing,
                instrument=instrument)
#         self.endog = endog
#         self.exog = exog
#         self.instrument = instrument
        self.nobs = endog.shape[0]
        if k_moms is not None:
            self.nmoms = k_moms
        elif instrument is not None:
            self.nmoms = instrument.shape[1]
        else:
            self.nmoms = np.nan

        if k_params is not None:
            self.k_params = k_params
        elif instrument is not None:
            self.k_params = exog.shape[1]
        else:
            self.k_params = np.nan

        self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def _check_inputs(self, instrument, endog):
        if instrument is not None:
            offset = np.asarray(instrument)
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("instrument is not the same length as endog")
        return instrument

    def _fix_param_names(self, params, param_names=None):
        # TODO: this is a temporary fix, need
        xnames = self.data.xnames

        if param_names is not None:
            if len(params) == len(param_names):
                self.data.xnames = param_names
            else:
                raise ValueError('param_names has the wrong length')

        else:
            if len(params) < len(xnames):
                # cut in front for poisson multiplicative
                self.data.xnames = xnames[-len(params):]
            elif len(params) > len(xnames):
                # use generic names
                self.data.xnames = ['p%2d' % i for i in range(len(params))]

    def set_param_names(self, param_names, k_params=None):
        """set the parameter names in the model

        Parameters
        ----------
        param_names : list[str]
            param_names should have the same length as the number of params
        k_params : None or int
            If k_params is None, then the k_params attribute is used, unless
            it is None.
            If k_params is not None, then it will also set the k_params
            attribute.
        """
        if k_params is not None:
            self.k_params = k_params
        else:
            k_params = self.k_params

        if k_params == len(param_names):
            self.data.xnames = param_names
        else:
            raise ValueError('param_names has the wrong length')


    def fit(self, start_params=None, maxiter=10, inv_weights=None,
                  weights_method='cov', wargs=(),
                  has_optimal_weights=True,
                  optim_method='bfgs', optim_args=None):
        '''
        Estimate parameters using GMM and return GMMResults

        TODO: weight and covariance arguments still need to be made consistent
        with similar options in other models,
        see RegressionResult.get_robustcov_results

        Parameters
        ----------
        start_params : array (optional)
            starting value for parameters ub minimization. If None then
            fitstart method is called for the starting values.
        maxiter : int or 'cue'
            Number of iterations in iterated GMM. The onestep estimate can be
            obtained with maxiter=0 or 1. If maxiter is large, then the
            iteration will stop either at maxiter or on convergence of the
            parameters (TODO: no options for convergence criteria yet.)
            If `maxiter == 'cue'`, the the continuously updated GMM is
            calculated which updates the weight matrix during the minimization
            of the GMM objective function. The CUE estimation uses the onestep
            parameters as starting values.
        inv_weights : None or ndarray
            inverse of the starting weighting matrix. If inv_weights are not
            given then the method `start_weights` is used which depends on
            the subclass, for IV subclasses `inv_weights = z'z` where `z` are
            the instruments, otherwise an identity matrix is used.
        weights_method : str, defines method for robust
            Options here are similar to :mod:`statsmodels.stats.robust_covariance`
            default is heteroscedasticity consistent, HC0

            currently available methods are

            - `cov` : HC0, optionally with degrees of freedom correction
            - `hac` :
            - `iid` : untested, only for Z*u case, IV cases with u as error indep of Z
            - `ac` : not available yet
            - `cluster` : not connected yet
            - others from robust_covariance

        wargs` : tuple or dict,
            required and optional arguments for weights_method

            - `centered` : bool,
              indicates whether moments are centered for the calculation of the weights
              and covariance matrix, applies to all weight_methods
            - `ddof` : int
              degrees of freedom correction, applies currently only to `cov`
            - `maxlag` : int
              number of lags to include in HAC calculation , applies only to `hac`
            - others not yet, e.g. groups for cluster robust

        has_optimal_weights: If true, then the calculation of the covariance
              matrix assumes that we have optimal GMM with :math:`W = S^{-1}`.
              Default is True.
              TODO: do we want to have a different default after `onestep`?
        optim_method : str, default is 'bfgs'
            numerical optimization method. Currently not all optimizers that
            are available in LikelihoodModels are connected.
        optim_args : dict
            keyword arguments for the numerical optimizer.

        Returns
        -------
        results : instance of GMMResults
            this is also attached as attribute results

        Notes
        -----

        Warning: One-step estimation, `maxiter` either 0 or 1, still has
        problems (at least compared to Stata's gmm).
        By default it uses a heteroscedasticity robust covariance matrix, but
        uses the assumption that the weight matrix is optimal.
        See options for cov_params in the results instance.

        The same options as for weight matrix also apply to the calculation of
        the estimate of the covariance matrix of the parameter estimates.

        '''
        # TODO: add check for correct wargs keys
        #       currently a misspelled key is not detected,
        #       because I'm still adding options

        # TODO: check repeated calls to fit with different options
        #       arguments are dictionaries, i.e. mutable
        #       unit test if anything  is stale or spilled over.

        #bug: where does start come from ???
        start = start_params  # alias for renaming
        if start is None:
            start = self.fitstart() #TODO: temporary hack

        if inv_weights is None:
            inv_weights

        if optim_args is None:
            optim_args = {}
        if 'disp' not in optim_args:
            optim_args['disp'] = 1

        if maxiter == 0 or maxiter == 'cue':
            if inv_weights is not None:
                weights = np.linalg.pinv(inv_weights)
            else:
                # let start_weights handle the inv=False for maxiter=0
                weights = self.start_weights(inv=False)

            params = self.fitgmm(start, weights=weights,
                                 optim_method=optim_method, optim_args=optim_args)
            weights_ = weights  # temporary alias used in jval
        else:
            params, weights = self.fititer(start,
                                           maxiter=maxiter,
                                           start_invweights=inv_weights,
                                           weights_method=weights_method,
                                           wargs=wargs,
                                           optim_method=optim_method,
                                           optim_args=optim_args)
            # TODO weights returned by fititer is inv_weights - not true anymore
            # weights_ currently not necessary and used anymore
            weights_ = np.linalg.pinv(weights)

        if maxiter == 'cue':
            #we have params from maxiter= 0 as starting value
            # TODO: need to give weights options to gmmobjective_cu
            params = self.fitgmm_cu(params,
                                     optim_method=optim_method,
                                     optim_args=optim_args)
            # weights is stored as attribute
            weights = self._weights_cu

        #TODO: use Bunch instead ?
        options_other = {'weights_method':weights_method,
                         'has_optimal_weights':has_optimal_weights,
                         'optim_method':optim_method}

        # check that we have the right number of xnames
        self._fix_param_names(params, param_names=None)
        results = results_class_dict[self.results_class](
                                        model = self,
                                        params = params,
                                        weights = weights,
                                        wargs = wargs,
                                        options_other = options_other,
                                        optim_args = optim_args)

        self.results = results # FIXME: remove, still keeping it temporarily
        return results

    def fitgmm(self, start, weights=None, optim_method='bfgs', optim_args=None):
        '''estimate parameters using GMM

        Parameters
        ----------
        start : array_like
            starting values for minimization
        weights : ndarray
            weighting matrix for moment conditions. If weights is None, then
            the identity matrix is used


        Returns
        -------
        paramest : ndarray
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        uses scipy.optimize.fmin

        '''
##        if not fixed is None:  #fixed not defined in this version
##            raise NotImplementedError

        # TODO: should start_weights only be in `fit`
        if weights is None:
            weights = self.start_weights(inv=False)

        if optim_args is None:
            optim_args = {}

        if optim_method == 'nm':
            optimizer = optimize.fmin
        elif optim_method == 'bfgs':
            optimizer = optimize.fmin_bfgs
            # TODO: add score
            optim_args['fprime'] = self.score #lambda params: self.score(params, weights)
        elif optim_method == 'ncg':
            optimizer = optimize.fmin_ncg
            optim_args['fprime'] = self.score
        elif optim_method == 'cg':
            optimizer = optimize.fmin_cg
            optim_args['fprime'] = self.score
        elif optim_method == 'fmin_l_bfgs_b':
            optimizer = optimize.fmin_l_bfgs_b
            optim_args['fprime'] = self.score
        elif optim_method == 'powell':
            optimizer = optimize.fmin_powell
        elif optim_method == 'slsqp':
            optimizer = optimize.fmin_slsqp
        else:
            raise ValueError('optimizer method not available')

        if DEBUG:
            print(np.linalg.det(weights))

        #TODO: add other optimization options and results
        return optimizer(self.gmmobjective, start, args=(weights,),
                         **optim_args)


    def fitgmm_cu(self, start, optim_method='bfgs', optim_args=None):
        '''estimate parameters using continuously updating GMM

        Parameters
        ----------
        start : array_like
            starting values for minimization

        Returns
        -------
        paramest : ndarray
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        uses scipy.optimize.fmin

        '''
##        if not fixed is None:  #fixed not defined in this version
##            raise NotImplementedError

        if optim_args is None:
            optim_args = {}

        if optim_method == 'nm':
            optimizer = optimize.fmin
        elif optim_method == 'bfgs':
            optimizer = optimize.fmin_bfgs
            optim_args['fprime'] = self.score_cu
        elif optim_method == 'ncg':
            optimizer = optimize.fmin_ncg
        else:
            raise ValueError('optimizer method not available')

        #TODO: add other optimization options and results
        return optimizer(self.gmmobjective_cu, start, args=(), **optim_args)

    def start_weights(self, inv=True):
        """Create identity matrix for starting weights"""
        return np.eye(self.nmoms)

    def gmmobjective(self, params, weights):
        '''
        objective function for GMM minimization

        Parameters
        ----------
        params : ndarray
            parameter values at which objective is evaluated
        weights : ndarray
            weighting matrix

        Returns
        -------
        jval : float
            value of objective function

        '''
        moms = self.momcond_mean(params)
        return np.dot(np.dot(moms, weights), moms)
        #moms = self.momcond(params)
        #return np.dot(np.dot(moms.mean(0),weights), moms.mean(0))


    def gmmobjective_cu(self, params, weights_method='cov',
                        wargs=()):
        '''
        objective function for continuously updating  GMM minimization

        Parameters
        ----------
        params : ndarray
            parameter values at which objective is evaluated

        Returns
        -------
        jval : float
            value of objective function

        '''
        moms = self.momcond(params)
        inv_weights = self.calc_weightmatrix(moms, weights_method=weights_method,
                                             wargs=wargs)
        weights = np.linalg.pinv(inv_weights)
        self._weights_cu = weights  # store if we need it later
        return np.dot(np.dot(moms.mean(0), weights), moms.mean(0))


    def fititer(self, start, maxiter=2, start_invweights=None,
                    weights_method='cov', wargs=(), optim_method='bfgs',
                    optim_args=None):
        '''iterative estimation with updating of optimal weighting matrix

        stopping criteria are maxiter or change in parameter estimate less
        than self.epsilon_iter, with default 1e-6.

        Parameters
        ----------
        start : ndarray
            starting value for parameters
        maxiter : int
            maximum number of iterations
        start_weights : array (nmoms, nmoms)
            initial weighting matrix; if None, then the identity matrix
            is used
        weights_method : {'cov', ...}
            method to use to estimate the optimal weighting matrix,
            see calc_weightmatrix for details

        Returns
        -------
        params : ndarray
            estimated parameters
        weights : ndarray
            optimal weighting matrix calculated with final parameter
            estimates

        Notes
        -----




        '''
        self.history = []
        momcond = self.momcond

        if start_invweights is None:
            w = self.start_weights(inv=True)
        else:
            w = start_invweights

        #call fitgmm function
        #args = (self.endog, self.exog, self.instrument)
        #args is not used in the method version
        winv_new = w
        for it in range(maxiter):
            winv = winv_new
            w = np.linalg.pinv(winv)
            #this is still calling function not method
##            resgmm = fitgmm(momcond, (), start, weights=winv, fixed=None,
##                            weightsoptimal=False)
            resgmm = self.fitgmm(start, weights=w, optim_method=optim_method,
                                 optim_args=optim_args)

            moms = momcond(resgmm)
            # the following is S = cov_moments
            winv_new = self.calc_weightmatrix(moms,
                                              weights_method=weights_method,
                                              wargs=wargs, params=resgmm)

            if it > 2 and maxabs(resgmm - start) < self.epsilon_iter:
                #check rule for early stopping
                # TODO: set has_optimal_weights = True
                break

            start = resgmm
        return resgmm, w


    def calc_weightmatrix(self, moms, weights_method='cov', wargs=(),
                          params=None):
        '''
        calculate omega or the weighting matrix

        Parameters
        ----------
        moms : ndarray
            moment conditions (nobs x nmoms) for all observations evaluated at
            a parameter value
        weights_method : str 'cov'
            If method='cov' is cov then the matrix is calculated as simple
            covariance of the moment conditions.
            see fit method for available aoptions for the weight and covariance
            matrix
        wargs : tuple or dict
            parameters that are required by some kernel methods to
            estimate the long-run covariance. Not used yet.

        Returns
        -------
        w : array (nmoms, nmoms)
            estimate for the weighting matrix or covariance of the moment
            condition


        Notes
        -----

        currently a constant cutoff window is used
        TODO: implement long-run cov estimators, kernel-based

        Newey-West
        Andrews
        Andrews-Moy????

        References
        ----------
        Greene
        Hansen, Bruce

        '''
        nobs, k_moms = moms.shape
        # TODO: wargs are tuple or dict ?
        if DEBUG:
            print(' momcov wargs', wargs)

        centered = not ('centered' in wargs and not wargs['centered'])
        if not centered:
            # caller does not want centered moment conditions
            moms_ = moms
        else:
            moms_ = moms - moms.mean()

        # TODO: store this outside to avoid doing this inside optimization loop
        # TODO: subclasses need to be able to add weights_methods, and remove
        #       IVGMM can have homoscedastic (OLS),
        #       some options will not make sense in some cases
        #       possible add all here and allow subclasses to define a list
        # TODO: should other weights_methods also have `ddof`
        if weights_method == 'cov':
            w = np.dot(moms_.T, moms_)
            if 'ddof' in wargs:
                # caller requests degrees of freedom correction
                if wargs['ddof'] == 'k_params':
                    w /= (nobs - self.k_params)
                else:
                    if DEBUG:
                        print(' momcov ddof', wargs['ddof'])
                    w /= (nobs - wargs['ddof'])
            else:
                # default: divide by nobs
                w /= nobs

        elif weights_method == 'flatkernel':
            #uniform cut-off window
            # This was a trial version, can use HAC with flatkernel
            if 'maxlag' not in wargs:
                raise ValueError('flatkernel requires maxlag')

            maxlag = wargs['maxlag']
            h = np.ones(maxlag + 1)
            w = np.dot(moms_.T, moms_)/nobs
            for i in range(1,maxlag+1):
                w += (h[i] * np.dot(moms_[i:].T, moms_[:-i]) / (nobs-i))

        elif weights_method == 'hac':
            maxlag = wargs['maxlag']
            if 'kernel' in wargs:
                weights_func = wargs['kernel']
            else:
                weights_func = smcov.weights_bartlett
                wargs['kernel'] = weights_func

            w = smcov.S_hac_simple(moms_, nlags=maxlag,
                                   weights_func=weights_func)
            w /= nobs #(nobs - self.k_params)

        elif weights_method == 'iid':
            # only when we have instruments and residual mom = Z * u
            # TODO: problem we do not have params in argument
            #       I cannot keep everything in here w/o params as argument
            u = self.get_error(params)

            if centered:
                # Note: I'm not centering instruments,
                #    should not we always center u? Ok, with centered as default
                u -= u.mean(0)  #demean inplace, we do not need original u

            instrument = self.instrument
            w = np.dot(instrument.T, instrument).dot(np.dot(u.T, u)) / nobs
            if 'ddof' in wargs:
                # caller requests degrees of freedom correction
                if wargs['ddof'] == 'k_params':
                    w /= (nobs - self.k_params)
                else:
                    # assume ddof is a number
                    if DEBUG:
                        print(' momcov ddof', wargs['ddof'])
                    w /= (nobs - wargs['ddof'])
            else:
                # default: divide by nobs
                w /= nobs

        else:
            raise ValueError('weight method not available')

        return w


    def momcond_mean(self, params):
        '''
        mean of moment conditions,

        '''

        momcond = self.momcond(params)
        self.nobs_moms, self.k_moms = momcond.shape
        return momcond.mean(0)


    def gradient_momcond(self, params, epsilon=1e-4, centered=True):
        '''gradient of moment conditions

        Parameters
        ----------
        params : ndarray
            parameter at which the moment conditions are evaluated
        epsilon : float
            stepsize for finite difference calculation
        centered : bool
            This refers to the finite difference calculation. If `centered`
            is true, then the centered finite difference calculation is
            used. Otherwise the one-sided forward differences are used.

        TODO: looks like not used yet
              missing argument `weights`

        '''

        momcond = self.momcond_mean

        # TODO: approx_fprime has centered keyword
        if centered:
            gradmoms = (approx_fprime(params, momcond, epsilon=epsilon) +
                    approx_fprime(params, momcond, epsilon=-epsilon))/2
        else:
            gradmoms = approx_fprime(params, momcond, epsilon=epsilon)

        return gradmoms

    def score(self, params, weights, epsilon=None, centered=True):
        """Score"""
        deriv = approx_fprime(params, self.gmmobjective, args=(weights,),
                              centered=centered, epsilon=epsilon)

        return deriv

    def score_cu(self, params, epsilon=None, centered=True):
        """Score cu"""
        deriv = approx_fprime(params, self.gmmobjective_cu, args=(),
                              centered=centered, epsilon=epsilon)

        return deriv


# TODO: wrong superclass, I want tvalues, ... right now
class GMMResults(LikelihoodModelResults):
    '''just a storage class right now'''

    use_t = False

    def __init__(self, *args, **kwds):
        self.__dict__.update(kwds)

        self.nobs = self.model.nobs
        self.df_resid = np.inf

        self.cov_params_default = self._cov_params()

    @cache_readonly
    def q(self):
        """Objective function at params"""
        return self.model.gmmobjective(self.params, self.weights)

    @cache_readonly
    def jval(self):
        """nobs_moms attached by momcond_mean"""
        return self.q * self.model.nobs_moms

    def _cov_params(self, **kwds):
        #TODO add options ???)
        # this should use by default whatever options have been specified in
        # fit

        # TODO: do not do this when we want to change options
#         if hasattr(self, '_cov_params'):
#             #replace with decorator later
#             return self._cov_params

        # set defaults based on fit arguments
        if 'wargs' not in kwds:
            # Note: we do not check the keys in wargs, use either all or nothing
            kwds['wargs'] = self.wargs
        if 'weights_method' not in kwds:
            kwds['weights_method'] = self.options_other['weights_method']
        if 'has_optimal_weights' not in kwds:
            kwds['has_optimal_weights'] = self.options_other['has_optimal_weights']

        gradmoms = self.model.gradient_momcond(self.params)
        moms = self.model.momcond(self.params)
        covparams = self.calc_cov_params(moms, gradmoms, **kwds)

        return covparams


    def calc_cov_params(self, moms, gradmoms, weights=None, use_weights=False,
                                              has_optimal_weights=True,
                                              weights_method='cov', wargs=()):
        '''calculate covariance of parameter estimates

        not all options tried out yet

        If weights matrix is given, then the formula use to calculate cov_params
        depends on whether has_optimal_weights is true.
        If no weights are given, then the weight matrix is calculated with
        the given method, and has_optimal_weights is assumed to be true.

        (API Note: The latter assumption could be changed if we allow for
        has_optimal_weights=None.)

        '''

        nobs = moms.shape[0]

        if weights is None:
            #omegahat = self.model.calc_weightmatrix(moms, method=method, wargs=wargs)
            #has_optimal_weights = True
            #add other options, Barzen, ...  longrun var estimators
            # TODO: this might still be inv_weights after fititer
            weights = self.weights
        else:
            pass
            #omegahat = weights   #2 different names used,
            #TODO: this is wrong, I need an estimate for omega

        if use_weights:
            omegahat = weights
        else:
            omegahat = self.model.calc_weightmatrix(
                                                moms,
                                                weights_method=weights_method,
                                                wargs=wargs,
                                                params=self.params)


        if has_optimal_weights: #has_optimal_weights:
            # TOD0 make has_optimal_weights depend on convergence or iter >2
            cov = np.linalg.inv(np.dot(gradmoms.T,
                                    np.dot(np.linalg.inv(omegahat), gradmoms)))
        else:
            gw = np.dot(gradmoms.T, weights)
            gwginv = np.linalg.inv(np.dot(gw, gradmoms))
            cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)
            #cov /= nobs

        return cov/nobs

    @property
    def bse_(self):
        '''standard error of the parameter estimates
        '''
        return self.get_bse()

    def get_bse(self, **kwds):
        '''standard error of the parameter estimates with options

        Parameters
        ----------
        kwds : optional keywords
            options for calculating cov_params

        Returns
        -------
        bse : ndarray
            estimated standard error of parameter estimates

        '''
        return np.sqrt(np.diag(self.cov_params(**kwds)))

    def jtest(self):
        '''overidentification test

        I guess this is missing a division by nobs,
        what's the normalization in jval ?
        '''

        jstat = self.jval
        nparams = self.params.size #self.nparams
        df = self.model.nmoms - nparams
        return jstat, stats.chi2.sf(jstat, df), df


    def compare_j(self, other):
        '''overidentification test for comparing two nested gmm estimates

        This assumes that some moment restrictions have been dropped in one
        of the GMM estimates relative to the other.

        Not tested yet

        We are comparing two separately estimated models, that use different
        weighting matrices. It is not guaranteed that the resulting
        difference is positive.

        TODO: Check in which cases Stata programs use the same weigths

        '''
        jstat1 = self.jval
        k_moms1 = self.model.nmoms
        jstat2 = other.jval
        k_moms2 = other.model.nmoms
        jdiff = jstat1 - jstat2
        df = k_moms1 - k_moms2
        if df < 0:
            # possible nested in other way, TODO allow this or not
            # flip sign instead of absolute
            df = - df
            jdiff = - jdiff
        return jdiff, stats.chi2.sf(jdiff, df), df

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Default is `var_##` for ## in p the number of regressors
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """
        #TODO: add a summary text for options that have been used

        jvalue, jpvalue, jdf = self.jtest()

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['GMM']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    #('Df Residuals:', None), #[self.df_resid]), #TODO: spelling
                    #('Df Model:', None), #[self.df_model])
                    ]

        top_right = [#('R-squared:', ["%#8.3f" % self.rsquared]),
                     #('Adj. R-squared:', ["%#8.3f" % self.rsquared_adj]),
                     ('Hansen J:', ["%#8.4g" % jvalue] ),
                     ('Prob (Hansen J):', ["%#6.3g" % jpvalue]),
                     #('F-statistic:', ["%#8.4g" % self.fvalue] ),
                     #('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     #('Log-Likelihood:', None), #["%#6.4g" % self.llf]),
                     #('AIC:', ["%#8.4g" % self.aic]),
                     #('BIC:', ["%#8.4g" % self.bic])
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Results"

        # create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        return smry



class IVGMM(GMM):
    '''
    Basic class for instrumental variables estimation using GMM

    A linear function for the conditional mean is defined as default but the
    methods should be overwritten by subclasses, currently `LinearIVGMM` and
    `NonlinearIVGMM` are implemented as subclasses.

    See Also
    --------
    LinearIVGMM
    NonlinearIVGMM

    '''

    results_class = 'IVGMMResults'

    def fitstart(self):
        """Create array of zeros"""
        return np.zeros(self.exog.shape[1])

    def start_weights(self, inv=True):
        """Starting weights"""
        zz = np.dot(self.instrument.T, self.instrument)
        nobs = self.instrument.shape[0]
        if inv:
            return zz / nobs
        else:
            return np.linalg.pinv(zz / nobs)

    def get_error(self, params):
        """Get error at params"""
        return self.endog - self.predict(params)

    def predict(self, params, exog=None):
        """Get prediction at params"""
        if exog is None:
            exog = self.exog

        return np.dot(exog, params)

    def momcond(self, params):
        """Error times instrument"""
        instrument = self.instrument
        return instrument * self.get_error(params)[:, None]


class LinearIVGMM(IVGMM):
    """class for linear instrumental variables models estimated with GMM

    Uses closed form expression instead of nonlinear optimizers for each step
    of the iterative GMM.

    The model is assumed to have the following moment condition

        E( z * (y - x beta)) = 0

    Where `y` is the dependent endogenous variable, `x` are the explanatory
    variables and `z` are the instruments. Variables in `x` that are exogenous
    need also be included in `z`.

    Notation Warning: our name `exog` stands for the explanatory variables,
    and includes both exogenous and explanatory variables that are endogenous,
    i.e. included endogenous variables

    Parameters
    ----------
    endog : array_like
        dependent endogenous variable
    exog : array_like
        explanatory, right hand side variables, including explanatory variables
        that are endogenous
    instrument : array_like
        Instrumental variables, variables that are exogenous to the error
        in the linear model containing both included and excluded exogenous
        variables
    """

    def fitgmm(self, start, weights=None, optim_method=None, **kwds):
        '''estimate parameters using GMM for linear model

        Uses closed form expression instead of nonlinear optimizers

        Parameters
        ----------
        start : not used
            starting values for minimization, not used, only for consistency
            of method signature
        weights : ndarray
            weighting matrix for moment conditions. If weights is None, then
            the identity matrix is used
        optim_method : not used,
            optimization method, not used, only for consistency of method
            signature
        **kwds : keyword arguments
            not used, will be silently ignored (for compatibility with generic)


        Returns
        -------
        paramest : ndarray
            estimated parameters

        '''
##        if not fixed is None:  #fixed not defined in this version
##            raise NotImplementedError

        # TODO: should start_weights only be in `fit`
        if weights is None:
            weights = self.start_weights(inv=False)

        y, x, z = self.endog, self.exog, self.instrument

        zTx = np.dot(z.T, x)
        zTy = np.dot(z.T, y)
        # normal equation, solved with pinv
        part0 = zTx.T.dot(weights)
        part1 = part0.dot(zTx)
        part2 = part0.dot(zTy)
        params = np.linalg.pinv(part1).dot(part2)

        return params


    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog

        return np.dot(exog, params)


    def gradient_momcond(self, params, **kwds):
        # **kwds for compatibility not used

        x, z = self.exog, self.instrument
        gradmoms = -np.dot(z.T, x) / self.nobs

        return gradmoms

    def score(self, params, weights, **kwds):
        # **kwds for compatibility, not used
        # Note: I coud use general formula with gradient_momcond instead

        x, z = self.exog, self.instrument
        nobs = z.shape[0]

        u = self.get_errors(params)
        score = -2 * np.dot(x.T, z).dot(weights.dot(np.dot(z.T, u)))
        score /= nobs * nobs

        return score



class NonlinearIVGMM(IVGMM):
    """
    Class for non-linear instrumental variables estimation using GMM

    The model is assumed to have the following moment condition

        E[ z * (y - f(X, beta)] = 0

    Where `y` is the dependent endogenous variable, `x` are the explanatory
    variables and `z` are the instruments. Variables in `x` that are exogenous
    need also be included in z. `f` is a nonlinear function.

    Notation Warning: our name `exog` stands for the explanatory variables,
    and includes both exogenous and explanatory variables that are endogenous,
    i.e. included endogenous variables

    Parameters
    ----------
    endog : array_like
        dependent endogenous variable
    exog : array_like
        explanatory, right hand side variables, including explanatory variables
        that are endogenous.
    instruments : array_like
        Instrumental variables, variables that are exogenous to the error
        in the linear model containing both included and excluded exogenous
        variables
    func : callable
        function for the mean or conditional expectation of the endogenous
        variable. The function will be called with parameters and the array of
        explanatory, right hand side variables, `func(params, exog)`

    Notes
    -----
    This class uses numerical differences to obtain the derivative of the
    objective function. If the jacobian of the conditional mean function, `func`
    is available, then it can be used by subclassing this class and defining
    a method `jac_func`.

    TODO: check required signature of jac_error and jac_func
    """
    # This should be reversed:
    # NonlinearIVGMM is IVGMM and need LinearIVGMM as special case (fit, predict)


    def fitstart(self):
        #might not make sense for more general functions
        return np.zeros(self.exog.shape[1])


    def __init__(self, endog, exog, instrument, func, **kwds):
        self.func = func
        super(NonlinearIVGMM, self).__init__(endog, exog, instrument, **kwds)


    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog

        return self.func(params, exog)

    #----------  the following a semi-general versions,
    # TODO: move to higher class after testing

    def jac_func(self, params, weights, args=None, centered=True, epsilon=None):

        # TODO: Why are ther weights in the signature - copy-paste error?
        deriv = approx_fprime(params, self.func, args=(self.exog,),
                              centered=centered, epsilon=epsilon)

        return deriv


    def jac_error(self, params, weights, args=None, centered=True,
                   epsilon=None):

        jac_func = self.jac_func(params, weights, args=None, centered=True,
                                 epsilon=None)

        return -jac_func


    def score(self, params, weights, **kwds):
        # **kwds for compatibility not used
        # Note: I coud use general formula with gradient_momcond instead

        z = self.instrument
        nobs = z.shape[0]

        jac_u = self.jac_error(params, weights, args=None, epsilon=None,
                               centered=True)
        x = -jac_u  # alias, plays the same role as X in linear model

        u = self.get_error(params)

        score = -2 * np.dot(np.dot(x.T, z), weights).dot(np.dot(z.T, u))
        score /= nobs * nobs

        return score


class IVGMMResults(GMMResults):
    """Results class of IVGMM"""
    # this assumes that we have an additive error model `(y - f(x, params))`

    @cache_readonly
    def fittedvalues(self):
        """Fitted values"""
        return self.model.predict(self.params)


    @cache_readonly
    def resid(self):
        """Residuals"""
        return self.model.endog - self.fittedvalues


    @cache_readonly
    def ssr(self):
        """Sum of square errors"""
        return (self.resid * self.resid).sum(0)




def spec_hausman(params_e, params_i, cov_params_e, cov_params_i, dof=None):
    '''Hausmans specification test

    Parameters
    ----------
    params_e : ndarray
        efficient and consistent under Null hypothesis,
        inconsistent under alternative hypothesis
    params_i : ndarray
        consistent under Null hypothesis,
        consistent under alternative hypothesis
    cov_params_e : ndarray, 2d
        covariance matrix of parameter estimates for params_e
    cov_params_i : ndarray, 2d
        covariance matrix of parameter estimates for params_i

    example instrumental variables OLS estimator is `e`, IV estimator is `i`


    Notes
    -----

    Todos,Issues
    - check dof calculations and verify for linear case
    - check one-sided hypothesis


    References
    ----------
    Greene section 5.5 p.82/83


    '''
    params_diff = (params_i - params_e)
    cov_diff = cov_params_i - cov_params_e
    #TODO: the following is very inefficient, solves problem (svd) twice
    #use linalg.lstsq or svd directly
    #cov_diff will very often be in-definite (singular)
    if not dof:
        dof = np.linalg.matrix_rank(cov_diff)
    cov_diffpinv = np.linalg.pinv(cov_diff)
    H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))
    pval = stats.chi2.sf(H, dof)

    evals = np.linalg.eigvalsh(cov_diff)

    return H, pval, dof, evals




###########

class DistQuantilesGMM(GMM):
    '''
    Estimate distribution parameters by GMM based on matching quantiles

    Currently mainly to try out different requirements for GMM when we cannot
    calculate the optimal weighting matrix.

    '''

    def __init__(self, endog, exog, instrument, **kwds):
        #TODO: something wrong with super
        super(DistQuantilesGMM, self).__init__(endog, exog, instrument)
        #self.func = func
        self.epsilon_iter = 1e-5

        self.distfn = kwds['distfn']
        #done by super does not work yet
        #TypeError: super does not take keyword arguments
        self.endog = endog

        #make this optional for fit
        if 'pquant' not in kwds:
            self.pquant = pquant = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
        else:
            self.pquant = pquant = kwds['pquant']

        #TODO: vectorize this: use edf
        self.xquant = np.array([stats.scoreatpercentile(endog, p) for p
                                in pquant*100])
        self.nmoms = len(self.pquant)

        #TODOcopied from GMM, make super work
        self.endog = endog
        self.exog = exog
        self.instrument = instrument
        self.results = GMMResults(model=self)
        #self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def fitstart(self):
        #todo: replace with or add call to distfn._fitstart
        #      added but not used during testing
        distfn = self.distfn
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(self.endog)
        else:
            start = [1]*distfn.numargs + [0.,1.]

        return np.asarray(start)

    def momcond(self, params): #drop distfn as argument
        #, mom2, quantile=None, shape=None
        '''moment conditions for estimating distribution parameters by matching
        quantiles, defines as many moment conditions as quantiles.

        Returns
        -------
        difference : ndarray
            difference between theoretical and empirical quantiles

        Notes
        -----
        This can be used for method of moments or for generalized method of
        moments.

        '''
        #this check looks redundant/unused know
        if len(params) == 2:
            loc, scale = params
        elif len(params) == 3:
            shape, loc, scale = params
        else:
            #raise NotImplementedError
            pass #see whether this might work, seems to work for beta with 2 shape args

        #mom2diff = np.array(distfn.stats(*params)) - mom2
        #if not quantile is None:
        pq, xq = self.pquant, self.xquant
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = self.distfn.cdf(xq, *params) - pq
        #return np.concatenate([mom2diff, cdfdiff[:1]])
        return np.atleast_2d(cdfdiff)

    def fitonce(self, start=None, weights=None, has_optimal_weights=False):
        '''fit without estimating an optimal weighting matrix and return results

        This is a convenience function that calls fitgmm and covparams with
        a given weight matrix or the identity weight matrix.
        This is useful if the optimal weight matrix is know (or is analytically
        given) or if an optimal weight matrix cannot be calculated.

        (Developer Notes: this function could go into GMM, but is needed in this
        class, at least at the moment.)

        Parameters
        ----------


        Returns
        -------
        results : GMMResult instance
            result instance with params and _cov_params attached

        See Also
        --------
        fitgmm
        cov_params

        '''
        if weights is None:
            weights = np.eye(self.nmoms)
        params = self.fitgmm(start=start)
        # TODO: rewrite this old hack, should use fitgmm or fit maxiter=0
        self.results.params = params  #required before call to self.cov_params
        self.results.wargs = {} #required before call to self.cov_params
        self.results.options_other = {'weights_method':'cov'}
        # TODO: which weights_method?  There should not be any needed ?
        _cov_params = self.results.cov_params(weights=weights,
                                      has_optimal_weights=has_optimal_weights)

        self.results.weights = weights
        self.results.jval = self.gmmobjective(params, weights)
        self.results.options_other.update({'has_optimal_weights':has_optimal_weights})

        return self.results


results_class_dict = {'GMMResults': GMMResults,
                      'IVGMMResults': IVGMMResults,
                      'DistQuantilesGMM': GMMResults}  #TODO: should be a default
