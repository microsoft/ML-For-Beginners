import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly

"""
Elastic net regularization.

Routines for fitting regression models using elastic net
regularization.  The elastic net minimizes the objective function

-llf / nobs + alpha((1 - L1_wt) * sum(params**2) / 2 +
    L1_wt * sum(abs(params)))

The algorithm implemented here closely follows the implementation in
the R glmnet package, documented here:

http://cran.r-project.org/web/packages/glmnet/index.html

and here:

http://www.jstatsoft.org/v33/i01/paper

This routine should work for any regression model that implements
loglike, score, and hess.
"""


def _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds):
    """
    Negative penalized log-likelihood functions.

    Returns the negative penalized log-likelihood, its derivative, and
    its Hessian.  The penalty only includes the smooth (L2) term.

    All three functions have argument signature (x, model), where
    ``x`` is a point in the parameter space and ``model`` is an
    arbitrary statsmodels regression model.
    """

    def nploglike(params, model):
        nobs = model.nobs
        pen_llf = alpha[k] * (1 - L1_wt) * np.sum(params**2) / 2
        llf = model.loglike(np.r_[params], **loglike_kwds)
        return - llf / nobs + pen_llf

    def npscore(params, model):
        nobs = model.nobs
        pen_grad = alpha[k] * (1 - L1_wt) * params
        gr = -model.score(np.r_[params], **score_kwds)[0] / nobs
        return gr + pen_grad

    def nphess(params, model):
        nobs = model.nobs
        pen_hess = alpha[k] * (1 - L1_wt)
        h = -model.hessian(np.r_[params], **hess_kwds)[0, 0] / nobs + pen_hess
        return h

    return nploglike, npscore, nphess


def fit_elasticnet(model, method="coord_descent", maxiter=100,
                   alpha=0., L1_wt=1., start_params=None, cnvrg_tol=1e-7,
                   zero_tol=1e-8, refit=False, check_step=True,
                   loglike_kwds=None, score_kwds=None, hess_kwds=None):
    """
    Return an elastic net regularized fit to a regression model.

    Parameters
    ----------
    model : model object
        A statsmodels object implementing ``loglike``, ``score``, and
        ``hessian``.
    method : {'coord_descent'}
        Only the coordinate descent algorithm is implemented.
    maxiter : int
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array_like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in one iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    refit : bool
        If True, the model is refit using only the variables that have
        non-zero coefficients in the regularized fit.  The refitted
        model is not regularized.
    check_step : bool
        If True, confirm that the first step is an improvement and search
        further if it is not.
    loglike_kwds : dict-like or None
        Keyword arguments for the log-likelihood function.
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    Results
        A results object.

    Notes
    -----
    The ``elastic net`` penalty is a combination of L1 and L2
    penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then repeatedly optimize the L1 penalized version of this function
    along coordinate axes.
    """

    k_exog = model.exog.shape[1]

    loglike_kwds = {} if loglike_kwds is None else loglike_kwds
    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds

    if np.isscalar(alpha):
        alpha = alpha * np.ones(k_exog)

    # Define starting params
    if start_params is None:
        params = np.zeros(k_exog)
    else:
        params = start_params.copy()

    btol = 1e-4
    params_zero = np.zeros(len(params), dtype=bool)

    init_args = model._get_init_kwds()
    # we do not need a copy of init_args b/c get_init_kwds provides new dict
    init_args['hasconst'] = False
    model_offset = init_args.pop('offset', None)
    if 'exposure' in init_args and init_args['exposure'] is not None:
        if model_offset is None:
            model_offset = np.log(init_args.pop('exposure'))
        else:
            model_offset += np.log(init_args.pop('exposure'))

    fgh_list = [
        _gen_npfuncs(k, L1_wt, alpha, loglike_kwds, score_kwds, hess_kwds)
        for k in range(k_exog)]

    converged = False

    for itr in range(maxiter):

        # Sweep through the parameters
        params_save = params.copy()
        for k in range(k_exog):

            # Under the active set method, if a parameter becomes
            # zero we do not try to change it again.
            # TODO : give the user the option to switch this off
            if params_zero[k]:
                continue

            # Set the offset to account for the variables that are
            # being held fixed in the current coordinate
            # optimization.
            params0 = params.copy()
            params0[k] = 0
            offset = np.dot(model.exog, params0)
            if model_offset is not None:
                offset += model_offset

            # Create a one-variable model for optimization.
            model_1var = model.__class__(
                model.endog, model.exog[:, k], offset=offset, **init_args)

            # Do the one-dimensional optimization.
            func, grad, hess = fgh_list[k]
            params[k] = _opt_1d(
                func, grad, hess, model_1var, params[k], alpha[k]*L1_wt,
                tol=btol, check_step=check_step)

            # Update the active set
            if itr > 0 and np.abs(params[k]) < zero_tol:
                params_zero[k] = True
                params[k] = 0.

        # Check for convergence
        pchange = np.max(np.abs(params - params_save))
        if pchange < cnvrg_tol:
            converged = True
            break

    # Set approximate zero coefficients to be exactly zero
    params[np.abs(params) < zero_tol] = 0

    if not refit:
        results = RegularizedResults(model, params)
        results.converged = converged
        return RegularizedResultsWrapper(results)

    # Fit the reduced model to get standard errors and other
    # post-estimation results.
    ii = np.flatnonzero(params)
    cov = np.zeros((k_exog, k_exog))
    init_args = dict([(k, getattr(model, k, None)) for k in model._init_keys])
    if len(ii) > 0:
        model1 = model.__class__(
            model.endog, model.exog[:, ii], **init_args)
        rslt = model1.fit()
        params[ii] = rslt.params
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params
    else:
        # Hack: no variables were selected but we need to run fit in
        # order to get the correct results class.  So just fit a model
        # with one variable.
        model1 = model.__class__(model.endog, model.exog[:, 0], **init_args)
        rslt = model1.fit(maxiter=0)

    # fit may return a results or a results wrapper
    if issubclass(rslt.__class__, wrap.ResultsWrapper):
        klass = rslt._results.__class__
    else:
        klass = rslt.__class__

    # Not all models have a scale
    if hasattr(rslt, 'scale'):
        scale = rslt.scale
    else:
        scale = 1.

    # The degrees of freedom should reflect the number of parameters
    # in the refit model, not including the zeros that are displayed
    # to indicate which variables were dropped.  See issue #1723 for
    # discussion about setting df parameters in model and results
    # classes.
    p, q = model.df_model, model.df_resid
    model.df_model = len(ii)
    model.df_resid = model.nobs - model.df_model

    # Assuming a standard signature for creating results classes.
    refit = klass(model, params, cov, scale=scale)
    refit.regularized = True
    refit.converged = converged
    refit.method = method
    refit.fit_history = {'iteration': itr + 1}

    # Restore df in model class, see issue #1723 for discussion.
    model.df_model, model.df_resid = p, q

    return refit


def _opt_1d(func, grad, hess, model, start, L1_wt, tol,
            check_step=True):
    """
    One-dimensional helper for elastic net.

    Parameters
    ----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    model : statsmodels model
        The model being fit.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.
    check_step : bool
        If True, check that the first step is an improvement and
        use bisection if it is not.  If False, return after the
        first step regardless.

    Notes
    -----
    ``func``, ``grad``, and ``hess`` have argument signature (x,
    model), where ``x`` is a point in the parameter space and
    ``model`` is the model being fit.

    If the log-likelihood for the model is exactly quadratic, the
    global minimum is returned in one step.  Otherwise numerical
    bisection is used.

    Returns
    -------
    The argmin of the objective function.
    """

    # Overview:
    # We want to minimize L(x) + L1_wt*abs(x), where L() is a smooth
    # loss function that includes the log-likelihood and L2 penalty.
    # This is a 1-dimensional optimization.  If L(x) is exactly
    # quadratic we can solve for the argmin exactly.  Otherwise we
    # approximate L(x) with a quadratic function Q(x) and try to use
    # the minimizer of Q(x) + L1_wt*abs(x).  But if this yields an
    # uphill step for the actual target function L(x) + L1_wt*abs(x),
    # then we fall back to a expensive line search.  The line search
    # is never needed for OLS.

    x = start
    f = func(x, model)
    b = grad(x, model)
    c = hess(x, model)
    d = b - c*x

    # The optimum is achieved by hard thresholding to zero
    if L1_wt > np.abs(d):
        return 0.

    # x + h is the minimizer of the Q(x) + L1_wt*abs(x)
    if d >= 0:
        h = (L1_wt - b) / c
    elif d < 0:
        h = -(L1_wt + b) / c
    else:
        return np.nan

    # If the new point is not uphill for the target function, take it
    # and return.  This check is a bit expensive and un-necessary for
    # OLS
    if not check_step:
        return x + h
    f1 = func(x + h, model) + L1_wt*np.abs(x + h)
    if f1 <= f + L1_wt*np.abs(x) + 1e-10:
        return x + h

    # Fallback for models where the loss is not quadratic
    from scipy.optimize import brent
    x_opt = brent(func, args=(model,), brack=(x-1, x+1), tol=tol)
    return x_opt


class RegularizedResults(Results):
    """
    Results for models estimated using regularization

    Parameters
    ----------
    model : Model
        The model instance used to estimate the parameters.
    params : ndarray
        The estimated (regularized) parameters.
    """
    def __init__(self, model, params):
        super(RegularizedResults, self).__init__(model, params)

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted values from the model at the estimated parameters.
        """
        return self.model.predict(self.params)


class RegularizedResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
    }
    _wrap_attrs = _attrs
wrap.populate_wrapper(RegularizedResultsWrapper,  # noqa:E305
                      RegularizedResults)
