from __future__ import annotations

from statsmodels.compat.python import lzip

from functools import reduce
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
    ContrastResults,
    WaldTestResults,
    t_test_pairwise,
)
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
    cache_readonly,
    cached_data,
    cached_value,
)
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
    HessianInversionWarning,
    ValueWarning,
)
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like

ERROR_INIT_KWARGS = False

_model_params_doc = """Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`."""

_missing_param_doc = """\
missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'."""
_extra_param_doc = """
    hasconst : None or bool
        Indicates whether the RHS includes a user-supplied constant. If True,
        a constant is not checked for and k_constant is set to 1 and all
        result statistics are calculated as if a constant is present. If
        False, a constant is not checked for and k_constant is set to 0.
    **kwargs
        Extra arguments that are used to set model properties when using the
        formula interface."""


class Model:
    __doc__ = """
    A (predictive) statistical model. Intended to be subclassed not used.

    %(params_doc)s
    %(extra_params_doc)s

    Attributes
    ----------
    exog_names
    endog_names

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """ % {'params_doc': _model_params_doc,
           'extra_params_doc': _missing_param_doc + _extra_param_doc}

    # Maximum number of endogenous variables when using a formula
    # Default is 1, which is more common. Override in models when needed
    # Set to None to skip check
    _formula_max_endog = 1
    # kwargs that are generically allowed, maybe not supported in all models
    _kwargs_allowed = [
        "missing", 'missing_idx', 'formula', 'design_info', "hasconst",
        ]

    def __init__(self, endog, exog=None, **kwargs):
        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)
        self.data = self._handle_data(endog, exog, missing, hasconst,
                                      **kwargs)
        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog
        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog'])
        if 'formula' not in kwargs:  # will not be able to unpickle without these
            self._data_attr.extend(['data.orig_endog', 'data.orig_exog'])
        # store keys for extras if we need to recreate model instance
        # we do not need 'missing', maybe we need 'hasconst'
        self._init_keys = list(kwargs.keys())
        if hasconst is not None:
            self._init_keys.append('hasconst')

    def _get_init_kwds(self):
        """return dictionary with extra keys used in model.__init__
        """
        kwds = dict(((key, getattr(self, key, None))
                     for key in self._init_keys))

        return kwds

    def _check_kwargs(self, kwargs, keys_extra=None, error=ERROR_INIT_KWARGS):

        kwargs_allowed = [
            "missing", 'missing_idx', 'formula', 'design_info', "hasconst",
            ]
        if keys_extra:
            kwargs_allowed.extend(keys_extra)

        kwargs_invalid = [i for i in kwargs if i not in kwargs_allowed]
        if kwargs_invalid:
            msg = "unknown kwargs " + repr(kwargs_invalid)
            if error is False:
                warnings.warn(msg, ValueWarning)
            else:
                raise ValueError(msg)

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        data = handle_data(endog, exog, missing, hasconst, **kwargs)
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
            if key in ['design_info', 'formula']:  # leave attached to data
                continue
            # pop so we do not start keeping all these twice or references
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:  # panel already pops keys in data handling
                pass
        return data

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model.
        data : array_like
            The data for the model. See Notes.
        subset : array_like
            An array-like object of booleans, integers, or index values that
            indicate the subset of df to use in the model. Assumes df is a
            `pandas.DataFrame`.
        drop_cols : array_like
            Columns to drop from the design matrix.  Cannot be used to
            drop terms involving categoricals.
        *args
            Additional positional argument that are passed to the model.
        **kwargs
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model
            The model instance.

        Notes
        -----
        data must define __getitem__ with the keys in the formula terms
        args and kwargs are passed on to the model instantiation. E.g.,
        a numpy structured or rec array, a dictionary, or a pandas DataFrame.
        """
        # TODO: provide a docs template for args/kwargs from child models
        # TODO: subset could use syntax. issue #469.
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop('eval_env', None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment
            eval_env = EvalEnvironment({})
        elif isinstance(eval_env, int):
            eval_env += 1  # we're going down the stack again
        missing = kwargs.get('missing', 'drop')
        if missing == 'none':  # with patsy it's drop or raise. let's raise.
            missing = 'raise'

        tmp = handle_formula_data(data, None, formula, depth=eval_env,
                                  missing=missing)
        ((endog, exog), missing_idx, design_info) = tmp
        max_endog = cls._formula_max_endog
        if (max_endog is not None and
                endog.ndim > 1 and endog.shape[1] > max_endog):
            raise ValueError('endog has evaluated to an array with multiple '
                             'columns that has shape {0}. This occurs when '
                             'the variable converted to endog is non-numeric'
                             ' (e.g., bool or str).'.format(endog.shape))
        if drop_cols is not None and len(drop_cols) > 0:
            cols = [x for x in exog.columns if x not in drop_cols]
            if len(cols) < len(exog.columns):
                exog = exog[cols]
                cols = list(design_info.term_names)
                for col in drop_cols:
                    try:
                        cols.remove(col)
                    except ValueError:
                        pass  # OK if not present
                design_info = design_info.subset(cols)

        kwargs.update({'missing_idx': missing_idx,
                       'missing': missing,
                       'formula': formula,  # attach formula for unpckling
                       'design_info': design_info})
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula
        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod

    @property
    def endog_names(self):
        """
        Names of endogenous variables.
        """
        return self.data.ynames

    @property
    def exog_names(self) -> list[str] | None:
        """
        Names of exogenous variables.
        """
        return self.data.xnames

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError


class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance.

        For example, if the the design matrix of a linear model changes then
        initialized can be used to recompute values using the modified design
        matrix.
        """
        pass

    # TODO: if the intent is to re-initialize the model with new data then this
    # method needs to take inputs...

    def loglike(self, params):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Notes
        -----
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None, retall=False,
            skip_hessian=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        skip_hessian : bool, optional
            If False (default), then the negative inverse hessian is calculated
            after the optimization. If True, then the hessian will not be
            calculated. However, it will be available in methods that use the
            hessian in the optimization (currently only with `"newton"`).
        kwargs : keywords
            All kwargs are passed to the chosen solver with one exception. The
            following keyword controls what happens after the fit::

                warn_convergence : bool, optional
                    If True, checks the model for the converged flag. If the
                    converged flag is False, a ConvergenceWarning is issued.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for solvers (see returned Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'lbfgs'
                m : int
                    This many terms are used for the Hessian approximation.
                factr : float
                    A stop condition that is a variant of relative error.
                pgtol : float
                    A stop condition that uses the projected gradient.
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
                maxfun : int
                    Maximum number of function evaluations to make.
                bounds : sequence
                    (min, max) pairs for each element in x,
                    defining the bounds on that parameter.
                    Use None for one of min or max when there is no bound
                    in that direction.
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : int
                    The number of basin hopping iterations.
                niter_success : int
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : int
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                      - `args` <- `fargs`
                      - `jac` <- `score`
                      - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        Hinv = None  # JP error if full_output=0, Hinv not defined

        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                # fails for shape (K,)?
                start_params = [0] * self.exog.shape[1]
            else:
                raise ValueError("If exog is None, then start_params should "
                                 "be specified")

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime does not take
        # args in most (any?) of the optimize function

        nobs = self.endog.shape[0]
        # f = lambda params, *args: -self.loglike(params, *args) / nobs

        def f(params, *args):
            return -self.loglike(params, *args) / nobs

        if method == 'newton':
            # TODO: why are score and hess positive?
            def score(params, *args):
                return self.score(params, *args) / nobs

            def hess(params, *args):
                return self.hessian(params, *args) / nobs
        else:
            def score(params, *args):
                return -self.score(params, *args) / nobs

            def hess(params, *args):
                return -self.hessian(params, *args) / nobs

        warn_convergence = kwargs.pop('warn_convergence', True)

        # Remove covariance args before calling fir to allow strict checking
        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
            if cov_kwds:
                del kwargs["cov_kwds"]
            del kwargs["cov_type"]
        else:
            kwds = {}
        if 'use_t' in kwargs:
            kwds['use_t'] = kwargs['use_t']
            del kwargs["use_t"]

        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)
        # Restore cov_type, cov_kwds and use_t
        optim_settings.update(kwds)
        # NOTE: this is for fit_regularized and should be generalized
        cov_params_func = kwargs.setdefault('cov_params_func', None)
        if cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        elif not skip_hessian:
            H = -1 * self.hessian(xopt)
            invertible = False
            if np.all(np.isfinite(H)):
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) > 0:
                    invertible = True

            if invertible:
                Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
                Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
            else:
                warnings.warn('Inverting hessian failed, no bse or cov_params '
                              'available', HessianInversionWarning)
                Hinv = None

        # TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1., **kwds)

        # TODO: hardcode scale?
        mlefit.mle_retvals = retvals
        if isinstance(retvals, dict):
            if warn_convergence and not retvals['converged']:
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.warn("Maximum Likelihood optimization failed to "
                              "converge. Check mle_retvals",
                              ConvergenceWarning)

        mlefit.mle_settings = optim_settings
        return mlefit

    def _fit_zeros(self, keep_index=None, start_params=None,
                   return_auxiliary=False, k_params=None, **fit_kwds):
        """experimental, fit the model subject to zero constraints

        Intended for internal use cases until we know what we need.
        API will need to change to handle models with two exog.
        This is not yet supported by all model subclasses.

        This is essentially a simplified version of `fit_constrained`, and
        does not need to use `offset`.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Some subclasses could use a more efficient calculation than using a
        new model.

        Parameters
        ----------
        keep_index : array_like (int or bool) or slice
            variables that should be dropped.
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        k_params : int or None
            If None, then we try to infer from start_params or model.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
        # we need to append index of extra params to keep_index as in
        # NegativeBinomial
        if hasattr(self, 'k_extra') and self.k_extra > 0:
            # we cannot change the original, TODO: should we add keep_index_params?
            keep_index = np.array(keep_index, copy=True)
            k = self.exog.shape[1]
            extra_index = np.arange(k, k + self.k_extra)
            keep_index_p = np.concatenate((keep_index, extra_index))
        else:
            keep_index_p = keep_index

        # not all models support start_params, drop if None, hide them in fit_kwds
        if start_params is not None:
            fit_kwds['start_params'] = start_params[keep_index_p]
            k_params = len(start_params)
            # ignore k_params in this case, or verify consisteny?

        # build auxiliary model and fit
        init_kwds = self._get_init_kwds()
        mod_constr = self.__class__(self.endog, self.exog[:, keep_index],
                                    **init_kwds)
        res_constr = mod_constr.fit(**fit_kwds)
        # switch name, only need keep_index for params below
        keep_index = keep_index_p

        if k_params is None:
            k_params = self.exog.shape[1]
            k_params += getattr(self, 'k_extra', 0)

        params_full = np.zeros(k_params)
        params_full[keep_index] = res_constr.params

        # create dummy results Instance, TODO: wire up properly
        # TODO: this could be moved into separate private method if needed
        # discrete L1 fit_regularized doens't reestimate AFAICS
        # RLM does not have method, disp nor warn_convergence keywords
        # OLS, WLS swallows extra kwds with **kwargs, but does not have method='nm'
        try:
            # Note: addding full_output=False causes exceptions
            res = self.fit(maxiter=0, disp=0, method='nm', skip_hessian=True,
                           warn_convergence=False, start_params=params_full)
            # we get a wrapper back
        except (TypeError, ValueError):
            res = self.fit()

        # Warning: make sure we are not just changing the wrapper instead of
        # results #2400
        # TODO: do we need to change res._results.scale in some models?
        if hasattr(res_constr.model, 'scale'):
            # Note: res.model is self
            # GLM problem, see #2399,
            # TODO: remove from model if not needed anymore
            res.model.scale = res._results.scale = res_constr.model.scale

        if hasattr(res_constr, 'mle_retvals'):
            res._results.mle_retvals = res_constr.mle_retvals
            # not available for not scipy optimization, e.g. glm irls
            # TODO: what retvals should be required?
            # res.mle_retvals['fcall'] = res_constr.mle_retvals.get('fcall', np.nan)
            # res.mle_retvals['iterations'] = res_constr.mle_retvals.get(
            #                                                 'iterations', np.nan)
            # res.mle_retvals['converged'] = res_constr.mle_retvals['converged']
        # overwrite all mle_settings
        if hasattr(res_constr, 'mle_settings'):
            res._results.mle_settings = res_constr.mle_settings

        res._results.params = params_full
        if (not hasattr(res._results, 'normalized_cov_params') or
                res._results.normalized_cov_params is None):
            res._results.normalized_cov_params = np.zeros((k_params, k_params))
        else:
            res._results.normalized_cov_params[...] = 0

        # fancy indexing requires integer array
        keep_index = np.array(keep_index)
        res._results.normalized_cov_params[keep_index[:, None], keep_index] = \
            res_constr.normalized_cov_params
        k_constr = res_constr.df_resid - res._results.df_resid
        if hasattr(res_constr, 'cov_params_default'):
            res._results.cov_params_default = np.zeros((k_params, k_params))
            res._results.cov_params_default[keep_index[:, None], keep_index] = \
                res_constr.cov_params_default
        if hasattr(res_constr, 'cov_type'):
            res._results.cov_type = res_constr.cov_type
            res._results.cov_kwds = res_constr.cov_kwds

        res._results.keep_index = keep_index
        res._results.df_resid = res_constr.df_resid
        res._results.df_model = res_constr.df_model

        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr

        # special temporary workaround for RLM
        # need to be able to override robust covariances
        if hasattr(res.model, 'M'):
            del res._results._cache['resid']
            del res._results._cache['fittedvalues']
            del res._results._cache['sresid']
            cov = res._results._cache['bcov_scaled']
            # inplace adjustment
            cov[...] = 0
            cov[keep_index[:, None], keep_index] = res_constr.bcov_scaled
            res._results.cov_params_default = cov

        return res

    def _fit_collinear(self, atol=1e-14, rtol=1e-13, **kwds):
        """experimental, fit of the model without collinear variables

        This currently uses QR to drop variables based on the given
        sequence.
        Options will be added in future, when the supporting functions
        to identify collinear variables become available.
        """

        # ------ copied from PR #2380 remove when merged
        x = self.exog
        tol = atol + rtol * x.var(0)
        r = np.linalg.qr(x, mode='r')
        mask = np.abs(r.diagonal()) < np.sqrt(tol)
        # TODO add to results instance
        # idx_collinear = np.where(mask)[0]
        idx_keep = np.where(~mask)[0]
        return self._fit_zeros(keep_index=idx_keep, **kwds)


# TODO: the below is unfinished
class GenericLikelihoodModel(LikelihoodModel):
    """
    Allows the fitting of any likelihood function via maximum likelihood.

    A subclass needs to specify at least the log-likelihood
    If the log-likelihood is specified for each observation, then results that
    require the Jacobian will be available. (The other case is not tested yet.)

    Notes
    -----
    Optimization methods that require only a likelihood function are 'nm' and
    'powell'

    Optimization methods that require a likelihood function and a
    score/gradient are 'bfgs', 'cg', and 'ncg'. A function to compute the
    Hessian is optional for 'ncg'.

    Optimization method that require a likelihood function, a score/gradient,
    and a Hessian is 'newton'

    If they are not overwritten by a subclass, then numerical gradient,
    Jacobian and Hessian of the log-likelihood are calculated by numerical
    forward differentiation. This might results in some cases in precision
    problems, and the Hessian might not be positive definite. Even if the
    Hessian is not positive definite the covariance matrix of the parameter
    estimates based on the outer product of the Jacobian might still be valid.


    Examples
    --------
    see also subclasses in directory miscmodels

    import statsmodels.api as sm
    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog)
    # in this dir
    from model import GenericLikelihoodModel
    probit_mod = sm.Probit(data.endog, data.exog)
    probit_res = probit_mod.fit()
    loglike = probit_mod.loglike
    score = probit_mod.score
    mod = GenericLikelihoodModel(data.endog, data.exog, loglike, score)
    res = mod.fit(method="nm", maxiter = 500)
    import numpy as np
    np.allclose(res.params, probit_res.params)
    """
    def __init__(self, endog, exog=None, loglike=None, score=None,
                 hessian=None, missing='none', extra_params_names=None,
                 **kwds):
        # let them be none in case user wants to use inheritance
        if loglike is not None:
            self.loglike = loglike
        if score is not None:
            self.score = score
        if hessian is not None:
            self.hessian = hessian

        hasconst = kwds.pop("hasconst", None)
        self.__dict__.update(kwds)

        # TODO: data structures?

        # TODO temporary solution, force approx normal
        # self.df_model = 9999
        # somewhere: CacheWriteWarning: 'df_model' cannot be overwritten
        super(GenericLikelihoodModel, self).__init__(endog, exog,
                                                     missing=missing,
                                                     hasconst=hasconst,
                                                     **kwds
                                                     )

        # this will not work for ru2nmnl, maybe np.ndim of a dict?
        if exog is not None:
            self.nparams = (exog.shape[1] if np.ndim(exog) == 2 else 1)

        if extra_params_names is not None:
            self._set_extra_params_names(extra_params_names)

    def _set_extra_params_names(self, extra_params_names):
        # check param_names
        if extra_params_names is not None:
            if self.exog is not None:
                self.exog_names.extend(extra_params_names)
            else:
                self.data.xnames = extra_params_names

            self.k_extra = len(extra_params_names)
            if hasattr(self, "df_resid"):
                self.df_resid -= self.k_extra

        self.nparams = len(self.exog_names)

    # this is redundant and not used when subclassing
    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        if not self.score:  # right now score is not optional
            self.score = lambda x: approx_fprime(x, self.loglike)
            if not self.hessian:
                pass
        else:   # can use approx_hess_p if we have a gradient
            if not self.hessian:
                pass
        # Initialize is called by
        # statsmodels.model.LikelihoodModel.__init__
        # and should contain any preprocessing that needs to be done for a model
        if self.exog is not None:
            # assume constant
            er = np.linalg.matrix_rank(self.exog)
            self.df_model = float(er - 1)
            self.df_resid = float(self.exog.shape[0] - er)
        else:
            self.df_model = np.nan
            self.df_resid = np.nan
        super(GenericLikelihoodModel, self).initialize()

    def expandparams(self, params):
        """
        expand to full parameter array when some parameters are fixed

        Parameters
        ----------
        params : ndarray
            reduced parameter array

        Returns
        -------
        paramsfull : ndarray
            expanded parameter array where fixed parameters are included

        Notes
        -----
        Calling this requires that self.fixed_params and self.fixed_paramsmask
        are defined.

        *developer notes:*

        This can be used in the log-likelihood to ...

        this could also be replaced by a more general parameter
        transformation.
        """
        paramsfull = self.fixed_params.copy()
        paramsfull[self.fixed_paramsmask] = params
        return paramsfull

    def reduceparams(self, params):
        """Reduce parameters"""
        return params[self.fixed_paramsmask]

    def loglike(self, params):
        """Log-likelihood of model at params"""
        return self.loglikeobs(params).sum(0)

    def nloglike(self, params):
        """Negative log-likelihood of model at params"""
        return -self.loglikeobs(params).sum(0)

    def loglikeobs(self, params):
        """
        Log-likelihood of the model for all observations at params.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : array_like
            The log likelihood of the model evaluated at `params`.
        """
        return -self.nloglikeobs(params)

    def score(self, params):
        """
        Gradient of log-likelihood evaluated at params
        """
        kwds = {}
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglike, **kwds).ravel()

    def score_obs(self, params, **kwds):
        """
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        """
        # kwds.setdefault('epsilon', 1e-4)
        kwds.setdefault('centered', True)
        return approx_fprime(params, self.loglikeobs, **kwds)

    def hessian(self, params):
        """
        Hessian of log-likelihood evaluated at params
        """
        from statsmodels.tools.numdiff import approx_hess

        # need options for hess (epsilon)
        return approx_hess(params, self.loglike)

    def hessian_factor(self, params, scale=None, observed=True):
        """Weights for calculating Hessian

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        hessian_factor : ndarray, 1d
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`
        """

        raise NotImplementedError

    def fit(self, start_params=None, method='nm', maxiter=500, full_output=1,
            disp=1, callback=None, retall=0, **kwargs):

        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.1 * np.ones(self.nparams)

        if "cov_type" not in kwargs:
            # this will add default cov_type name and description
            kwargs["cov_type"] = 'nonrobust'

        fit_method = super(GenericLikelihoodModel, self).fit
        mlefit = fit_method(start_params=start_params,
                            method=method, maxiter=maxiter,
                            full_output=full_output,
                            disp=disp, callback=callback, **kwargs)

        results_class = getattr(self, 'results_class',
                                GenericLikelihoodModelResults)
        genericmlefit = results_class(self, mlefit)

        # amend param names
        exog_names = [] if (self.exog_names is None) else self.exog_names
        k_miss = len(exog_names) - len(mlefit.params)
        if not k_miss == 0:
            if k_miss < 0:
                self._set_extra_params_names(['par%d' % i
                                              for i in range(-k_miss)])
            else:
                # I do not want to raise after we have already fit()
                warnings.warn('more exog_names than parameters', ValueWarning)

        return genericmlefit


class Results:
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        the previously specified model instance
    params : ndarray
        parameter estimates from the fit model
    """
    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)
        self._data_attr = []
        # Variables to clear from cache
        self._data_in_cache = ['fittedvalues', 'resid', 'wresid']

    def initialize(self, model, params, **kwargs):
        """
        Initialize (possibly re-initialize) a Results instance.

        Parameters
        ----------
        model : Model
            The model instance.
        params : ndarray
            The model parameters.
        **kwargs
            Any additional keyword arguments required to initialize the model.
        """
        self.params = params
        self.model = model
        if hasattr(model, 'k_constant'):
            self.k_constant = model.k_constant

    def _transform_predict_exog(self, exog, transform=True):

        is_pandas = _is_using_pandas(exog, None)
        exog_index = None
        if is_pandas:
            if exog.ndim == 2 or self.params.size == 1:
                exog_index = exog.index
            else:
                exog_index = [exog.index.name]

        if transform and hasattr(self.model, 'formula') and (exog is not None):
            # allow both location of design_info, see #7043
            design_info = (getattr(self.model, "design_info", None) or
                           self.model.data.design_info)
            from patsy import dmatrix
            if isinstance(exog, pd.Series):
                # we are guessing whether it should be column or row
                if (hasattr(exog, 'name') and isinstance(exog.name, str) and
                        exog.name in design_info.describe()):
                    # assume we need one column
                    exog = pd.DataFrame(exog)
                else:
                    # assume we need a row
                    exog = pd.DataFrame(exog).T
                exog_index = exog.index
            orig_exog_len = len(exog)
            is_dict = isinstance(exog, dict)
            try:
                exog = dmatrix(design_info, exog, return_type="dataframe")
            except Exception as exc:
                msg = ('predict requires that you use a DataFrame when '
                       'predicting from a model\nthat was created using the '
                       'formula api.'
                       '\n\nThe original error message returned by patsy is:\n'
                       '{0}'.format(str(str(exc))))
                raise exc.__class__(msg)
            if orig_exog_len > len(exog) and not is_dict:
                if exog_index is None:
                    warnings.warn('nan values have been dropped', ValueWarning)
                else:
                    exog = exog.reindex(exog_index)
            exog_index = exog.index

        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                                   self.model.exog.shape[1] == 1):
                exog = exog[:, None]
            exog = np.atleast_2d(exog)  # needed in count model shape[1]

        return exog, exog_index

    def predict(self, exog=None, transform=True, *args, **kwargs):
        """
        Call self.model.predict with self.params as the first argument.

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict. see Notes below.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        *args
            Additional arguments to pass to the model, see the
            predict method of the model for the details.
        **kwargs
            Additional keywords arguments to pass to the model, see the
            predict method of the model for the details.

        Returns
        -------
        array_like
            See self.model.predict.

        Notes
        -----
        The types of exog that are supported depends on whether a formula
        was used in the specification of the model.

        If a formula was used, then exog is processed in the same way as
        the original data. This transformation needs to have key access to the
        same variable names, and can be a pandas DataFrame or a dict like
        object that contains numpy arrays.

        If no formula was used, then the provided exog needs to have the
        same number of columns as the original exog in the model. No
        transformation of the data is performed except converting it to
        a numpy array.

        Row indices as in pandas data frames are supported, and added to the
        returned prediction.
        """
        exog, exog_index = self._transform_predict_exog(exog,
                                                        transform=transform)

        predict_results = self.model.predict(self.params, exog, *args,
                                             **kwargs)

        if exog_index is not None and not hasattr(predict_results,
                                                  'predicted_values'):
            if predict_results.ndim == 1:
                return pd.Series(predict_results, index=exog_index)
            else:
                return pd.DataFrame(predict_results, index=exog_index)
        else:
            return predict_results

    def summary(self):
        """
        Summary

        Not implemented
        """
        raise NotImplementedError


# TODO: public method?
class LikelihoodModelResults(Results):
    """
    Class to contain results from likelihood models

    Parameters
    ----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)

    Attributes
    ----------
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : ndarray
        The t-values of the standard errors.


    Notes
    -----
    The covariance of params is given by scale times normalized_cov_params.

    Return values by solver if full_output is True during fit:

        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'lbfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                Warning flag:

                - 0 if converged
                - 1 if too many function evaluations or too many iterations
                - 2 if stopped for another reason

            converged : bool
                True: converged.  False: did not converge.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        """

    # by default we use normal distribution
    # can be overwritten by instances or subclasses

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 **kwargs):
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        self._use_t = False
        # robust covariance
        # We put cov_type in kwargs so subclasses can decide in fit whether to
        # use this generic implementation
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            self.use_t = use_t if use_t is not None else False
        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})

            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' +
                                 'covariance matrix of the errors is correctly ' +
                                 'specified.'}
            else:
                from statsmodels.base.covtype import get_robustcov_results
                if cov_kwds is None:
                    cov_kwds = {}
                use_t = self.use_t
                # TODO: we should not need use_t in get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                      use_t=use_t, **cov_kwds)

    def normalized_cov_params(self):
        """See specific model class docstring"""
        raise NotImplementedError

    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True,
                               use_t=None, **cov_kwds):
        if use_self is False:
            raise ValueError("use_self should have been removed long ago.  "
                             "See GH#4401")
        from statsmodels.base.covtype import get_robustcov_results
        if cov_kwds is None:
            cov_kwds = {}

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the ' +
                             'covariance matrix of the errors is correctly ' +
                             'specified.'}
        else:
            # TODO: we should not need use_t in get_robustcov_results
            get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                  use_t=use_t, **cov_kwds)
    @property
    def use_t(self):
        """Flag indicating to use the Student's distribution in inference."""
        return self._use_t

    @use_t.setter
    def use_t(self, value):
        self._use_t = bool(value)

    @cached_value
    def llf(self):
        """Log-likelihood of model"""
        return self.model.loglike(self.params)

    @cached_value
    def bse(self):
        """The standard errors of the parameter estimates."""
        # Issue 3299
        if ((not hasattr(self, 'cov_params_default')) and
                (self.normalized_cov_params is None)):
            bse_ = np.empty(len(self.params))
            bse_[:] = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bse_ = np.sqrt(np.diag(self.cov_params()))
        return bse_

    @cached_value
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.params / self.bse

    @cached_value
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.use_t:
                df_resid = getattr(self, 'df_resid_inference', self.df_resid)
                return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
            else:
                return stats.norm.sf(np.abs(self.tvalues)) * 2

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
                   other=None):
        """
        Compute the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast of the
        estimated parameters or all params multiplied by scale which will
        usually be an estimate of sigma^2.  Scale is assumed to be a scalar.

        Parameters
        ----------
        r_matrix : array_like
            Can be 1d, or 2d.  Can be used alone or with other.
        column : array_like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        cov_p : ndarray, optional
            The covariance of the parameters. If not provided, this value is
            read from `self.normalized_cov_params` or
            `self.cov_params_default`.
        other : array_like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        ndarray
            The covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.

        Notes
        -----
        (The below are assumed to be in matrix notation.)

        If no argument is specified returns the covariance matrix of a model
        ``(scale)*(X.T X)^(-1)``

        If contrast is specified it pre and post-multiplies as follows
        ``(scale) * r_matrix (X.T X)^(-1) r_matrix.T``

        If contrast and other are specified returns
        ``(scale) * r_matrix (X.T X)^(-1) other.T``

        If column is specified returns
        ``(scale) * (X.T X)^(-1)[column,column]`` if column is 0d

        OR

        ``(scale) * (X.T X)^(-1)[column][:,column]`` if column is 1d
        """
        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            dot_fun = nan_dot
        else:
            dot_fun = np.dot

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             '(unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other '
                             'arguments.')
        if other is not None and r_matrix is None:
            raise ValueError('other can only be specified with r_matrix')

        if cov_p is None:
            if hasattr(self, 'cov_params_default'):
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  # if r_matrix is None and column is None:
            return cov_p

    # TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, cov_p=None, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q). If q is given,
              can be either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        use_t : bool, optional
            If use_t is None, then the default of the model is used. If use_t
            is True, then the p-values are based on the t distribution. If
            use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.

        See Also
        --------
        tvalues : Individual t statistics for the estimated parameters.
        f_test : Perform an F tests on model parameters.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[5:] = [1,-1]
        >>> print(r)
        [ 0.  0.  0.  0.  0.  1. -1.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>> T_test = results.t_test(r)
        >>> print(T_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0         -1829.2026    455.391     -4.017      0.003   -2859.368    -799.037
        ==============================================================================
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.tvalue
        -4.0167754636411717
        >>> T_test.pvalue
        0.0015163772380899498

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.formula.api import ols
        >>> dta = sm.datasets.longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR/1829 = 1'
        >>> t_test = results.t_test(hypotheses)
        >>> print(t_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0            15.0977     84.937      0.178      0.863    -177.042     207.238
        c1            -2.0202      0.488     -8.231      0.000      -3.125      -0.915
        c2             1.0001      0.249      0.000      1.000       0.437       1.563
        ==============================================================================
        """
        from patsy import DesignInfo
        use_t = bool_like(use_t, "use_t", strict=True, optional=True)
        names = self.model.data.cov_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('Need covariance of parameters for computing '
                             'T statistics')
        params = self.params.ravel()
        if num_params != params.shape[0]:
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        if use_t is None:
            # switch to use_t false if undefined
            use_t = (hasattr(self, 'use_t') and self.use_t)

        _effect = np.dot(r_matrix, params)

        # Perform the test
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(
                r_matrix=r_matrix, cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
        _t = (_effect - q_matrix) * recipr(_sd)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)

        if use_t:
            return ContrastResults(effect=_effect, t=_t, sd=_sd,
                                   df_denom=df_resid)
        else:
            return ContrastResults(effect=_effect, statistic=_t, sd=_sd,
                                   df_denom=df_resid,
                                   distribution='norm')

    def f_test(self, r_matrix, cov_p=None, invcov=None):
        """
        Compute the F-test for a joint linear hypothesis.

        This is a special case of `wald_test` that always uses the F
        distribution.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length k row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        t_test : Perform a single hypothesis test.
        wald_test : Perform a Wald-test using a quadratic form.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print(results.f_test(A))
        <F test: F=array([[ 330.28533923]]), p=4.984030528700946e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.fvalue
        330.2853392346658
        >>> results.f_pvalue
        4.98403096572e-10

        >>> B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print(results.f_test(B))
        <F test: F=array([[ 9.74046187]]), p=0.005605288531708235, df_denom=9, df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.datasets import longley
        >>> from statsmodels.formula.api import ols
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print(f_test)
        <F test: F=array([[ 144.17976065]]), p=6.322026217355609e-08, df_denom=9, df_num=3>
        """
        res = self.wald_test(r_matrix, cov_p=cov_p, invcov=invcov, use_f=True, scalar=True)
        return res

    # TODO: untested for GLMs?
    def wald_test(self, r_matrix, cov_p=None, invcov=None,
                  use_f=None, df_constraints=None, scalar=None):
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        f_test : Perform an F tests on model parameters.
        t_test : Perform a single hypothesis test.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        use_f = bool_like(use_f, "use_f", strict=True, optional=True)
        scalar = bool_like(scalar, "scalar", strict=True, optional=True)
        if use_f is None:
            # switch to use_t false if undefined
            use_f = (hasattr(self, 'use_t') and self.use_t)

        from patsy import DesignInfo
        names = self.model.data.cov_names
        params = self.params.ravel()
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        if (self.normalized_cov_params is None and cov_p is None and
                invcov is None and not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             'F statistics')

        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions

        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                                 "dimensions that are asymptotically "
                                 "non-normal")
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                warnings.warn('covariance of constraints does not have full '
                              'rank. The number of constraints is %d, but '
                              'rank is %d' % (J, J_), ValueWarning)
                J = J_

        # TODO streamline computation, we do not need to compute J if given
        if df_constraints is not None:
            # let caller override J by df_constraint
            J = df_constraints

        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if scalar is None:
            warnings.warn(
                "The behavior of wald_test will change after 0.14 to returning "
                "scalar test statistic values. To get the future behavior now, "
                "set scalar to True. To silence this message while retaining "
                "the legacy behavior, set scalar to False.",
                FutureWarning
            )
            scalar = False
        if scalar and F.size == 1:
            F = float(np.squeeze(F))
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid,
                                   df_num=J) #invcov.shape[0])
        else:
            return ContrastResults(chi2=F, df_denom=J, statistic=F,
                                   distribution='chi2', distargs=(J,))

    def wald_test_terms(self, skip_single=False, extra_constraints=None,
                        combine_terms=None, scalar=None):
        """
        Compute a sequence of Wald tests for terms over multiple columns.

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.
        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : bool
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            Additional constraints to test. Note that this input has not been
            tested.
        combine_terms : {list[str], None}
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        WaldTestResults
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.

        Examples
        --------
        >>> res_ols = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()
        >>> res_ols.wald_test_terms()
        <class 'statsmodels.stats.contrast.WaldTestResults'>
                                                  F                P>F  df constraint  df denom
        Intercept                        279.754525  2.37985521351e-22              1        51
        C(Duration, Sum)                   5.367071    0.0245738436636              1        51
        C(Weight, Sum)                    12.432445  3.99943118767e-05              2        51
        C(Duration, Sum):C(Weight, Sum)    0.176002      0.83912310946              2        51

        >>> res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)", \
                                           data).fit(cov_type='HC0')
        >>> wt = res_poi.wald_test_terms(skip_single=False, \
                                         combine_terms=['Duration', 'Weight'])
        >>> print(wt)
                                    chi2             P>chi2  df constraint
        Intercept              15.695625  7.43960374424e-05              1
        C(Weight)              16.132616  0.000313940174705              2
        C(Duration)             1.009147     0.315107378931              1
        C(Weight):C(Duration)   0.216694     0.897315972824              2
        Duration               11.187849     0.010752286833              3
        Weight                 30.263368  4.32586407145e-06              4
        """
        # lazy import
        from collections import defaultdict

        result = self
        if extra_constraints is None:
            extra_constraints = []
        if combine_terms is None:
            combine_terms = []
        design_info = getattr(result.model.data, 'design_info', None)

        if design_info is None and extra_constraints is None:
            raise ValueError('no constraints, nothing to do')

        identity = np.eye(len(result.params))
        constraints = []
        combined = defaultdict(list)
        if design_info is not None:
            for term in design_info.terms:
                cols = design_info.slice(term)
                name = term.name()
                constraint_matrix = identity[cols]

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                k_constraint = constraint_matrix.shape[0]
                if skip_single:
                    if k_constraint == 1:
                        continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))
        else:
            # check by exog/params names if there is no formula info
            for col, name in enumerate(result.model.exog_names):
                constraint_matrix = np.atleast_2d(identity[col])

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                if skip_single:
                    continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))

        use_t = result.use_t
        distribution = ['chi2', 'F'][use_t]

        res_wald = []
        index = []
        for name, constraint in constraints + combined_constraints + extra_constraints:
            wt = result.wald_test(constraint, scalar=scalar)
            row = [wt.statistic, wt.pvalue, constraint.shape[0]]
            if use_t:
                row.append(wt.df_denom)
            res_wald.append(row)
            index.append(name)

        # distribution nerutral names
        col_names = ['statistic', 'pvalue', 'df_constraint']
        if use_t:
            col_names.append('df_denom')
        # TODO: maybe move DataFrame creation to results class
        from pandas import DataFrame
        table = DataFrame(res_wald, index=index, columns=col_names)
        res = WaldTestResults(None, distribution, None, table=table)
        # TODO: remove temp again, added for testing
        res.temp = constraints + combined_constraints + extra_constraints
        return res

    def t_test_pairwise(self, term_name, method='hs', alpha=0.05,
                        factor_labels=None):
        """
        Perform pairwise t_test with multiple testing corrected p-values.

        This uses the formula design_info encoding contrast matrix and should
        work for all encodings of a main effect.

        Parameters
        ----------
        term_name : str
            The name of the term for which pairwise comparisons are computed.
            Term names for categorical effects are created by patsy and
            correspond to the main part of the exog names.
        method : {str, list[str]}
            The multiple testing p-value correction to apply. The default is
            'hs'. See stats.multipletesting.
        alpha : float
            The significance level for multiple testing reject decision.
        factor_labels : {list[str], None}
            Labels for the factor levels used for pairwise labels. If not
            provided, then the labels from the formula design_info are used.

        Returns
        -------
        MultiCompResult
            The results are stored as attributes, the main attributes are the
            following two. Other attributes are added for debugging purposes
            or as background information.

            - result_frame : pandas DataFrame with t_test results and multiple
              testing corrected p-values.
            - contrasts : matrix of constraints of the null hypothesis in the
              t_test.

        Notes
        -----
        Status: experimental. Currently only checked for treatment coding with
        and without specified reference level.

        Currently there are no multiple testing corrected confidence intervals
        available.

        Examples
        --------
        >>> res = ols("np.log(Days+1) ~ C(Weight) + C(Duration)", data).fit()
        >>> pw = res.t_test_pairwise("C(Weight)")
        >>> pw.result_frame
                 coef   std err         t         P>|t|  Conf. Int. Low
        2-1  0.632315  0.230003  2.749157  8.028083e-03        0.171563
        3-1  1.302555  0.230003  5.663201  5.331513e-07        0.841803
        3-2  0.670240  0.230003  2.914044  5.119126e-03        0.209488
             Conf. Int. Upp.  pvalue-hs reject-hs
        2-1         1.093067   0.010212      True
        3-1         1.763307   0.000002      True
        3-2         1.130992   0.010212      True
        """
        res = t_test_pairwise(self, term_name, method=method, alpha=alpha,
                              factor_labels=factor_labels)
        return res

    def _get_wald_nonlinear(self, func, deriv=None):
        """Experimental method for nonlinear prediction and tests

        Parameters
        ----------
        func : callable, f(params)
            nonlinear function of the estimation parameters. The return of
            the function can be vector valued, i.e. a 1-D array
        deriv : function or None
            first derivative or Jacobian of func. If deriv is None, then a
            numerical derivative will be used. If func returns a 1-D array,
            then the `deriv` should have rows corresponding to the elements
            of the return of func.

        Returns
        -------
        nl : instance of `NonlinearDeltaCov` with attributes and methods to
            calculate the results for the prediction or tests

        """
        from statsmodels.stats._delta_method import NonlinearDeltaCov
        func_args = None  # TODO: not yet implemented, maybe skip - use partial
        nl = NonlinearDeltaCov(func, self.params, self.cov_params(),
                               deriv=deriv, func_args=func_args)

        return nl

    def conf_int(self, alpha=.05, cols=None):
        """
        Construct confidence interval for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Specifies which confidence intervals to return.

        .. deprecated: 0.13

           cols is deprecated and will be removed after 0.14 is released.
           cols only works when inputs are NumPy arrays and will fail
           when using pandas Series or DataFrames as input. You can
           subset the confidence intervals using slices.

        Returns
        -------
        array_like
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.

        Notes
        -----
        The confidence interval is based on the standard normal distribution
        if self.use_t is False. If self.use_t is True, then uses a Student's t
        with self.df_resid_inference (or self.df_resid if df_resid_inference is
        not defined) degrees of freedom.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])

        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])
        """
        bse = self.bse

        if self.use_t:
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
            q = dist.ppf(1 - alpha / 2)

        params = self.params
        lower = params - q * bse
        upper = params + q * bse
        if cols is not None:
            warnings.warn(
                "cols is deprecated and will be removed after 0.14 is "
                "released. cols only works when inputs are NumPy arrays and "
                "will fail when using pandas Series or DataFrames as input. "
                "Subsets of confidence intervals can be selected using slices "
                "of the full confidence interval array.",
                FutureWarning
            )
            cols = np.asarray(cols)
            lower = lower[cols]
            upper = upper[cols]
        return np.asarray(lzip(lower, upper))

    def save(self, fname, remove_data=False):
        """
        Save a pickle of this instance.

        Parameters
        ----------
        fname : {str, handle}
            A string filename or a file handle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        Notes
        -----
        If remove_data is true and the model result does not implement a
        remove_data method then this will raise an exception.
        """

        from statsmodels.iolib.smpickle import save_pickle

        if remove_data:
            self.remove_data()

        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        """
        Load a pickled results instance

        .. warning::

           Loading pickled models is not secure against erroneous or
           maliciously constructed data. Never unpickle data received from
           an untrusted or unauthenticated source.

        Parameters
        ----------
        fname : {str, handle, pathlib.Path}
            A string filename or a file handle.

        Returns
        -------
        Results
            The unpickled results instance.
        """

        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)

    def remove_data(self):
        """
        Remove data arrays, all nobs arrays from result and model.

        This reduces the size of the instance, so it can be pickled with less
        memory. Currently tested for use with predict from an unpickled
        results and model instance.

        .. warning::

           Since data and some intermediate results have been removed
           calculating new statistics that require them will raise exceptions.
           The exception will occur the first time an attribute is accessed
           that has been set to None.

        Not fully tested for time series models, tsa, and might delete too much
        for prediction or not all that would be possible.

        The lists of arrays to delete are maintained as attributes of
        the result and model instance, except for cached values. These
        lists could be changed before calling remove_data.

        The attributes to remove are named in:

        model._data_attr : arrays attached to both the model instance
            and the results instance with the same attribute name.

        result._data_in_cache : arrays that may exist as values in
            result._cache

        result._data_attr_model : arrays attached to the model
            instance but not to the results instance
        """
        cls = self.__class__
        # Note: we cannot just use `getattr(cls, x)` or `getattr(self, x)`
        # because of redirection involved with property-like accessors
        cls_attrs = {}
        for name in dir(cls):
            try:
                attr = object.__getattribute__(cls, name)
            except AttributeError:
                pass
            else:
                cls_attrs[name] = attr
        data_attrs = [x for x in cls_attrs
                      if isinstance(cls_attrs[x], cached_data)]
        for name in data_attrs:
            self._cache[name] = None

        def wipe(obj, att):
            # get to last element in attribute path
            p = att.split('.')
            att_ = p.pop(-1)
            try:
                obj_ = reduce(getattr, [obj] + p)
                if hasattr(obj_, att_):
                    setattr(obj_, att_, None)
            except AttributeError:
                pass

        model_only = ['model.' + i for i in getattr(self, "_data_attr_model", [])]
        model_attr = ['model.' + i for i in self.model._data_attr]
        for att in self._data_attr + model_attr + model_only:
            if att in data_attrs:
                # these have been handled above, and trying to call wipe
                # would raise an Exception anyway, so skip these
                continue
            wipe(self, att)

        for key in self._data_in_cache:
            try:
                self._cache[key] = None
            except (AttributeError, KeyError):
                pass


class LikelihoodResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'bse': 'columns',
        'pvalues': 'columns',
        'tvalues': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
        'normalized_cov_params': 'cov',
    }

    _wrap_attrs = _attrs
    _wrap_methods = {
        'cov_params': 'cov',
        'conf_int': 'columns'
    }

wrap.populate_wrapper(LikelihoodResultsWrapper,  # noqa:E305
                      LikelihoodModelResults)


class ResultMixin:

    @cache_readonly
    def df_modelwc(self):
        """Model WC"""
        # collect different ways of defining the number of parameters, used for
        # aic, bic
        k_extra = getattr(self.model, "k_extra", 0)
        if hasattr(self, 'df_model'):
            if hasattr(self, 'k_constant'):
                hasconst = self.k_constant
            elif hasattr(self, 'hasconst'):
                hasconst = self.hasconst
            else:
                # default assumption
                hasconst = 1
            return self.df_model + hasconst + k_extra
        else:
            return self.params.size

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        return -2 * self.llf + 2 * (self.df_modelwc)

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        return -2 * self.llf + np.log(self.nobs) * (self.df_modelwc)

    @cache_readonly
    def score_obsv(self):
        """cached Jacobian of log-likelihood
        """
        return self.model.score_obs(self.params)

    @cache_readonly
    def hessv(self):
        """cached Hessian of log-likelihood
        """
        return self.model.hessian(self.params)

    @cache_readonly
    def covjac(self):
        """
        covariance of parameters based on outer product of jacobian of
        log-likelihood
        """
        #  if not hasattr(self, '_results'):
        #      raise ValueError('need to call fit first')
        #      #self.fit()
        #  self.jacv = jacv = self.jac(self._results.params)
        jacv = self.score_obsv
        return np.linalg.inv(np.dot(jacv.T, jacv))

    @cache_readonly
    def covjhj(self):
        """covariance of parameters based on HJJH

        dot product of Hessian, Jacobian, Jacobian, Hessian of likelihood

        name should be covhjh
        """
        jacv = self.score_obsv
        hessv = self.hessv
        hessinv = np.linalg.inv(hessv)
        #  self.hessinv = hessin = self.cov_params()
        return np.dot(hessinv, np.dot(np.dot(jacv.T, jacv), hessinv))

    @cache_readonly
    def bsejhj(self):
        """standard deviation of parameter estimates based on covHJH
        """
        return np.sqrt(np.diag(self.covjhj))

    @cache_readonly
    def bsejac(self):
        """standard deviation of parameter estimates based on covjac
        """
        return np.sqrt(np.diag(self.covjac))

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        """simple bootstrap to get mean and variance of estimator

        see notes

        Parameters
        ----------
        nrep : int
            number of bootstrap replications
        method : str
            optimization method to use
        disp : bool
            If true, then optimization prints results
        store : bool
            If true, then parameter estimates for all bootstrap iterations
            are attached in self.bootstrap_results

        Returns
        -------
        mean : ndarray
            mean of parameter estimates over bootstrap replications
        std : ndarray
            standard deviation of parameter estimates over bootstrap
            replications

        Notes
        -----
        This was mainly written to compare estimators of the standard errors of
        the parameter estimates.  It uses independent random sampling from the
        original endog and exog, and therefore is only correct if observations
        are independently distributed.

        This will be moved to apply only to models with independently
        distributed observations.
        """
        results = []
        hascloneattr = True if hasattr(self.model, 'cloneattr') else False
        for i in range(nrep):
            rvsind = np.random.randint(self.nobs, size=self.nobs)
            # this needs to set startparam and get other defining attributes
            # need a clone method on model
            if self.exog is not None:
                exog_resamp = self.exog[rvsind, :]
            else:
                exog_resamp = None
            # build auxiliary model and fit
            init_kwds = self.model._get_init_kwds()
            fitmod = self.model.__class__(self.endog[rvsind],
                                          exog=exog_resamp, **init_kwds)
            if hascloneattr:
                for attr in self.model.cloneattr:
                    setattr(fitmod, attr, getattr(self.model, attr))

            fitres = fitmod.fit(method=method, disp=disp)
            results.append(fitres.params)
        results = np.array(results)
        if store:
            self.bootstrap_results = results
        return results.mean(0), results.std(0), results

    def get_nlfun(self, fun):
        """
        get_nlfun

        This is not Implemented
        """
        # I think this is supposed to get the delta method that is currently
        # in miscmodels count (as part of Poisson example)
        raise NotImplementedError


class _LLRMixin():
    """Mixin class for Null model and likelihood ratio
    """
    # methods copied from DiscreteResults, adjusted pseudo R2

    def pseudo_rsquared(self, kind="mcf"):
        """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
        kind = kind.lower()
        if kind.startswith("mcf"):
            prsq = 1 - self.llf / self.llnull
        elif kind.startswith("cox") or kind in ["cs", "lr"]:
            prsq = 1 - np.exp((self.llnull - self.llf) * (2 / self.nobs))
        else:
            raise ValueError("only McFadden and Cox-Snell are available")
        return prsq

    @cache_readonly
    def llr(self):
        """
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
        """
        return -2*(self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        """
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
        """
        # see also RegressionModel compare_lr_test
        llr = self.llr
        df_full = self.df_resid
        df_restr = self.df_resid_null
        lrdf = (df_restr - df_full)
        self.df_lr_null = lrdf
        return stats.distributions.chi2.sf(llr, lrdf)

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        """
        Set the fit options for the Null (constant-only) model.

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : {None, float}
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        **kwargs
            Additional keyword arguments used as fit keyword arguments for the
            null model. The override and model default values.

        Notes
        -----
        Modifies attributes of this instance, and so has no return.
        """
        # reset cache, note we need to add here anything that depends on
        # llnullor the null model. If something is missing, then the attribute
        # might be incorrect.
        self._cache.pop('llnull', None)
        self._cache.pop('llr', None)
        self._cache.pop('llr_pvalue', None)
        self._cache.pop('prsquared', None)
        if hasattr(self, 'res_null'):
            del self.res_null

        if llnull is not None:
            self._cache['llnull'] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwargs

    @cache_readonly
    def llnull(self):
        """
        Value of the constant-only loglikelihood
        """
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, '_null_drop_keys', []):
            del kwds[key]
        # TODO: what parameters to pass to fit?
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        # TODO: consider catching and warning on convergence failure?
        # in the meantime, try hard to converge. see
        # TestPoissonConstrained1a.test_smoke

        optim_kwds = getattr(self, '_optim_kwds_null', {}).copy()

        if 'start_params' in optim_kwds:
            # user provided
            sp_null = optim_kwds.pop('start_params')
        elif hasattr(model, '_get_start_params_null'):
            # get moment estimates if available
            sp_null = model._get_start_params_null()
        else:
            sp_null = None

        opt_kwds = dict(method='bfgs', warn_convergence=False, maxiter=10000,
                        disp=0)
        opt_kwds.update(optim_kwds)

        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            # this should be a reasonably method case across versions
            res_null = mod_null.fit(start_params=sp_null, method='nm',
                                    warn_convergence=False,
                                    maxiter=10000, disp=0)
            res_null = mod_null.fit(start_params=res_null.params, method='bfgs',
                                    warn_convergence=False,
                                    maxiter=10000, disp=0)

        if getattr(self, '_attach_nullmodel', False) is not False:
            self.res_null = res_null

        self.k_null = len(res_null.params)
        self.df_resid_null = res_null.df_resid
        return res_null.llf


class GenericLikelihoodModelResults(LikelihoodModelResults, ResultMixin):
    """
    A results class for the discrete dependent variable models.

    ..Warning :

    The following description has not been updated to this version/class.
    Where are AIC, BIC, ....? docstring looks like copy from discretemod

    Parameters
    ----------
    model : A DiscreteModel instance
    mlefit : instance of LikelihoodResults
        This contains the numerical optimization results as returned by
        LikelihoodModel.fit(), in a superclass of GnericLikelihoodModels


    Attributes
    ----------
    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : ndarray
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : ndarray
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)
    """

    def __init__(self, model, mlefit):
        self.model = model
        self.endog = model.endog
        self.exog = model.exog
        self.nobs = model.endog.shape[0]

        # TODO: possibly move to model.fit()
        #       and outsource together with patching names
        k_extra = getattr(self.model, "k_extra", 0)
        if hasattr(model, 'df_model') and not np.isnan(model.df_model):
            self.df_model = model.df_model
        else:
            df_model = len(mlefit.params) - self.model.k_constant - k_extra
            self.df_model = df_model
            # retrofitting the model, used in t_test TODO: check design
            self.model.df_model = df_model

        if hasattr(model, 'df_resid') and not np.isnan(model.df_resid):
            self.df_resid = model.df_resid
        else:
            self.df_resid = self.endog.shape[0] - self.df_model - k_extra
            # retrofitting the model, used in t_test TODO: check design
            self.model.df_resid = self.df_resid

        self._cache = {}
        self.__dict__.update(mlefit.__dict__)

        k_params = len(mlefit.params)
        # checks mainly for adding new models or subclassing

        if self.df_model + self.model.k_constant + k_extra != k_params:
            warnings.warn("df_model + k_constant + k_extra "
                          "differs from k_params")

        if self.df_resid != self.nobs - k_params:
            warnings.warn("df_resid differs from nobs - k_params")

    def get_prediction(
            self,
            exog=None,
            which="mean",
            transform=True,
            row_labels=None,
            average=False,
            agg_weights=None,
            **kwargs
            ):
        """
        Compute prediction results when endpoint transformation is valid.

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        which : str
            Which statistic is to be predicted. Default is "mean".
            The available statistics and options depend on the model.
            see the model.predict docstring
        row_labels : list of str or None
            If row_lables are provided, then they will replace the generated
            labels.
        average : bool
            If average is True, then the mean prediction is computed, that is,
            predictions are computed for individual exog and then the average
            over observation is used.
            If average is False, then the results are the predictions for all
            observations, i.e. same length as ``exog``.
        agg_weights : ndarray, optional
            Aggregation weights, only used if average is True.
            The weights are not normalized.
        **kwargs :
            Some models can take additional keyword arguments, such as offset,
            exposure or additional exog in multi-part models like zero inflated
            models.
            See the predict method of the model for the details.

        Returns
        -------
        prediction_results : PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary dataframe for the prediction.

        Notes
        -----
        Status: new in 0.14, experimental
        """
        from statsmodels.base._prediction_inference import get_prediction

        pred_kwds = kwargs

        res = get_prediction(
            self,
            exog=exog,
            which=which,
            transform=transform,
            row_labels=row_labels,
            average=average,
            agg_weights=agg_weights,
            pred_kwds=pred_kwds
            )
        return res

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
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
        statsmodels.iolib.summary.Summary : class to hold summary results
        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Maximum Likelihood']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None),
                    ('Df Model:', None),
                    ]

        top_right = [('Log-Likelihood:', None),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
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
