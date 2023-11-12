#Splitting out maringal effects to see if they can be generalized

from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly

#### margeff helper functions ####
#NOTE: todo marginal effects for group 2
# group 2 oprobit, ologit, gologit, mlogit, biprobit

def _check_margeff_args(at, method):
    """
    Checks valid options for margeff
    """
    if at not in ['overall','mean','median','zero','all']:
        raise ValueError("%s not a valid option for `at`." % at)
    if method not in ['dydx','eyex','dyex','eydx']:
        raise ValueError("method is not understood.  Got %s" % method)

def _check_discrete_args(at, method):
    """
    Checks the arguments for margeff if the exogenous variables are discrete.
    """
    if method in ['dyex','eyex']:
        raise ValueError("%s not allowed for discrete variables" % method)
    if at in ['median', 'zero']:
        raise ValueError("%s not allowed for discrete variables" % at)

def _get_const_index(exog):
    """
    Returns a boolean array of non-constant column indices in exog and
    an scalar array of where the constant is or None
    """
    effects_idx = exog.var(0) != 0
    if np.any(~effects_idx):
        const_idx = np.where(~effects_idx)[0]
    else:
        const_idx = None
    return effects_idx, const_idx

def _isdummy(X):
    """
    Given an array X, returns the column indices for the dummy variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _isdummy(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    if X.ndim > 1:
        ind = np.zeros(X.shape[1]).astype(bool)
    max = (np.max(X, axis=0) == 1)
    min = (np.min(X, axis=0) == 0)
    remainder = np.all(X % 1. == 0, axis=0)
    ind = min & max & remainder
    if X.ndim == 1:
        ind = np.asarray([ind])
    return np.where(ind)[0]

def _get_dummy_index(X, const_idx):
    dummy_ind = _isdummy(X)
    dummy = True

    if dummy_ind.size == 0: # do not waste your time
        dummy = False
        dummy_ind = None # this gets passed to stand err func
    return dummy_ind, dummy

def _iscount(X):
    """
    Given an array X, returns the column indices for count variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 10, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _iscount(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    remainder = np.logical_and(np.logical_and(np.all(X % 1. == 0, axis = 0),
                               X.var(0) != 0), np.all(X >= 0, axis=0))
    dummy = _isdummy(X)
    remainder = np.where(remainder)[0].tolist()
    for idx in dummy:
        remainder.remove(idx)
    return np.array(remainder)

def _get_count_index(X, const_idx):
    count_ind = _iscount(X)
    count = True

    if count_ind.size == 0: # do not waste your time
        count = False
        count_ind = None # for stand err func
    return count_ind, count

def _get_margeff_exog(exog, at, atexog, ind):
    if atexog is not None: # user supplied
        if isinstance(atexog, dict):
            # assumes values are singular or of len(exog)
            for key in atexog:
                exog[:,key] = atexog[key]
        elif isinstance(atexog, np.ndarray): #TODO: handle DataFrames
            if atexog.ndim == 1:
                k_vars = len(atexog)
            else:
                k_vars = atexog.shape[1]
            try:
                assert k_vars == exog.shape[1]
            except:
                raise ValueError("atexog does not have the same number "
                        "of variables as exog")
            exog = atexog

    #NOTE: we should fill in atexog after we process at
    if at == 'mean':
        exog = np.atleast_2d(exog.mean(0))
    elif at == 'median':
        exog = np.atleast_2d(np.median(exog, axis=0))
    elif at == 'zero':
        exog = np.zeros((1,exog.shape[1]))
        exog[0,~ind] = 1
    return exog

def _get_count_effects(effects, exog, count_ind, method, model, params):
    """
    If there's a count variable, the predicted difference is taken by
    subtracting one and adding one to exog then averaging the difference
    """
    # this is the index for the effect and the index for count col in exog
    for i in count_ind:
        exog0 = exog.copy()
        exog0[:, i] -= 1
        effect0 = model.predict(params, exog0)
        exog0[:, i] += 2
        effect1 = model.predict(params, exog0)
        #NOTE: done by analogy with dummy effects but untested bc
        # stata does not handle both count and eydx anywhere
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[:, i] = ((effect1 - effect0)/2)
    return effects

def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    If there's a dummy variable, the predicted difference is taken at
    0 and 1
    """
    # this is the index for the effect and the index for dummy col in exog
    for i in dummy_ind:
        exog0 = exog.copy() # only copy once, can we avoid a copy?
        exog0[:,i] = 0
        effect0 = model.predict(params, exog0)
        #fittedvalues0 = np.dot(exog0,params)
        exog0[:,i] = 1
        effect1 = model.predict(params, exog0)
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[:, i] = (effect1 - effect0)
    return effects

def _effects_at(effects, at):
    if at == 'all':
        effects = effects
    elif at == 'overall':
        effects = effects.mean(0)
    else:
        effects = effects[0,:]
    return effects

def _margeff_cov_params_dummy(model, cov_margins, params, exog, dummy_ind,
        method, J):
    r"""
    Returns the Jacobian for discrete regressors for use in margeff_cov_params.

    For discrete regressors the marginal effect is

    \Delta F = F(XB) | d = 1 - F(XB) | d = 0

    The row of the Jacobian for this variable is given by

    f(XB)*X | d = 1 - f(XB)*X | d = 0

    Where F is the default prediction of the model.
    """
    for i in dummy_ind:
        exog0 = exog.copy()
        exog1 = exog.copy()
        exog0[:,i] = 0
        exog1[:,i] = 1
        dfdb0 = model._derivative_predict(params, exog0, method)
        dfdb1 = model._derivative_predict(params, exog1, method)
        dfdb = (dfdb1 - dfdb0)
        if dfdb.ndim >= 2: # for overall
            dfdb = dfdb.mean(0)
        if J > 1:
            K = dfdb.shape[1] // (J-1)
            cov_margins[i::K, :] = dfdb
        else:
            # dfdb could be too short if there are extra params, k_extra > 0
            cov_margins[i, :len(dfdb)] = dfdb # how each F changes with change in B
    return cov_margins

def _margeff_cov_params_count(model, cov_margins, params, exog, count_ind,
                             method, J):
    r"""
    Returns the Jacobian for discrete regressors for use in margeff_cov_params.

    For discrete regressors the marginal effect is

    \Delta F = F(XB) | d += 1 - F(XB) | d -= 1

    The row of the Jacobian for this variable is given by

    (f(XB)*X | d += 1 - f(XB)*X | d -= 1) / 2

    where F is the default prediction for the model.
    """
    for i in count_ind:
        exog0 = exog.copy()
        exog0[:,i] -= 1
        dfdb0 = model._derivative_predict(params, exog0, method)
        exog0[:,i] += 2
        dfdb1 = model._derivative_predict(params, exog0, method)
        dfdb = (dfdb1 - dfdb0)
        if dfdb.ndim >= 2: # for overall
            dfdb = dfdb.mean(0) / 2
        if J > 1:
            K = dfdb.shape[1] / (J-1)
            cov_margins[i::K, :] = dfdb
        else:
            # dfdb could be too short if there are extra params, k_extra > 0
            cov_margins[i, :len(dfdb)] = dfdb # how each F changes with change in B
    return cov_margins

def margeff_cov_params(model, params, exog, cov_params, at, derivative,
                       dummy_ind, count_ind, method, J):
    """
    Computes the variance-covariance of marginal effects by the delta method.

    Parameters
    ----------
    model : model instance
        The model that returned the fitted results. Its pdf method is used
        for computing the Jacobian of discrete variables in dummy_ind and
        count_ind
    params : array_like
        estimated model parameters
    exog : array_like
        exogenous variables at which to calculate the derivative
    cov_params : array_like
        The variance-covariance of the parameters
    at : str
       Options are:

        - 'overall', The average of the marginal effects at each
          observation.
        - 'mean', The marginal effects at the mean of each regressor.
        - 'median', The marginal effects at the median of each regressor.
        - 'zero', The marginal effects at zero for each regressor.
        - 'all', The marginal effects at each observation.

        Only overall has any effect here.you

    derivative : function or array_like
        If a function, it returns the marginal effects of the model with
        respect to the exogenous variables evaluated at exog. Expected to be
        called derivative(params, exog). This will be numerically
        differentiated. Otherwise, it can be the Jacobian of the marginal
        effects with respect to the parameters.
    dummy_ind : array_like
        Indices of the columns of exog that contain dummy variables
    count_ind : array_like
        Indices of the columns of exog that contain count variables

    Notes
    -----
    For continuous regressors, the variance-covariance is given by

    Asy. Var[MargEff] = [d margeff / d params] V [d margeff / d params]'

    where V is the parameter variance-covariance.

    The outer Jacobians are computed via numerical differentiation if
    derivative is a function.
    """
    if callable(derivative):
        from statsmodels.tools.numdiff import approx_fprime_cs
        params = params.ravel('F')  # for Multinomial
        try:
            jacobian_mat = approx_fprime_cs(params, derivative,
                                            args=(exog,method))
        except TypeError:  # norm.cdf does not take complex values
            from statsmodels.tools.numdiff import approx_fprime
            jacobian_mat = approx_fprime(params, derivative,
                                            args=(exog,method))
        if at == 'overall':
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        else:
            jacobian_mat = jacobian_mat.squeeze()  # exog was 2d row vector
        if dummy_ind is not None:
            jacobian_mat = _margeff_cov_params_dummy(model, jacobian_mat,
                                params, exog, dummy_ind, method, J)
        if count_ind is not None:
            jacobian_mat = _margeff_cov_params_count(model, jacobian_mat,
                                params, exog, count_ind, method, J)
    else:
        jacobian_mat = derivative

    #NOTE: this will not go through for at == 'all'
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)

def margeff_cov_with_se(model, params, exog, cov_params, at, derivative,
                        dummy_ind, count_ind, method, J):
    """
    See margeff_cov_params.

    Same function but returns both the covariance of the marginal effects
    and their standard errors.
    """
    cov_me = margeff_cov_params(model, params, exog, cov_params, at,
                                              derivative, dummy_ind,
                                              count_ind, method, J)
    return cov_me, np.sqrt(np.diag(cov_me))


def margeff():
    raise NotImplementedError



def _check_at_is_all(method):
    if method['at'] == 'all':
        raise ValueError("Only margeff are available when `at` is "
                         "'all'. Please input specific points if you would "
                         "like to do inference.")


_transform_names = dict(dydx='dy/dx',
                        eyex='d(lny)/d(lnx)',
                        dyex='dy/d(lnx)',
                        eydx='d(lny)/dx')

class Margins:
    """
    Mostly a do nothing class. Lays out the methods expected of a sub-class.

    This is just a sketch of what we may want out of a general margins class.
    I (SS) need to look at details of other models.
    """
    def __init__(self, results, get_margeff, derivative, dist=None,
                       margeff_args=()):
        self._cache = {}
        self.results = results
        self.dist = dist
        self.get_margeff(margeff_args)

    def _reset(self):
        self._cache = {}

    def get_margeff(self, *args, **kwargs):
        self._reset()
        self.margeff = self.get_margeff(*args)

    @cache_readonly
    def tvalues(self):
        raise NotImplementedError

    @cache_readonly
    def cov_margins(self):
        raise NotImplementedError

    @cache_readonly
    def margins_se(self):
        raise NotImplementedError

    def summary_frame(self):
        raise NotImplementedError

    @cache_readonly
    def pvalues(self):
        raise NotImplementedError

    def conf_int(self, alpha=.05):
        raise NotImplementedError

    def summary(self, alpha=.05):
        raise NotImplementedError

#class DiscreteMargins(Margins):
class DiscreteMargins:
    """Get marginal effects of a Discrete Choice model.

    Parameters
    ----------
    results : DiscreteResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """
    def __init__(self, results, args, kwargs={}):
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)

    def _reset(self):
        self._cache = {}

    @cache_readonly
    def tvalues(self):
        _check_at_is_all(self.margeff_options)
        return self.margeff / self.margeff_se

    def summary_frame(self, alpha=.05):
        """
        Returns a DataFrame summarizing the marginal effects.

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        frame : DataFrames
            A DataFrame summarizing the marginal effects.

        Notes
        -----
        The dataframe is created on each call and not cached, as are the
        tables build in `summary()`
        """
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = self.results.model
        from pandas import DataFrame, MultiIndex
        names = [_transform_names[self.margeff_options['method']],
                                  'Std. Err.', 'z', 'Pr(>|z|)',
                                  'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0 # True if not a constant
        exog_names = self.results.model.exog_names
        k_extra = getattr(model, 'k_extra', 0)
        if k_extra > 0:
            exog_names = exog_names[:-k_extra]
        var_names = [name for i,name in enumerate(exog_names) if ind[i]]

        if self.margeff.ndim == 2:
            # MNLogit case
            ci = self.conf_int(alpha)
            table = np.column_stack([i.ravel("F") for i in
                        [self.margeff, self.margeff_se, self.tvalues,
                         self.pvalues, ci[:, 0, :], ci[:, 1, :]]])

            _, yname_list = results._get_endog_name(model.endog_names,
                                                        None, all=True)
            ynames = np.repeat(yname_list, len(var_names))
            xnames = np.tile(var_names, len(yname_list))
            index = MultiIndex.from_tuples(list(zip(ynames, xnames)),
                                           names=['endog', 'exog'])
        else:
            table = np.column_stack((self.margeff, self.margeff_se, self.tvalues,
                                     self.pvalues, self.conf_int(alpha)))
            index=var_names

        return DataFrame(table, columns=names, index=index)


    @cache_readonly
    def pvalues(self):
        _check_at_is_all(self.margeff_options)
        return norm.sf(np.abs(self.tvalues)) * 2

    def conf_int(self, alpha=.05):
        """
        Returns the confidence intervals of the marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        conf_int : ndarray
            An array with lower, upper confidence intervals for the marginal
            effects.
        """
        _check_at_is_all(self.margeff_options)
        me_se = self.margeff_se
        q = norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(lzip(lower, upper))

    def summary(self, alpha=.05):
        """
        Returns a summary table for marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        Summary : SummaryTable
            A SummaryTable instance
        """
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = results.model
        title = model.__class__.__name__ + " Marginal Effects"
        method = self.margeff_options['method']
        top_left = [('Dep. Variable:', [model.endog_names]),
                ('Method:', [method]),
                ('At:', [self.margeff_options['at']]),]

        from statsmodels.iolib.summary import (Summary, summary_params,
                                                table_extend)
        exog_names = model.exog_names[:] # copy
        smry = Summary()

        # TODO: sigh, we really need to hold on to this in _data...
        _, const_idx = _get_const_index(model.exog)
        if const_idx is not None:
            exog_names.pop(const_idx[0])
        if getattr(model, 'k_extra', 0) > 0:
            exog_names = exog_names[:-model.k_extra]

        J = int(getattr(model, "J", 1))
        if J > 1:
            yname, yname_list = results._get_endog_name(model.endog_names,
                                                None, all=True)
        else:
            yname = model.endog_names
            yname_list = [yname]

        smry.add_table_2cols(self, gleft=top_left, gright=[],
                yname=yname, xname=exog_names, title=title)

        # NOTE: add_table_params is not general enough yet for margeff
        # could use a refactor with getattr instead of hard-coded params
        # tvalues etc.
        table = []
        conf_int = self.conf_int(alpha)
        margeff = self.margeff
        margeff_se = self.margeff_se
        tvalues = self.tvalues
        pvalues = self.pvalues
        if J > 1:
            for eq in range(J):
                restup = (results, margeff[:,eq], margeff_se[:,eq],
                          tvalues[:,eq], pvalues[:,eq], conf_int[:,:,eq])
                tble = summary_params(restup, yname=yname_list[eq],
                              xname=exog_names, alpha=alpha, use_t=False,
                              skip_header=True)
                tble.title = yname_list[eq]
                # overwrite coef with method name
                header = ['', _transform_names[method], 'std err', 'z',
                        'P>|z|', '[' + str(alpha/2), str(1-alpha/2) + ']']
                tble.insert_header_row(0, header)
                table.append(tble)

            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names,
                    alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z',
                        'P>|z|', '[' + str(alpha/2), str(1-alpha/2) + ']']
            table.insert_header_row(0, header)

        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                          dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        self._reset() # always reset the cache when this is called
        #TODO: if at is not all or overall, we can also put atexog values
        # in summary table head
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy() # copy because values are changed
        effects_idx, const_idx =  _get_const_index(exog)

        if dummy:
            _check_discrete_args(at, method)
            dummy_idx, dummy = _get_dummy_index(exog, const_idx)
        else:
            dummy_idx = None

        if count:
            _check_discrete_args(at, method)
            count_idx, count = _get_count_index(exog, const_idx)
        else:
            count_idx = None

        # attach dummy_idx and cout_idx
        self.dummy_idx = dummy_idx
        self.count_idx = count_idx

        # get the exogenous variables
        exog = _get_margeff_exog(exog, at, atexog, effects_idx)

        # get base marginal effects, handled by sub-classes
        effects = model._derivative_exog(params, exog, method,
                                                    dummy_idx, count_idx)

        J = getattr(model, 'J', 1)
        effects_idx = np.tile(effects_idx, J) # adjust for multi-equation.

        effects = _effects_at(effects, at)

        if at == 'all':
            if J > 1:
                K = model.K - np.any(~effects_idx) # subtract constant
                self.margeff = effects[:, effects_idx].reshape(-1, K, J,
                                                                order='F')
            else:
                self.margeff = effects[:, effects_idx]
        else:
            # Set standard error of the marginal effects by Delta method.
            margeff_cov, margeff_se = margeff_cov_with_se(model, params, exog,
                                                results.cov_params(), at,
                                                model._derivative_exog,
                                                dummy_idx, count_idx,
                                                method, J)

            # reshape for multi-equation
            if J > 1:
                K = model.K - np.any(~effects_idx) # subtract constant
                self.margeff = effects[effects_idx].reshape(K, J, order='F')
                self.margeff_se = margeff_se[effects_idx].reshape(K, J,
                                                                  order='F')
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            else:
                # do not care about at constant
                # hack truncate effects_idx again if necessary
                # if eyex, then effects is truncated to be without extra params
                effects_idx = effects_idx[:len(effects)]
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
                self.margeff_se = margeff_se[effects_idx]
                self.margeff = effects[effects_idx]
