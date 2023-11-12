from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
    _calc_nodewise_weight, _calc_approx_inv_cov
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np

"""
Distributed estimation routines. Currently, we support several
methods of distribution

- sequential, has no extra dependencies
- parallel
    - with joblib
        A variety of backends are supported through joblib
        This allows for different types of clusters besides
        standard local clusters.  Some examples of
        backends supported by joblib are
          - dask.distributed
          - yarn
          - ipyparallel

The framework is very general and allows for a variety of
estimation methods.  Currently, these include

- debiased regularized estimation
- simple coefficient averaging (naive)
    - regularized
    - unregularized

Currently, the default is regularized estimation with debiasing
which follows the methods outlined in

Jason D. Lee, Qiang Liu, Yuekai Sun and Jonathan E. Taylor.
"Communication-Efficient Sparse Regression: A One-Shot Approach."
arXiv:1503.04337. 2015. https://arxiv.org/abs/1503.04337.

There are several variables that are taken from the source paper
for which the interpretation may not be directly clear from the
code, these are mostly used to help form the estimate of the
approximate inverse covariance matrix as part of the
debiasing procedure.

    wexog

    A weighted design matrix used to perform the node-wise
    regression procedure.

    nodewise_row

    nodewise_row is produced as part of the node-wise regression
    procedure used to produce the approximate inverse covariance
    matrix.  One is produced for each variable using the
    LASSO.

    nodewise_weight

    nodewise_weight is produced using the gamma_hat values for
    each p to produce weights to reweight the gamma_hat values which
    are ultimately used to form approx_inv_cov.

    approx_inv_cov

    This is the estimate of the approximate inverse covariance
    matrix.  This is used to debiase the coefficient average
    along with the average gradient.  For the OLS case,
    approx_inv_cov is an approximation for

        n * (X^T X)^{-1}

    formed by node-wise regression.
"""


def _est_regularized_naive(mod, pnum, partitions, fit_kwds=None):
    """estimates the regularized fitted parameters.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    pnum : scalar
        Index of current partition
    partitions : scalar
        Total number of partitions
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized

    Returns
    -------
    An array of the parameters for the regularized fit
    """

    if fit_kwds is None:
        raise ValueError("_est_regularized_naive currently " +
                         "requires that fit_kwds not be None.")

    return mod.fit_regularized(**fit_kwds).params


def _est_unregularized_naive(mod, pnum, partitions, fit_kwds=None):
    """estimates the unregularized fitted parameters.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    pnum : scalar
        Index of current partition
    partitions : scalar
        Total number of partitions
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit

    Returns
    -------
    An array of the parameters for the fit
    """

    if fit_kwds is None:
        raise ValueError("_est_unregularized_naive currently " +
                         "requires that fit_kwds not be None.")

    return mod.fit(**fit_kwds).params


def _join_naive(params_l, threshold=0):
    """joins the results from each run of _est_<type>_naive
    and returns the mean estimate of the coefficients

    Parameters
    ----------
    params_l : list
        A list of arrays of coefficients.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """

    p = len(params_l[0])
    partitions = len(params_l)

    params_mn = np.zeros(p)
    for params in params_l:
        params_mn += params
    params_mn /= partitions

    params_mn[np.abs(params_mn) < threshold] = 0

    return params_mn


def _calc_grad(mod, params, alpha, L1_wt, score_kwds):
    """calculates the log-likelihood gradient for the debiasing

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    score_kwds : dict-like or None
        Keyword arguments for the score function.

    Returns
    -------
    An array-like object of the same dimension as params

    Notes
    -----
    In general:

    gradient l_k(params)

    where k corresponds to the index of the partition

    For OLS:

    X^T(y - X^T params)
    """

    grad = -mod.score(np.asarray(params), **score_kwds)
    grad += alpha * (1 - L1_wt)
    return grad


def _calc_wdesign_mat(mod, params, hess_kwds):
    """calculates the weighted design matrix necessary to generate
    the approximate inverse covariance matrix

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    hess_kwds : dict-like or None
        Keyword arguments for the hessian function.

    Returns
    -------
    An array-like object, updated design matrix, same dimension
    as mod.exog
    """

    rhess = np.sqrt(mod.hessian_factor(np.asarray(params), **hess_kwds))
    return rhess[:, None] * mod.exog


def _est_regularized_debiased(mod, mnum, partitions, fit_kwds=None,
                              score_kwds=None, hess_kwds=None):
    """estimates the regularized fitted parameters, is the default
    estimation_method for class DistributedModel.

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    mnum : scalar
        Index of current partition.
    partitions : scalar
        Total number of partitions.
    fit_kwds : dict-like or None
        Keyword arguments to be given to fit_regularized
    score_kwds : dict-like or None
        Keyword arguments for the score function.
    hess_kwds : dict-like or None
        Keyword arguments for the Hessian function.

    Returns
    -------
    A tuple of parameters for regularized fit
        An array-like object of the fitted parameters, params
        An array-like object for the gradient
        A list of array like objects for nodewise_row
        A list of array like objects for nodewise_weight
    """

    score_kwds = {} if score_kwds is None else score_kwds
    hess_kwds = {} if hess_kwds is None else hess_kwds

    if fit_kwds is None:
        raise ValueError("_est_regularized_debiased currently " +
                         "requires that fit_kwds not be None.")
    else:
        alpha = fit_kwds["alpha"]

    if "L1_wt" in fit_kwds:
        L1_wt = fit_kwds["L1_wt"]
    else:
        L1_wt = 1

    nobs, p = mod.exog.shape
    p_part = int(np.ceil((1. * p) / partitions))

    params = mod.fit_regularized(**fit_kwds).params
    grad = _calc_grad(mod, params, alpha, L1_wt, score_kwds) / nobs

    wexog = _calc_wdesign_mat(mod, params, hess_kwds)

    nodewise_row_l = []
    nodewise_weight_l = []
    for idx in range(mnum * p_part, min((mnum + 1) * p_part, p)):

        nodewise_row = _calc_nodewise_row(wexog, idx, alpha)
        nodewise_row_l.append(nodewise_row)

        nodewise_weight = _calc_nodewise_weight(wexog, nodewise_row, idx,
                                                alpha)
        nodewise_weight_l.append(nodewise_weight)

    return params, grad, nodewise_row_l, nodewise_weight_l


def _join_debiased(results_l, threshold=0):
    """joins the results from each run of _est_regularized_debiased
    and returns the debiased estimate of the coefficients

    Parameters
    ----------
    results_l : list
        A list of tuples each one containing the params, grad,
        nodewise_row and nodewise_weight values for each partition.
    threshold : scalar
        The threshold at which the coefficients will be cut.
    """

    p = len(results_l[0][0])
    partitions = len(results_l)

    params_mn = np.zeros(p)
    grad_mn = np.zeros(p)

    nodewise_row_l = []
    nodewise_weight_l = []

    for r in results_l:

        params_mn += r[0]
        grad_mn += r[1]

        nodewise_row_l.extend(r[2])
        nodewise_weight_l.extend(r[3])

    nodewise_row_l = np.array(nodewise_row_l)
    nodewise_weight_l = np.array(nodewise_weight_l)

    params_mn /= partitions
    grad_mn *= -1. / partitions

    approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)

    debiased_params = params_mn + approx_inv_cov.dot(grad_mn)

    debiased_params[np.abs(debiased_params) < threshold] = 0

    return debiased_params


def _helper_fit_partition(self, pnum, endog, exog, fit_kwds,
                          init_kwds_e={}):
    """handles the model fitting for each machine. NOTE: this
    is primarily handled outside of DistributedModel because
    joblib cannot handle class methods.

    Parameters
    ----------
    self : DistributedModel class instance
        An instance of DistributedModel.
    pnum : scalar
        index of current partition.
    endog : array_like
        endogenous data for current partition.
    exog : array_like
        exogenous data for current partition.
    fit_kwds : dict-like
        Keywords needed for the model fitting.
    init_kwds_e : dict-like
        Additional init_kwds to add for each partition.

    Returns
    -------
    estimation_method result.  For the default,
    _est_regularized_debiased, a tuple.
    """

    temp_init_kwds = self.init_kwds.copy()
    temp_init_kwds.update(init_kwds_e)

    model = self.model_class(endog, exog, **temp_init_kwds)
    results = self.estimation_method(model, pnum, self.partitions,
                                     fit_kwds=fit_kwds,
                                     **self.estimation_kwds)
    return results


class DistributedModel:
    __doc__ = """
    Distributed model class

    Parameters
    ----------
    partitions : scalar
        The number of partitions that the data will be split into.
    model_class : statsmodels model class
        The model class which will be used for estimation. If None
        this defaults to OLS.
    init_kwds : dict-like or None
        Keywords needed for initializing the model, in addition to
        endog and exog.
    init_kwds_generator : generator or None
        Additional keyword generator that produces model init_kwds
        that may vary based on data partition.  The current usecase
        is for WLS and GLS
    estimation_method : function or None
        The method that performs the estimation for each partition.
        If None this defaults to _est_regularized_debiased.
    estimation_kwds : dict-like or None
        Keywords to be passed to estimation_method.
    join_method : function or None
        The method used to recombine the results from each partition.
        If None this defaults to _join_debiased.
    join_kwds : dict-like or None
        Keywords to be passed to join_method.
    results_class : results class or None
        The class of results that should be returned.  If None this
        defaults to RegularizedResults.
    results_kwds : dict-like or None
        Keywords to be passed to results class.

    Attributes
    ----------
    partitions : scalar
        See Parameters.
    model_class : statsmodels model class
        See Parameters.
    init_kwds : dict-like
        See Parameters.
    init_kwds_generator : generator or None
        See Parameters.
    estimation_method : function
        See Parameters.
    estimation_kwds : dict-like
        See Parameters.
    join_method : function
        See Parameters.
    join_kwds : dict-like
        See Parameters.
    results_class : results class
        See Parameters.
    results_kwds : dict-like
        See Parameters.

    Notes
    -----

    Examples
    --------
    """

    def __init__(self, partitions, model_class=None,
                 init_kwds=None, estimation_method=None,
                 estimation_kwds=None, join_method=None, join_kwds=None,
                 results_class=None, results_kwds=None):

        self.partitions = partitions

        if model_class is None:
            self.model_class = OLS
        else:
            self.model_class = model_class

        if init_kwds is None:
            self.init_kwds = {}
        else:
            self.init_kwds = init_kwds

        if estimation_method is None:
            self.estimation_method = _est_regularized_debiased
        else:
            self.estimation_method = estimation_method

        if estimation_kwds is None:
            self.estimation_kwds = {}
        else:
            self.estimation_kwds = estimation_kwds

        if join_method is None:
            self.join_method = _join_debiased
        else:
            self.join_method = join_method

        if join_kwds is None:
            self.join_kwds = {}
        else:
            self.join_kwds = join_kwds

        if results_class is None:
            self.results_class = RegularizedResults
        else:
            self.results_class = results_class

        if results_kwds is None:
            self.results_kwds = {}
        else:
            self.results_kwds = results_kwds

    def fit(self, data_generator, fit_kwds=None, parallel_method="sequential",
            parallel_backend=None, init_kwds_generator=None):
        """Performs the distributed estimation using the corresponding
        DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like or None
            Keywords needed for the model fitting.
        parallel_method : str
            type of distributed estimation to be used, currently
            "sequential", "joblib" and "dask" are supported.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """

        if fit_kwds is None:
            fit_kwds = {}

        if parallel_method == "sequential":
            results_l = self.fit_sequential(data_generator, fit_kwds,
                                            init_kwds_generator)

        elif parallel_method == "joblib":
            results_l = self.fit_joblib(data_generator, fit_kwds,
                                        parallel_backend,
                                        init_kwds_generator)

        else:
            raise ValueError("parallel_method: %s is currently not supported"
                             % parallel_method)

        params = self.join_method(results_l, **self.join_kwds)

        # NOTE that currently, the dummy result model that is initialized
        # here does not use any init_kwds from the init_kwds_generator event
        # if it is provided.  It is possible to imagine an edge case where
        # this might be a problem but given that the results model instance
        # does not correspond to any data partition this seems reasonable.
        res_mod = self.model_class([0], [0], **self.init_kwds)

        return self.results_class(res_mod, params, **self.results_kwds)

    def fit_sequential(self, data_generator, fit_kwds,
                       init_kwds_generator=None):
        """Sequentially performs the distributed estimation using
        the corresponding DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """

        results_l = []

        if init_kwds_generator is None:

            for pnum, (endog, exog) in enumerate(data_generator):

                results = _helper_fit_partition(self, pnum, endog, exog,
                                                fit_kwds)
                results_l.append(results)

        else:

            tup_gen = enumerate(zip(data_generator,
                                    init_kwds_generator))

            for pnum, ((endog, exog), init_kwds_e) in tup_gen:

                results = _helper_fit_partition(self, pnum, endog, exog,
                                                fit_kwds, init_kwds_e)
                results_l.append(results)

        return results_l

    def fit_joblib(self, data_generator, fit_kwds, parallel_backend,
                   init_kwds_generator=None):
        """Performs the distributed estimation in parallel using joblib

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """

        from statsmodels.tools.parallel import parallel_func

        par, f, n_jobs = parallel_func(_helper_fit_partition, self.partitions)

        if parallel_backend is None and init_kwds_generator is None:
            results_l = par(f(self, pnum, endog, exog, fit_kwds)
                            for pnum, (endog, exog)
                            in enumerate(data_generator))

        elif parallel_backend is not None and init_kwds_generator is None:
            with parallel_backend:
                results_l = par(f(self, pnum, endog, exog, fit_kwds)
                                for pnum, (endog, exog)
                                in enumerate(data_generator))

        elif parallel_backend is None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            results_l = par(f(self, pnum, endog, exog, fit_kwds, init_kwds)
                            for pnum, ((endog, exog), init_kwds)
                            in tup_gen)

        elif parallel_backend is not None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            with parallel_backend:
                results_l = par(f(self, pnum, endog, exog, fit_kwds, init_kwds)
                                for pnum, ((endog, exog), init_kwds)
                                in tup_gen)

        return results_l


class DistributedResults(LikelihoodModelResults):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        Class instance for model used for distributed data,
        this particular instance uses fake data and is really
        only to allow use of methods like predict.
    params : ndarray
        Parameter estimates from the fit model.
    """

    def __init__(self, model, params):
        super(DistributedResults, self).__init__(model, params)

    def predict(self, exog, *args, **kwargs):
        """Calls self.model.predict for the provided exog.  See
        Results.predict.

        Parameters
        ----------
        exog : array_like NOT optional
            The values for which we want to predict, unlike standard
            predict this is NOT optional since the data in self.model
            is fake.
        *args :
            Some models can take additional arguments. See the
            predict method of the model for the details.
        **kwargs :
            Some models can take additional keywords arguments. See the
            predict method of the model for the details.

        Returns
        -------
            prediction : ndarray, pandas.Series or pandas.DataFrame
            See self.model.predict
        """

        return self.model.predict(self.params, exog, *args, **kwargs)
