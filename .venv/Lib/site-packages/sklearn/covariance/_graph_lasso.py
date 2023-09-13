"""GraphicalLasso: sparse inverse covariance estimation with an l1-penalized
estimator.
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause
# Copyright: INRIA
import operator
import sys
import time
import warnings
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from ..base import _fit_context
from ..exceptions import ConvergenceWarning

# mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
from ..linear_model import _cd_fast as cd_fast  # type: ignore
from ..linear_model import lars_path_gram
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _is_arraylike_not_scalar,
    check_random_state,
    check_scalar,
)
from . import EmpiricalCovariance, empirical_covariance, log_likelihood


# Helper functions to compute the objective and dual objective functions
# of the l1-penalized estimator
def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return cost


def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap


# The g-lasso algorithm
def _graphical_lasso(
    emp_cov,
    alpha,
    *,
    cov_init=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    _, n_features = emp_cov.shape
    if alpha == 0:
        # Early return without regularization
        precision_ = linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_)
        cost += n_features * np.log(2 * np.pi)
        d_gap = np.sum(emp_cov * precision_) - n_features
        return emp_cov, precision_, (cost, d_gap), 0

    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[:: n_features + 1]
    covariance_.flat[:: n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    i = 0  # initialize the counter to be robust to `max_iter=0`
    costs = list()
    # The different l1 regression solver have different numerical errors
    if mode == "cd":
        errors = dict(over="raise", invalid="ignore")
    else:
        errors = dict(invalid="raise")
    try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order="C")
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == "cd":
                        # Use coordinate descent
                        coefs = -(
                            precision_[indices != idx, idx]
                            / (precision_[idx, idx] + 1000 * eps)
                        )
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs,
                            alpha,
                            0,
                            sub_covariance,
                            row,
                            row,
                            max_iter,
                            enet_tol,
                            check_random_state(None),
                            False,
                        )
                    else:  # mode == "lars"
                        _, _, coefs = lars_path_gram(
                            Xy=row,
                            Gram=sub_covariance,
                            n_samples=row.size,
                            alpha_min=alpha / (n_features - 1),
                            copy_Gram=True,
                            eps=eps,
                            method="lars",
                            return_path=False,
                        )
                # Update the precision matrix
                precision_[idx, idx] = 1.0 / (
                    covariance_[idx, idx]
                    - np.dot(covariance_[indices != idx, idx], coefs)
                )
                precision_[indices != idx, idx] = -precision_[idx, idx] * coefs
                precision_[idx, indices != idx] = -precision_[idx, idx] * coefs
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs
            if not np.isfinite(precision_.sum()):
                raise FloatingPointError(
                    "The system is too ill-conditioned for this solver"
                )
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)
            if verbose:
                print(
                    "[graphical_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e"
                    % (i, cost, d_gap)
                )
            costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError(
                    "Non SPD result: the system is too ill-conditioned for this solver"
                )
        else:
            warnings.warn(
                "graphical_lasso: did not converge after %i iteration: dual gap: %.3e"
                % (max_iter, d_gap),
                ConvergenceWarning,
            )
    except FloatingPointError as e:
        e.args = (e.args[0] + ". The system is too ill-conditioned for this solver",)
        raise e

    return covariance_, precision_, costs, i + 1


def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        The sample covariance matrix.

    Notes
    -----
    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.
    """
    A = np.copy(emp_cov)
    A.flat[:: A.shape[0] + 1] = 0
    return np.max(np.abs(A))


@validate_params(
    {
        "emp_cov": ["array-like"],
        "cov_init": ["array-like", None],
        "return_costs": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def graphical_lasso(
    emp_cov,
    alpha,
    *,
    cov_init=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    return_costs=False,
    eps=np.finfo(np.float64).eps,
    return_n_iter=False,
):
    """L1-penalized covariance estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        graph_lasso has been renamed to graphical_lasso

    Parameters
    ----------
    emp_cov : array-like of shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance. If None, then the empirical
        covariance is used.

        .. deprecated:: 1.3
           `cov_init` is deprecated in 1.3 and will be removed in 1.5.
           It currently has no effect.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : bool, default=False
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        The estimated covariance matrix.

    precision : ndarray of shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    GraphicalLassoCV : Sparse inverse covariance with
        cross-validated choice of the l1 penalty.

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.
    """

    if cov_init is not None:
        warnings.warn(
            (
                "The cov_init parameter is deprecated in 1.3 and will be removed in "
                "1.5. It does not have any effect."
            ),
            FutureWarning,
        )

    model = GraphicalLasso(
        alpha=alpha,
        mode=mode,
        covariance="precomputed",
        tol=tol,
        enet_tol=enet_tol,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        assume_centered=True,
    ).fit(emp_cov)

    output = [model.covariance_, model.precision_]
    if return_costs:
        output.append(model.costs_)
    if return_n_iter:
        output.append(model.n_iter_)
    return tuple(output)


class BaseGraphicalLasso(EmpiricalCovariance):
    _parameter_constraints: dict = {
        **EmpiricalCovariance._parameter_constraints,
        "tol": [Interval(Real, 0, None, closed="right")],
        "enet_tol": [Interval(Real, 0, None, closed="right")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "mode": [StrOptions({"cd", "lars"})],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0, None, closed="both")],
    }
    _parameter_constraints.pop("store_precision")

    def __init__(
        self,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        super().__init__(assume_centered=assume_centered)
        self.tol = tol
        self.enet_tol = enet_tol
        self.max_iter = max_iter
        self.mode = mode
        self.verbose = verbose
        self.eps = eps


class GraphicalLasso(BaseGraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLasso has been renamed to GraphicalLasso

    Parameters
    ----------
    alpha : float, default=0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.
        Range is (0, inf].

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    covariance : "precomputed", default=None
        If covariance is "precomputed", the input data in `fit` is assumed
        to be the covariance matrix. If `None`, the empirical covariance
        is estimated from the data `X`.

        .. versionadded:: 1.3

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        The maximum number of iterations.

    verbose : bool, default=False
        If verbose is True, the objective function and dual gap are
        plotted at each iteration.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    costs_ : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

        .. versionadded:: 1.3

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1-penalized covariance estimator.
    GraphicalLassoCV : Sparse inverse covariance with
        cross-validated choice of the l1 penalty.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLasso
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLasso().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.049, 0.218, 0.019],
           [0.049, 0.364, 0.017, 0.034],
           [0.218, 0.017, 0.322, 0.093],
           [0.019, 0.034, 0.093, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])
    """

    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="both")],
        "covariance": [StrOptions({"precomputed"}), None],
    }

    def __init__(
        self,
        alpha=0.01,
        *,
        mode="cd",
        covariance=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alpha = alpha
        self.covariance = covariance

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the GraphicalLasso model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Covariance does not make sense for a single feature
        X = self._validate_data(X, ensure_min_features=2, ensure_min_samples=2)

        if self.covariance == "precomputed":
            emp_cov = X.copy()
            self.location_ = np.zeros(X.shape[1])
        else:
            emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
            if self.assume_centered:
                self.location_ = np.zeros(X.shape[1])
            else:
                self.location_ = X.mean(0)

        self.covariance_, self.precision_, self.costs_, self.n_iter_ = _graphical_lasso(
            emp_cov,
            alpha=self.alpha,
            cov_init=None,
            mode=self.mode,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
        )
        return self


# Cross-validation with GraphicalLasso
def graphical_lasso_path(
    X,
    alphas,
    cov_init=None,
    X_test=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    """l1-penalized covariance estimator along a path of decreasing alphas

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    alphas : array-like of shape (n_alphas,)
        The list of regularization parameters, decreasing order.

    cov_init : array of shape (n_features, n_features), default=None
        The initial guess for the covariance.

    X_test : array of shape (n_test_samples, n_features), default=None
        Optional test matrix to measure generalisation error.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. The tolerance must be a positive
        number.

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. The tolerance must be a positive number.

    max_iter : int, default=100
        The maximum number of iterations. This parameter should be a strictly
        positive integer.

    verbose : int or bool, default=False
        The higher the verbosity flag, the more information is printed
        during the fitting.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    Returns
    -------
    covariances_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated covariance matrices.

    precisions_ : list of shape (n_alphas,) of ndarray of shape \
            (n_features, n_features)
        The estimated (sparse) precision matrices.

    scores_ : list of shape (n_alphas,), dtype=float
        The generalisation error (log-likelihood) on the test data.
        Returned only if test data is passed.
    """
    inner_verbose = max(0, verbose - 1)
    emp_cov = empirical_covariance(X)
    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init
    covariances_ = list()
    precisions_ = list()
    scores_ = list()
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    for alpha in alphas:
        try:
            # Capture the errors, and move on
            covariance_, precision_, _, _ = _graphical_lasso(
                emp_cov,
                alpha=alpha,
                cov_init=covariance_,
                mode=mode,
                tol=tol,
                enet_tol=enet_tol,
                max_iter=max_iter,
                verbose=inner_verbose,
                eps=eps,
            )
            covariances_.append(covariance_)
            precisions_.append(precision_)
            if X_test is not None:
                this_score = log_likelihood(test_emp_cov, precision_)
        except FloatingPointError:
            this_score = -np.inf
            covariances_.append(np.nan)
            precisions_.append(np.nan)
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose == 1:
            sys.stderr.write(".")
        elif verbose > 1:
            if X_test is not None:
                print(
                    "[graphical_lasso_path] alpha: %.2e, score: %.2e"
                    % (alpha, this_score)
                )
            else:
                print("[graphical_lasso_path] alpha: %.2e" % alpha)
    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_


class GraphicalLassoCV(BaseGraphicalLasso):
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    .. versionchanged:: v0.20
        GraphLassoCV has been renamed to GraphicalLassoCV

    Parameters
    ----------
    alphas : int or array-like of shape (n_alphas,), dtype=float, default=4
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details. Range is [1, inf) for an integer.
        Range is (0, inf] for an array-like of floats.

    n_refinements : int, default=4
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed. Range is [1, inf).

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None changed from 3-fold to 5-fold.

    tol : float, default=1e-4
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped. Range is (0, inf].

    enet_tol : float, default=1e-4
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'. Range is (0, inf].

    max_iter : int, default=100
        Maximum number of iterations.

    mode : {'cd', 'lars'}, default='cd'
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where number of features is greater
        than number of samples. Elsewhere prefer cd which is more numerically
        stable.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    verbose : bool, default=False
        If verbose is True, the objective function and duality gap are
        printed at each iteration.

    eps : float, default=eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Default is `np.finfo(np.float64).eps`.

        .. versionadded:: 1.3

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix (inverse covariance).

    costs_ : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

        .. versionadded:: 1.3

    alpha_ : float
        Penalization parameter selected.

    cv_results_ : dict of ndarrays
        A dict with keys:

        alphas : ndarray of shape (n_alphas,)
            All penalization parameters explored.

        split(k)_test_score : ndarray of shape (n_alphas,)
            Log-likelihood score on left-out data across (k)th fold.

            .. versionadded:: 1.0

        mean_test_score : ndarray of shape (n_alphas,)
            Mean of scores over the folds.

            .. versionadded:: 1.0

        std_test_score : ndarray of shape (n_alphas,)
            Standard deviation of scores over the folds.

            .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run for the optimal alpha.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    graphical_lasso : L1-penalized covariance estimator.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.

    Notes
    -----
    The search for the optimal penalization parameter (`alpha`) is done on an
    iteratively refined grid: first the cross-validated scores on a grid are
    computed, then a new refined grid is centered around the maximum, and so
    on.

    One of the challenges which is faced here is that the solvers can
    fail to converge to a well-conditioned estimate. The corresponding
    values of `alpha` then come out as missing values, but the optimum may
    be close to these missing values.

    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import GraphicalLassoCV
    >>> true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
    ...                      [0.0, 0.4, 0.0, 0.0],
    ...                      [0.2, 0.0, 0.3, 0.1],
    ...                      [0.0, 0.0, 0.1, 0.7]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0, 0, 0],
    ...                                   cov=true_cov,
    ...                                   size=200)
    >>> cov = GraphicalLassoCV().fit(X)
    >>> np.around(cov.covariance_, decimals=3)
    array([[0.816, 0.051, 0.22 , 0.017],
           [0.051, 0.364, 0.018, 0.036],
           [0.22 , 0.018, 0.322, 0.094],
           [0.017, 0.036, 0.094, 0.69 ]])
    >>> np.around(cov.location_, decimals=3)
    array([0.073, 0.04 , 0.038, 0.143])
    """

    _parameter_constraints: dict = {
        **BaseGraphicalLasso._parameter_constraints,
        "alphas": [Interval(Integral, 0, None, closed="left"), "array-like"],
        "n_refinements": [Interval(Integral, 1, None, closed="left")],
        "cv": ["cv_object"],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        *,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",
        n_jobs=None,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
    ):
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the GraphicalLasso covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Covariance does not make sense for a single feature
        X = self._validate_data(X, ensure_min_features=2)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)

        cv = check_cv(self.cv, y, classifier=False)

        # List of (alpha, scores, covs)
        path = list()
        n_alphas = self.alphas
        inner_verbose = max(0, self.verbose - 1)

        if _is_arraylike_not_scalar(n_alphas):
            for alpha in self.alphas:
                check_scalar(
                    alpha,
                    "alpha",
                    Real,
                    min_val=0,
                    max_val=np.inf,
                    include_boundaries="right",
                )
            alphas = self.alphas
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            alpha_1 = alpha_max(emp_cov)
            alpha_0 = 1e-2 * alpha_1
            alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1), n_alphas)[::-1]

        t0 = time.time()
        for i in range(n_refinements):
            with warnings.catch_warnings():
                # No need to see the convergence warnings on this grid:
                # they will always be points that will not converge
                # during the cross-validation
                warnings.simplefilter("ignore", ConvergenceWarning)
                # Compute the cross-validated loss on the current grid

                # NOTE: Warm-restarting graphical_lasso_path has been tried,
                # and this did not allow to gain anything
                # (same execution time with or without).
                this_path = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(graphical_lasso_path)(
                        X[train],
                        alphas=alphas,
                        X_test=X[test],
                        mode=self.mode,
                        tol=self.tol,
                        enet_tol=self.enet_tol,
                        max_iter=int(0.1 * self.max_iter),
                        verbose=inner_verbose,
                        eps=self.eps,
                    )
                    for train, test in cv.split(X, y)
                )

            # Little danse to transform the list in what we need
            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using built in 'max' function to
            # have a fully-reproducible selection of the smallest alpha
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (alpha, scores, _) in enumerate(path):
                this_score = np.mean(scores)
                if this_score >= 0.1 / np.finfo(np.float64).eps:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # Refine the grid
            if best_index == 0:
                # We do not need to go back: we have chosen
                # the highest value of alpha for which there are
                # non-zero coefficients
                alpha_1 = path[0][0]
                alpha_0 = path[1][0]
            elif best_index == last_finite_idx and not best_index == len(path) - 1:
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                alpha_1 = path[best_index][0]
                alpha_0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                alpha_1 = path[best_index][0]
                alpha_0 = 0.01 * path[best_index][0]
            else:
                alpha_1 = path[best_index - 1][0]
                alpha_0 = path[best_index + 1][0]

            if not _is_arraylike_not_scalar(n_alphas):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0), n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print(
                    "[GraphicalLassoCV] Done refinement % 2i out of %i: % 3is"
                    % (i + 1, n_refinements, time.time() - t0)
                )

        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas = list(path[0])
        # Finally, compute the score with alpha = 0
        alphas.append(0)
        grid_scores.append(
            cross_val_score(
                EmpiricalCovariance(),
                X,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=inner_verbose,
            )
        )
        grid_scores = np.array(grid_scores)

        self.cv_results_ = {"alphas": np.array(alphas)}

        for i in range(grid_scores.shape[1]):
            self.cv_results_[f"split{i}_test_score"] = grid_scores[:, i]

        self.cv_results_["mean_test_score"] = np.mean(grid_scores, axis=1)
        self.cv_results_["std_test_score"] = np.std(grid_scores, axis=1)

        best_alpha = alphas[best_index]
        self.alpha_ = best_alpha

        # Finally fit the model with the selected alpha
        self.covariance_, self.precision_, self.costs_, self.n_iter_ = _graphical_lasso(
            emp_cov,
            alpha=best_alpha,
            mode=self.mode,
            tol=self.tol,
            enet_tol=self.enet_tol,
            max_iter=self.max_iter,
            verbose=inner_verbose,
            eps=self.eps,
        )
        return self
