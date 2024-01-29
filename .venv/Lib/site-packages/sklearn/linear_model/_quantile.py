# Authors: David Dale <dale.david@mail.ru>
#          Christian Lorentzen <lorentzen.ch@gmail.com>
# License: BSD 3 clause
import warnings
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..base import BaseEstimator, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import _safe_indexing
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import _check_sample_weight
from ._base import LinearModel


class QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator):
    """Linear regression model that predicts conditional quantiles.

    The linear :class:`QuantileRegressor` optimizes the pinball loss for a
    desired `quantile` and is robust to outliers.

    This model uses an L1 regularization like
    :class:`~sklearn.linear_model.Lasso`.

    Read more in the :ref:`User Guide <quantile_regression>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    quantile : float, default=0.5
        The quantile that the model tries to predict. It must be strictly
        between 0 and 1. If 0.5 (default), the model predicts the 50%
        quantile, i.e. the median.

    alpha : float, default=1.0
        Regularization constant that multiplies the L1 penalty term.

    fit_intercept : bool, default=True
        Whether or not to fit the intercept.

    solver : {'highs-ds', 'highs-ipm', 'highs', 'interior-point', \
            'revised simplex'}, default='highs'
        Method used by :func:`scipy.optimize.linprog` to solve the linear
        programming formulation.

        From `scipy>=1.6.0`, it is recommended to use the highs methods because
        they are the fastest ones. Solvers "highs-ds", "highs-ipm" and "highs"
        support sparse input data and, in fact, always convert to sparse csc.

        From `scipy>=1.11.0`, "interior-point" is not available anymore.

        .. versionchanged:: 1.4
           The default of `solver` changed to `"highs"` in version 1.4.

    solver_options : dict, default=None
        Additional parameters passed to :func:`scipy.optimize.linprog` as
        options. If `None` and if `solver='interior-point'`, then
        `{"lstsq": True}` is passed to :func:`scipy.optimize.linprog` for the
        sake of stability.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the features.

    intercept_ : float
        The intercept of the model, aka bias term.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations performed by the solver.

    See Also
    --------
    Lasso : The Lasso is a linear model that estimates sparse coefficients
        with l1 regularization.
    HuberRegressor : Linear regression model that is robust to outliers.

    Examples
    --------
    >>> from sklearn.linear_model import QuantileRegressor
    >>> import numpy as np
    >>> n_samples, n_features = 10, 2
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> # the two following lines are optional in practice
    >>> from sklearn.utils.fixes import sp_version, parse_version
    >>> solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    >>> reg = QuantileRegressor(quantile=0.8, solver=solver).fit(X, y)
    >>> np.mean(y <= reg.predict(X))
    0.8
    """

    _parameter_constraints: dict = {
        "quantile": [Interval(Real, 0, 1, closed="neither")],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "fit_intercept": ["boolean"],
        "solver": [
            StrOptions(
                {
                    "highs-ds",
                    "highs-ipm",
                    "highs",
                    "interior-point",
                    "revised simplex",
                }
            ),
        ],
        "solver_options": [dict, None],
    }

    def __init__(
        self,
        *,
        quantile=0.5,
        alpha=1.0,
        fit_intercept=True,
        solver="highs",
        solver_options=None,
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_options = solver_options

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csc", "csr", "coo"],
            y_numeric=True,
            multi_output=False,
        )
        sample_weight = _check_sample_weight(sample_weight, X)

        n_features = X.shape[1]
        n_params = n_features

        if self.fit_intercept:
            n_params += 1
            # Note that centering y and X with _preprocess_data does not work
            # for quantile regression.

        # The objective is defined as 1/n * sum(pinball loss) + alpha * L1.
        # So we rescale the penalty term, which is equivalent.
        alpha = np.sum(sample_weight) * self.alpha

        if self.solver in (
            "highs-ds",
            "highs-ipm",
            "highs",
        ) and sp_version < parse_version("1.6.0"):
            raise ValueError(
                f"Solver {self.solver} is only available "
                f"with scipy>=1.6.0, got {sp_version}"
            )
        else:
            solver = self.solver

        if solver == "interior-point" and sp_version >= parse_version("1.11.0"):
            raise ValueError(
                f"Solver {solver} is not anymore available in SciPy >= 1.11.0."
            )

        if sparse.issparse(X) and solver not in ["highs", "highs-ds", "highs-ipm"]:
            raise ValueError(
                f"Solver {self.solver} does not support sparse X. "
                "Use solver 'highs' for example."
            )
        # make default solver more stable
        if self.solver_options is None and solver == "interior-point":
            solver_options = {"lstsq": True}
        else:
            solver_options = self.solver_options

        # After rescaling alpha, the minimization problem is
        #     min sum(pinball loss) + alpha * L1
        # Use linear programming formulation of quantile regression
        #     min_x c x
        #           A_eq x = b_eq
        #                0 <= x
        # x = (s0, s, t0, t, u, v) = slack variables >= 0
        # intercept = s0 - t0
        # coef = s - t
        # c = (0, alpha * 1_p, 0, alpha * 1_p, quantile * 1_n, (1-quantile) * 1_n)
        # residual = y - X@coef - intercept = u - v
        # A_eq = (1_n, X, -1_n, -X, diag(1_n), -diag(1_n))
        # b_eq = y
        # p = n_features
        # n = n_samples
        # 1_n = vector of length n with entries equal one
        # see https://stats.stackexchange.com/questions/384909/
        #
        # Filtering out zero sample weights from the beginning makes life
        # easier for the linprog solver.
        indices = np.nonzero(sample_weight)[0]
        n_indices = len(indices)  # use n_mask instead of n_samples
        if n_indices < len(sample_weight):
            sample_weight = sample_weight[indices]
            X = _safe_indexing(X, indices)
            y = _safe_indexing(y, indices)
        c = np.concatenate(
            [
                np.full(2 * n_params, fill_value=alpha),
                sample_weight * self.quantile,
                sample_weight * (1 - self.quantile),
            ]
        )
        if self.fit_intercept:
            # do not penalize the intercept
            c[0] = 0
            c[n_params] = 0

        if solver in ["highs", "highs-ds", "highs-ipm"]:
            # Note that highs methods always use a sparse CSC memory layout internally,
            # even for optimization problems parametrized using dense numpy arrays.
            # Therefore, we work with CSC matrices as early as possible to limit
            # unnecessary repeated memory copies.
            eye = sparse.eye(n_indices, dtype=X.dtype, format="csc")
            if self.fit_intercept:
                ones = sparse.csc_matrix(np.ones(shape=(n_indices, 1), dtype=X.dtype))
                A_eq = sparse.hstack([ones, X, -ones, -X, eye, -eye], format="csc")
            else:
                A_eq = sparse.hstack([X, -X, eye, -eye], format="csc")
        else:
            eye = np.eye(n_indices)
            if self.fit_intercept:
                ones = np.ones((n_indices, 1))
                A_eq = np.concatenate([ones, X, -ones, -X, eye, -eye], axis=1)
            else:
                A_eq = np.concatenate([X, -X, eye, -eye], axis=1)

        b_eq = y

        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            method=solver,
            options=solver_options,
        )
        solution = result.x
        if not result.success:
            failure = {
                1: "Iteration limit reached.",
                2: "Problem appears to be infeasible.",
                3: "Problem appears to be unbounded.",
                4: "Numerical difficulties encountered.",
            }
            warnings.warn(
                "Linear programming for QuantileRegressor did not succeed.\n"
                f"Status is {result.status}: "
                + failure.setdefault(result.status, "unknown reason")
                + "\n"
                + "Result message of linprog:\n"
                + result.message,
                ConvergenceWarning,
            )

        # positive slack - negative slack
        # solution is an array with (params_pos, params_neg, u, v)
        params = solution[:n_params] - solution[n_params : 2 * n_params]

        self.n_iter_ = result.nit

        if self.fit_intercept:
            self.coef_ = params[1:]
            self.intercept_ = params[0]
        else:
            self.coef_ = params
            self.intercept_ = 0.0
        return self
