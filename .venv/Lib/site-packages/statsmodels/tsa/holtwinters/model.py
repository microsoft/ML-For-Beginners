"""
Notes
-----
Code written using below textbook as a reference.
Results are checked against the expected outcomes in the text book.

Properties:
Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and
practice. OTexts, 2014.

Author: Terence L van Zyl
Modified: Kevin Sheppard
"""
from statsmodels.compat.pandas import deprecate_kwarg

import contextlib
from typing import Any, Hashable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
    _initialization_heuristic,
    _initialization_simple,
)
from statsmodels.tsa.holtwinters import (
    _exponential_smoothers as smoothers,
    _smoothers as py_smoothers,
)
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
    to_restricted,
    to_unrestricted,
)
from statsmodels.tsa.holtwinters.results import (
    HoltWintersResults,
    HoltWintersResultsWrapper,
)
from statsmodels.tsa.tsatools import freq_to_period

SMOOTHERS = {
    ("mul", "add"): smoothers.holt_win_add_mul_dam,
    ("mul", "mul"): smoothers.holt_win_mul_mul_dam,
    ("mul", None): smoothers.holt_win__mul,
    ("add", "add"): smoothers.holt_win_add_add_dam,
    ("add", "mul"): smoothers.holt_win_mul_add_dam,
    ("add", None): smoothers.holt_win__add,
    (None, "add"): smoothers.holt_add_dam,
    (None, "mul"): smoothers.holt_mul_dam,
    (None, None): smoothers.holt__,
}

PY_SMOOTHERS = {
    ("mul", "add"): py_smoothers.holt_win_add_mul_dam,
    ("mul", "mul"): py_smoothers.holt_win_mul_mul_dam,
    ("mul", None): py_smoothers.holt_win__mul,
    ("add", "add"): py_smoothers.holt_win_add_add_dam,
    ("add", "mul"): py_smoothers.holt_win_mul_add_dam,
    ("add", None): py_smoothers.holt_win__add,
    (None, "add"): py_smoothers.holt_add_dam,
    (None, "mul"): py_smoothers.holt_mul_dam,
    (None, None): py_smoothers.holt__,
}


def opt_wrapper(func):
    def f(*args, **kwargs):
        err = func(*args, **kwargs)
        if isinstance(err, np.ndarray):
            return err.T @ err
        return err

    return f


class _OptConfig:
    alpha: float
    beta: float
    phi: float
    gamma: float
    level: float
    trend: float
    seasonal: np.ndarray
    y: np.ndarray
    params: np.ndarray
    mask: np.ndarray
    mle_retvals: Any

    def unpack_parameters(self, params) -> "_OptConfig":
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.level = params[3]
        self.trend = params[4]
        self.phi = params[5]
        self.seasonal = params[6:]

        return self


class ExponentialSmoothing(TimeSeriesModel):
    """
    Holt Winter's Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use the value as lambda.
    bounds : dict[str, tuple[float, float]], optional
        An dictionary containing bounds for the parameters in the model,
        excluding the initial values if estimated. The keys of the dictionary
        are the variable names, e.g., smoothing_level or initial_slope.
        The initial seasonal variables are labeled initial_seasonal.<j>
        for j=0,...,m-1 where m is the number of period in a full season.
        Use None to indicate a non-binding constraint, e.g., (0, None)
        constrains a parameter to be non-negative.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    This is a full implementation of the holt winters exponential smoothing as
    per [1]_. This includes all the unstable methods as well as the stable
    methods. The implementation of the library covers the functionality of the
    R library as much as possible whilst still being Pythonic.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    @deprecate_kwarg("damped", "damped_trend")
    def __init__(
        self,
        endog,
        trend=None,
        damped_trend=False,
        seasonal=None,
        *,
        seasonal_periods=None,
        initialization_method="estimated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        use_boxcox=False,
        bounds=None,
        dates=None,
        freq=None,
        missing="none",
    ):
        super().__init__(endog, None, dates, freq, missing=missing)
        self._y = self._data = array_like(
            endog, "endog", ndim=1, contiguous=True, order="C"
        )
        options = ("add", "mul", "additive", "multiplicative")
        trend = string_like(trend, "trend", options=options, optional=True)
        if trend in ["additive", "multiplicative"]:
            trend = {"additive": "add", "multiplicative": "mul"}[trend]
        self.trend = trend
        self.damped_trend = bool_like(damped_trend, "damped_trend")
        seasonal = string_like(
            seasonal, "seasonal", options=options, optional=True
        )
        if seasonal in ["additive", "multiplicative"]:
            seasonal = {"additive": "add", "multiplicative": "mul"}[seasonal]
        self.seasonal = seasonal
        self.has_trend = trend in ["mul", "add"]
        self.has_seasonal = seasonal in ["mul", "add"]
        if (self.trend == "mul" or self.seasonal == "mul") and not np.all(
            self._data > 0.0
        ):
            raise ValueError(
                "endog must be strictly positive when using"
                "multiplicative trend or seasonal components."
            )
        if self.damped_trend and not self.has_trend:
            raise ValueError("Can only dampen the trend component")
        if self.has_seasonal:
            self.seasonal_periods = int_like(
                seasonal_periods, "seasonal_periods", optional=True
            )
            if seasonal_periods is None:
                try:
                    self.seasonal_periods = freq_to_period(self._index_freq)
                except Exception:
                    raise ValueError(
                        "seasonal_periods has not been provided and index "
                        "does not have a known freq. You must provide "
                        "seasonal_periods"
                    )
            if self.seasonal_periods <= 1:
                raise ValueError("seasonal_periods must be larger than 1.")
            assert self.seasonal_periods is not None
        else:
            self.seasonal_periods = 0
        self.nobs = len(self.endog)
        options = ("known", "estimated", "heuristic", "legacy-heuristic")
        self._initialization_method = string_like(
            initialization_method,
            "initialization_method",
            optional=False,
            options=options,
        )
        self._initial_level = float_like(
            initial_level, "initial_level", optional=True
        )
        self._initial_trend = float_like(
            initial_trend, "initial_trend", optional=True
        )
        self._initial_seasonal = array_like(
            initial_seasonal, "initial_seasonal", optional=True
        )
        estimated = self._initialization_method == "estimated"
        self._estimate_level = estimated
        self._estimate_trend = estimated and self.trend is not None
        self._estimate_seasonal = estimated and self.seasonal is not None
        self._bounds = self._check_bounds(bounds)
        self._use_boxcox = use_boxcox
        self._lambda = np.nan
        self._y = self._boxcox()
        self._initialize()
        self._fixed_parameters = {}

    def _check_bounds(self, bounds):
        bounds = dict_like(bounds, "bounds", optional=True)
        if bounds is None:
            return
        msg = (
            "bounds must be a dictionary of 2-element tuples of the form"
            " (lb, ub) where lb < ub, lb>=0 and ub<=1"
        )
        variables = self._ordered_names()
        for key in bounds:
            if key not in variables:
                supported = ", ".join(variables[:-1])
                supported += ", and " + variables[-1]
                raise KeyError(
                    f"{key} does not match the list of supported variables "
                    f"names: {supported}."
                )
            bound = bounds[key]
            if not isinstance(bound, tuple):
                raise TypeError(msg)
            lb = bound[0] if bound[0] is not None else -np.inf
            ub = bound[1] if bound[1] is not None else np.inf
            if len(bound) != 2 or lb >= ub:
                raise ValueError(msg)
            if ("smoothing" in key or "damp" in key) and (
                bound[0] < 0.0 or bound[1] > 1.0
            ):
                raise ValueError(
                    f"{key} must have a lower bound >= 0.0 and <= 1.0"
                )
        return bounds

    def _boxcox(self):
        if self._use_boxcox is None or self._use_boxcox is False:
            self._lambda = np.nan
            return self._y
        if self._use_boxcox is True:
            y, self._lambda = boxcox(self._y)
        elif isinstance(self._use_boxcox, (int, float)):
            self._lambda = float(self._use_boxcox)
            y = boxcox(self._y, self._use_boxcox)
        else:
            raise TypeError("use_boxcox must be True, False or a float.")
        return y

    @contextlib.contextmanager
    def fix_params(self, values):
        """
        Temporarily fix parameters for estimation.

        Parameters
        ----------
        values : dict
            Values to fix. The key is the parameter name and the value is the
            fixed value.

        Yields
        ------
        None
            No value returned.

        Examples
        --------
        >>> from statsmodels.datasets.macrodata import load_pandas
        >>> data = load_pandas()
        >>> import statsmodels.tsa.api as tsa
        >>> mod = tsa.ExponentialSmoothing(data.data.realcons, trend="add",
        ...                                initialization_method="estimated")
        >>> with mod.fix_params({"smoothing_level": 0.2}):
        ...     mod.fit()
        """
        values = dict_like(values, "values")
        valid_keys = ("smoothing_level",)
        if self.has_trend:
            valid_keys += ("smoothing_trend",)
        if self.has_seasonal:
            valid_keys += ("smoothing_seasonal",)
            m = self.seasonal_periods
            valid_keys += tuple([f"initial_seasonal.{i}" for i in range(m)])
        if self.damped_trend:
            valid_keys += ("damping_trend",)
        if self._initialization_method in ("estimated", None):
            extra_keys = [
                key.replace("smoothing_", "initial_")
                for key in valid_keys
                if "smoothing_" in key
            ]
            valid_keys += tuple(extra_keys)

        for key in values:
            if key not in valid_keys:
                valid = ", ".join(valid_keys[:-1]) + ", and " + valid_keys[-1]
                raise KeyError(
                    f"{key} if not allowed. Only {valid} are supported in "
                    "this specification."
                )

        if "smoothing_level" in values:
            alpha = values["smoothing_level"]
            if alpha <= 0.0:
                raise ValueError("smoothing_level must be in (0, 1)")
            beta = values.get("smoothing_trend", 0.0)
            if beta > alpha:
                raise ValueError("smoothing_trend must be <= smoothing_level")
            gamma = values.get("smoothing_seasonal", 0.0)
            if gamma > 1 - alpha:
                raise ValueError(
                    "smoothing_seasonal must be <= 1 - smoothing_level"
                )

        try:
            self._fixed_parameters = values
            yield
        finally:
            self._fixed_parameters = {}

    def _initialize(self):
        if self._initialization_method == "known":
            return self._initialize_known()
        msg = (
            f"initialization method is {self._initialization_method} but "
            "initial_{0} has been set."
        )
        if self._initial_level is not None:
            raise ValueError(msg.format("level"))
        if self._initial_trend is not None:
            raise ValueError(msg.format("trend"))
        if self._initial_seasonal is not None:
            raise ValueError(msg.format("seasonal"))
        if self._initialization_method == "legacy-heuristic":
            return self._initialize_legacy()
        elif self._initialization_method == "heuristic":
            return self._initialize_heuristic()
        elif self._initialization_method == "estimated":
            if self.nobs < 10 + 2 * (self.seasonal_periods // 2):
                return self._initialize_simple()
            else:
                return self._initialize_heuristic()

    def _initialize_simple(self):
        trend = self.trend if self.has_trend else False
        seasonal = self.seasonal if self.has_seasonal else False
        lvl, trend, seas = _initialization_simple(
            self._y, trend, seasonal, self.seasonal_periods
        )
        self._initial_level = lvl
        self._initial_trend = trend
        self._initial_seasonal = seas

    def _initialize_heuristic(self):
        trend = self.trend if self.has_trend else False
        seasonal = self.seasonal if self.has_seasonal else False
        lvl, trend, seas = _initialization_heuristic(
            self._y, trend, seasonal, self.seasonal_periods
        )
        self._initial_level = lvl
        self._initial_trend = trend
        self._initial_seasonal = seas

    def _initialize_legacy(self):
        lvl, trend, seasonal = self.initial_values(force=True)
        self._initial_level = lvl
        self._initial_trend = trend
        self._initial_seasonal = seasonal

    def _initialize_known(self):
        msg = "initialization is 'known' but initial_{0} not given"
        if self._initial_level is None:
            raise ValueError(msg.format("level"))
        excess = "initial_{0} set but model has no {0} component"
        if self.has_trend and self._initial_trend is None:
            raise ValueError(msg.format("trend"))
        elif not self.has_trend and self._initial_trend is not None:
            raise ValueError(excess.format("trend"))
        if self.has_seasonal and self._initial_seasonal is None:
            raise ValueError(msg.format("seasonal"))
        elif not self.has_seasonal and self._initial_seasonal is not None:
            raise ValueError(excess.format("seasonal"))

    def predict(self, params, start=None, end=None):
        """
        In-sample and out-of-sample prediction.

        Parameters
        ----------
        params : ndarray
            The fitted model parameters.
        start : int, str, or datetime
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.
        end : int, str, or datetime
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type.

        Returns
        -------
        ndarray
            The predicted values.
        """
        if start is None:
            freq = getattr(self._index, "freq", 1)
            if isinstance(freq, int):
                start = self._index.shape[0]
            else:
                start = self._index[-1] + freq
        start, end, out_of_sample, _ = self._get_prediction_index(
            start=start, end=end
        )
        if out_of_sample > 0:
            res = self._predict(h=out_of_sample, **params)
        else:
            res = self._predict(h=0, **params)
        return res.fittedfcast[start : end + out_of_sample + 1]

    def _enforce_bounds(self, p, sel, lb, ub):
        initial_p = p[sel]

        # Ensure strictly inbounds
        loc = initial_p <= lb
        upper = ub[loc].copy()
        upper[~np.isfinite(upper)] = 100.0
        eps = 1e-4
        initial_p[loc] = lb[loc] + eps * (upper - lb[loc])

        loc = initial_p >= ub
        lower = lb[loc].copy()
        lower[~np.isfinite(lower)] = -100.0
        eps = 1e-4
        initial_p[loc] = ub[loc] - eps * (ub[loc] - lower)

        return initial_p

    @staticmethod
    def _check_blocked_keywords(
        d: dict, keys: Sequence[Hashable], name="kwargs"
    ):
        for key in keys:
            if key in d:
                raise ValueError(f"{name} must not contain '{key}'")

    def _check_bound_feasibility(self, bounds):
        if bounds[1][0] > bounds[0][1]:
            raise ValueError(
                "The bounds for smoothing_trend and smoothing_level are "
                "incompatible since smoothing_trend <= smoothing_level."
            )
        if bounds[2][0] > (1 - bounds[0][1]):
            raise ValueError(
                "The bounds for smoothing_seasonal and smoothing_level "
                "are incompatible since smoothing_seasonal <= "
                "1 - smoothing_level."
            )

    @staticmethod
    def _setup_brute(sel, bounds, alpha):
        # More points when fewer parameters
        ns = 87 // sel[:3].sum()

        if not sel[0]:
            # Easy case since no cross-constraints
            nparams = int(sel[1]) + int(sel[2])
            args = []
            for i in range(1, 3):
                if sel[i]:
                    bound = bounds[i]
                    step = bound[1] - bound[0]
                    lb = bound[0] + 0.005 * step
                    if i == 1:
                        ub = min(bound[1], alpha) - 0.005 * step
                    else:
                        ub = min(bound[1], 1 - alpha) - 0.005 * step
                    args.append(np.linspace(lb, ub, ns))
            points = np.stack(np.meshgrid(*args))
            points = points.reshape((nparams, -1)).T
            return np.ascontiguousarray(points)

        bound = bounds[0]
        step = 0.005 * (bound[1] - bound[0])
        points = np.linspace(bound[0] + step, bound[1] - step, ns)
        if not sel[1] and not sel[2]:
            return points[:, None]

        combined = []
        b_bounds = bounds[1]
        g_bounds = bounds[2]
        if sel[1] and sel[2]:
            for a in points:
                b_lb = b_bounds[0]
                b_ub = min(b_bounds[1], a)
                g_lb = g_bounds[0]
                g_ub = min(g_bounds[1], 1 - a)
                if b_lb > b_ub or g_lb > g_ub:
                    # infeasible point
                    continue
                nb = int(np.ceil(ns * np.sqrt(a)))
                ng = int(np.ceil(ns * np.sqrt(1 - a)))
                b = np.linspace(b_lb, b_ub, nb)
                g = np.linspace(g_lb, g_ub, ng)
                both = np.stack(np.meshgrid(b, g)).reshape(2, -1).T
                final = np.empty((both.shape[0], 3))
                final[:, 0] = a
                final[:, 1:] = both
                combined.append(final)
        elif sel[1]:
            for a in points:
                b_lb = b_bounds[0]
                b_ub = min(b_bounds[1], a)
                if b_lb > b_ub:
                    # infeasible point
                    continue
                nb = int(np.ceil(ns * np.sqrt(a)))
                final = np.empty((nb, 2))
                final[:, 0] = a
                final[:, 1] = np.linspace(b_lb, b_ub, nb)
                combined.append(final)
        else:  # sel[2]
            for a in points:
                g_lb = g_bounds[0]
                g_ub = min(g_bounds[1], 1 - a)
                if g_lb > g_ub:
                    # infeasible point
                    continue
                ng = int(np.ceil(ns * np.sqrt(1 - a)))
                final = np.empty((ng, 2))
                final[:, 1] = np.linspace(g_lb, g_ub, ng)
                final[:, 0] = a
                combined.append(final)

        return np.vstack(combined)

    def _ordered_names(self):
        names = (
            "smoothing_level",
            "smoothing_trend",
            "smoothing_seasonal",
            "initial_level",
            "initial_trend",
            "damping_trend",
        )
        m = self.seasonal_periods
        names += tuple([f"initial_seasonal.{i}" for i in range(m)])
        return names

    def _update_for_fixed(self, sel, alpha, beta, gamma, phi, l0, b0, s0):
        if self._fixed_parameters:
            fixed = self._fixed_parameters
            names = self._ordered_names()
            not_fixed = np.array([name not in fixed for name in names])
            if (~sel[~not_fixed]).any():
                invalid = []
                for name, s, nf in zip(names, sel, not_fixed):
                    if not s and not nf:
                        invalid.append(name)
                invalid_names = ", ".join(invalid)
                raise ValueError(
                    "Cannot fix a parameter that is not being "
                    f"estimated: {invalid_names}"
                )

            sel &= not_fixed
            alpha = fixed.get("smoothing_level", alpha)
            beta = fixed.get("smoothing_trend", beta)
            gamma = fixed.get("smoothing_seasonal", gamma)
            phi = fixed.get("damping_trend", phi)
            l0 = fixed.get("initial_level", l0)
            b0 = fixed.get("initial_trend", b0)
            for i in range(self.seasonal_periods):
                s0[i] = fixed.get(f"initial_seasonal.{i}", s0[i])
        return sel, alpha, beta, gamma, phi, l0, b0, s0

    def _construct_bounds(self):
        trend_lb = 0.0 if self.trend == "mul" else None
        season_lb = 0.0 if self.seasonal == "mul" else None
        lvl_lb = None if trend_lb is None and season_lb is None else 0.0
        bounds = [
            (0.0, 1.0),  # alpha
            (0.0, 1.0),  # beta
            (0.0, 1.0),  # gamma
            (lvl_lb, None),  # level
            (trend_lb, None),  # trend
            (0.8, 0.995),  # phi
        ]
        bounds += [(season_lb, None)] * self.seasonal_periods
        if self._bounds is not None:
            assert isinstance(self._bounds, dict)
            for i, name in enumerate(self._ordered_names()):
                bounds[i] = self._bounds.get(name, bounds[i])
        # Update bounds to account for fixed parameters
        fixed = self._fixed_parameters
        if "smoothing_level" in fixed:
            # Update bounds if fixed alpha
            alpha = fixed["smoothing_level"]
            # beta <= alpha
            if bounds[1][1] > alpha:
                bounds[1] = (bounds[1][0], alpha)
            # gamma <= 1 - alpha
            if bounds[2][1] > (1 - alpha):
                bounds[2] = (bounds[2][0], 1 - alpha)
            # gamma <= 1 - alpha
        if "smoothing_trend" in fixed:
            # beta <= alpha
            beta = fixed["smoothing_trend"]
            bounds[0] = (max(beta, bounds[0][0]), bounds[0][1])
        if "smoothing_seasonal" in fixed:
            gamma = fixed["smoothing_seasonal"]
            # gamma <= 1 - alpha => alpha <= 1 - gamma
            bounds[0] = (bounds[0][0], min(1 - gamma, bounds[0][1]))
        # Ensure bounds are feasible
        for i, name in enumerate(self._ordered_names()):
            lb = bounds[i][0] if bounds[i][0] is not None else -np.inf
            ub = bounds[i][1] if bounds[i][1] is not None else np.inf
            if lb >= ub:
                raise ValueError(
                    "After adjusting for user-provided bounds fixed values, "
                    f"the resulting set of bounds for {name}, {bounds[i]}, "
                    "are infeasible."
                )
        self._check_bound_feasibility(bounds)
        return bounds

    def _get_starting_values(
        self,
        params,
        start_params,
        use_brute,
        sel,
        hw_args,
        bounds,
        alpha,
        func,
    ):
        if start_params is None and use_brute and np.any(sel[:3]):
            # Have a quick look in the region for a good starting place for
            # alpha, beta & gamma using fixed values for initial
            m = self.seasonal_periods
            sv_sel = np.array([False] * (6 + m))
            sv_sel[:3] = True
            sv_sel &= sel
            hw_args.xi = sv_sel.astype(int)
            hw_args.transform = False
            # Setup the grid points, respecting constraints
            points = self._setup_brute(sv_sel, bounds, alpha)
            opt = opt_wrapper(func)
            best_val = np.inf
            best_params = points[0]
            for point in points:
                val = opt(point, hw_args)
                if val < best_val:
                    best_params = point
                    best_val = val
            params[sv_sel] = best_params
        elif start_params is not None:
            if len(start_params) != sel.sum():
                msg = "start_params must have {0} values but has {1}."
                nxi, nsp = len(sel), len(start_params)
                raise ValueError(msg.format(nxi, nsp))
            params[sel] = start_params
        return params

    def _optimize_parameters(
        self, data: _OptConfig, use_brute, method, kwargs
    ) -> _OptConfig:
        # Prepare starting values
        alpha = data.alpha
        beta = data.beta
        phi = data.phi
        gamma = data.gamma
        y = data.y
        start_params = data.params

        has_seasonal = self.has_seasonal
        has_trend = self.has_trend
        trend = self.trend
        seasonal = self.seasonal
        damped_trend = self.damped_trend

        m = self.seasonal_periods
        params = np.zeros(6 + m)
        l0, b0, s0 = self.initial_values(
            initial_level=data.level, initial_trend=data.trend
        )

        init_alpha = alpha if alpha is not None else 0.5 / max(m, 1)
        init_beta = beta
        if beta is None and has_trend:
            init_beta = 0.1 * init_alpha
        init_gamma = gamma
        if has_seasonal and gamma is None:
            init_gamma = 0.05 * (1 - init_alpha)
        init_phi = phi if phi is not None else 0.99
        # Selection of parameters to optimize
        sel = np.array(
            [
                alpha is None,
                has_trend and beta is None,
                has_seasonal and gamma is None,
                self._estimate_level,
                self._estimate_trend,
                damped_trend and phi is None,
            ]
            + [has_seasonal and self._estimate_seasonal] * m,
        )
        (
            sel,
            init_alpha,
            init_beta,
            init_gamma,
            init_phi,
            l0,
            b0,
            s0,
        ) = self._update_for_fixed(
            sel, init_alpha, init_beta, init_gamma, init_phi, l0, b0, s0
        )

        func = SMOOTHERS[(seasonal, trend)]
        params[:6] = [init_alpha, init_beta, init_gamma, l0, b0, init_phi]
        if m:
            params[-m:] = s0
        if not np.any(sel):
            from statsmodels.tools.sm_exceptions import EstimationWarning

            message = (
                "Model has no free parameters to estimate. Set "
                "optimized=False to suppress this warning"
            )
            warnings.warn(message, EstimationWarning, stacklevel=3)
            data = data.unpack_parameters(params)
            data.params = params
            data.mask = sel

            return data
        orig_bounds = self._construct_bounds()

        bounds = np.array(orig_bounds[:3], dtype=float)
        hw_args = HoltWintersArgs(
            sel.astype(int), params, bounds, y, m, self.nobs
        )
        params = self._get_starting_values(
            params,
            start_params,
            use_brute,
            sel,
            hw_args,
            bounds,
            init_alpha,
            func,
        )

        # We always use [0, 1] for a, b and g and handle transform inside
        mod_bounds = [(0, 1)] * 3 + orig_bounds[3:]
        relevant_bounds = [bnd for bnd, flag in zip(mod_bounds, sel) if flag]
        bounds = np.array(relevant_bounds, dtype=float)
        lb, ub = bounds.T
        lb[np.isnan(lb)] = -np.inf
        ub[np.isnan(ub)] = np.inf
        hw_args.xi = sel.astype(int)

        # Ensure strictly inbounds
        initial_p = self._enforce_bounds(params, sel, lb, ub)
        # Transform to unrestricted space
        params[sel] = initial_p
        params[:3] = to_unrestricted(params, sel, hw_args.bounds)
        initial_p = params[sel]
        # Ensure parameters are transformed internally
        hw_args.transform = True
        if method in ("least_squares", "ls"):
            # Least squares uses a different format for bounds
            ls_bounds = lb, ub
            self._check_blocked_keywords(kwargs, ("args", "bounds"))
            res = least_squares(
                func, initial_p, bounds=ls_bounds, args=(hw_args,), **kwargs
            )
            success = res.success
        elif method in ("basinhopping", "bh"):
            # Take a deeper look in the local minimum we are in to find the
            # best solution to parameters, maybe hop around to try escape the
            # local minimum we may be in.
            minimizer_kwargs = {"args": (hw_args,), "bounds": relevant_bounds}
            kwargs = kwargs.copy()
            if "minimizer_kwargs" in kwargs:
                self._check_blocked_keywords(
                    kwargs["minimizer_kwargs"],
                    ("args", "bounds"),
                    name="kwargs['minimizer_kwargs']",
                )
                minimizer_kwargs.update(kwargs["minimizer_kwargs"])
                del kwargs["minimizer_kwargs"]
            default_kwargs = {
                "minimizer_kwargs": minimizer_kwargs,
                "stepsize": 0.01,
            }
            default_kwargs.update(kwargs)
            obj = opt_wrapper(func)
            res = basinhopping(obj, initial_p, **default_kwargs)
            success = res.lowest_optimization_result.success
        else:
            obj = opt_wrapper(func)
            self._check_blocked_keywords(kwargs, ("args", "bounds", "method"))
            res = minimize(
                obj,
                initial_p,
                args=(hw_args,),
                bounds=relevant_bounds,
                method=method,
                **kwargs,
            )
            success = res.success
        # finally transform to restricted space
        params[sel] = res.x
        params[:3] = to_restricted(params, sel, hw_args.bounds)
        res.x = params[sel]

        if not success:
            from statsmodels.tools.sm_exceptions import ConvergenceWarning

            warnings.warn(
                "Optimization failed to converge. Check mle_retvals.",
                ConvergenceWarning,
            )
        params[sel] = res.x

        data.unpack_parameters(params)
        data.params = params
        data.mask = sel
        data.mle_retvals = res

        return data

    @deprecate_kwarg("smoothing_slope", "smoothing_trend")
    @deprecate_kwarg("initial_slope", "initial_trend")
    @deprecate_kwarg("damping_slope", "damping_trend")
    def fit(
        self,
        smoothing_level=None,
        smoothing_trend=None,
        smoothing_seasonal=None,
        damping_trend=None,
        *,
        optimized=True,
        remove_bias=False,
        start_params=None,
        method=None,
        minimize_kwargs=None,
        use_brute=True,
        use_boxcox=None,
        use_basinhopping=None,
        initial_level=None,
        initial_trend=None,
    ):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal : float, optional
            The gamma value of the holt winters seasonal method, if the value
            is set then this value will be used as the value.
        damping_trend : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        start_params : array_like, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data. See
            the notes for the structure of the model parameters.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" , "TNC",
            "SLSQP" (default), "Powell", "trust-constr", "basinhopping" (also
            "bh") and "least_squares" (also "ls"). basinhopping tries multiple
            starting values in an attempt to find a global minimizer in
            non-convex problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B", "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares functions. The valid keywords are optimizer
            specific. Consult SciPy's documentation for the full set of
            options.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.

            .. deprecated:: 0.12

               Set use_boxcox when constructing the model

        use_basinhopping : bool, optional
            Deprecated. Using Basin Hopping optimizer to find optimal values.
            Use ``method`` instead.

            .. deprecated:: 0.12

               Use ``method`` instead.

        initial_level : float, optional
            Value to use when initializing the fitted level.

            .. deprecated:: 0.12

               Set initial_level when constructing the model

        initial_trend : float, optional
            Value to use when initializing the fitted trend.

            .. deprecated:: 0.12

               Set initial_trend when constructing the model
               or set initialization_method.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the holt winters exponential smoothing
        as per [1]. This includes all the unstable methods as well as the
        stable methods. The implementation of the library covers the
        functionality of the R library as much as possible whilst still
        being Pythonic.

        The parameters are ordered

        [alpha, beta, gamma, initial_level, initial_trend, phi]

        which are then followed by m seasonal values if the model contains
        a seasonal smoother. Any parameter not relevant for the model is
        omitted. For example, a model that has a level and a seasonal
        component, but no trend and is not damped, would have starting
        values

        [alpha, gamma, initial_level, s0, s1, ..., s<m-1>]

        where sj is the initial value for seasonal component j.

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        # Variable renames to alpha,beta, etc as this helps with following the
        # mathematical notation in general
        alpha = float_like(smoothing_level, "smoothing_level", True)
        beta = float_like(smoothing_trend, "smoothing_trend", True)
        gamma = float_like(smoothing_seasonal, "smoothing_seasonal", True)
        phi = float_like(damping_trend, "damping_trend", True)
        initial_level = float_like(initial_level, "initial_level", True)
        initial_trend = float_like(initial_trend, "initial_trend", True)
        start_params = array_like(start_params, "start_params", optional=True)
        minimize_kwargs = dict_like(
            minimize_kwargs, "minimize_kwargs", optional=True
        )
        minimize_kwargs = {} if minimize_kwargs is None else minimize_kwargs
        use_basinhopping = bool_like(
            use_basinhopping, "use_basinhopping", optional=True
        )
        supported_methods = ("basinhopping", "bh")
        supported_methods += ("least_squares", "ls")
        supported_methods += (
            "L-BFGS-B",
            "TNC",
            "SLSQP",
            "Powell",
            "trust-constr",
        )
        method = string_like(
            method,
            "method",
            options=supported_methods,
            lower=False,
            optional=True,
        )
        # TODO: Deprecate initial_level and related parameters from fit
        if initial_level is not None or initial_trend is not None:
            raise ValueError(
                "Initial values were set during model construction. These "
                "cannot be changed during fit."
            )
        if use_boxcox is not None:
            raise ValueError(
                "use_boxcox was set at model initialization and cannot "
                "be changed"
            )
        elif self._use_boxcox is None:
            use_boxcox = False
        else:
            use_boxcox = self._use_boxcox

        if use_basinhopping is not None:
            raise ValueError(
                "use_basinhopping is deprecated. Set optimization method "
                "using 'method'."
            )

        data = self._data
        damped = self.damped_trend
        phi = phi if damped else 1.0
        if self._use_boxcox is None:
            if use_boxcox == "log":
                lamda = 0.0
                y = boxcox(data, lamda)
            elif isinstance(use_boxcox, float):
                lamda = use_boxcox
                y = boxcox(data, lamda)
            elif use_boxcox:
                y, lamda = boxcox(data)
                # use_boxcox = lamda
            else:
                y = data.squeeze()
        else:
            y = self._y

        self._y = y
        res = _OptConfig()
        res.alpha = alpha
        res.beta = beta
        res.phi = phi
        res.gamma = gamma
        res.level = initial_level
        res.trend = initial_trend
        res.seasonal = None
        res.y = y
        res.params = start_params
        res.mle_retvals = res.mask = None
        method = "SLSQP" if method is None else method
        if optimized:
            res = self._optimize_parameters(
                res, use_brute, method, minimize_kwargs
            )
        else:
            l0, b0, s0 = self.initial_values(
                initial_level=initial_level, initial_trend=initial_trend
            )
            res.level = l0
            res.trend = b0
            res.seasonal = s0
            if self._fixed_parameters:
                fp = self._fixed_parameters
                res.alpha = fp.get("smoothing_level", res.alpha)
                res.beta = fp.get("smoothing_trend", res.beta)
                res.gamma = fp.get("smoothing_seasonal", res.gamma)
                res.phi = fp.get("damping_trend", res.phi)
                res.level = fp.get("initial_level", res.level)
                res.trend = fp.get("initial_trend", res.trend)
                res.seasonal = fp.get("initial_seasonal", res.seasonal)

        hwfit = self._predict(
            h=0,
            smoothing_level=res.alpha,
            smoothing_trend=res.beta,
            smoothing_seasonal=res.gamma,
            damping_trend=res.phi,
            initial_level=res.level,
            initial_trend=res.trend,
            initial_seasons=res.seasonal,
            use_boxcox=use_boxcox,
            remove_bias=remove_bias,
            is_optimized=res.mask,
        )
        hwfit._results.mle_retvals = res.mle_retvals
        return hwfit

    def initial_values(
        self, initial_level=None, initial_trend=None, force=False
    ):
        """
        Compute initial values used in the exponential smoothing recursions.

        Parameters
        ----------
        initial_level : {float, None}
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        force : bool
            Force the calculation even if initial values exist.

        Returns
        -------
        initial_level : float
            The initial value used for the level component.
        initial_trend : {float, None}
            The initial value used for the trend component.
        initial_seasons : list
            The initial values used for the seasonal components.

        Notes
        -----
        Convenience function the exposes the values used to initialize the
        recursions. When optimizing parameters these are used as starting
        values.

        Method used to compute the initial value depends on when components
        are included in the model.  In a simple exponential smoothing model
        without trend or a seasonal components, the initial value is set to the
        first observation. When a trend is added, the trend is initialized
        either using y[1]/y[0], if multiplicative, or y[1]-y[0]. When the
        seasonal component is added the initialization adapts to account for
        the modified structure.
        """
        if self._initialization_method is not None and not force:
            return (
                self._initial_level,
                self._initial_trend,
                self._initial_seasonal,
            )
        y = self._y
        trend = self.trend
        seasonal = self.seasonal
        has_seasonal = self.has_seasonal
        has_trend = self.has_trend
        m = self.seasonal_periods
        l0 = initial_level
        b0 = initial_trend
        if has_seasonal:
            l0 = y[np.arange(self.nobs) % m == 0].mean() if l0 is None else l0
            if b0 is None and has_trend:
                # TODO: Fix for short m
                lead, lag = y[m : m + m], y[:m]
                if trend == "mul":
                    b0 = np.exp((np.log(lead.mean()) - np.log(lag.mean())) / m)
                else:
                    b0 = ((lead - lag) / m).mean()
            s0 = list(y[:m] / l0) if seasonal == "mul" else list(y[:m] - l0)
        elif has_trend:
            l0 = y[0] if l0 is None else l0
            if b0 is None:
                b0 = y[1] / y[0] if trend == "mul" else y[1] - y[0]
            s0 = []
        else:
            if l0 is None:
                l0 = y[0]
            b0 = None
            s0 = []

        return l0, b0, s0

    @deprecate_kwarg("smoothing_slope", "smoothing_trend")
    @deprecate_kwarg("damping_slope", "damping_trend")
    def _predict(
        self,
        h=None,
        smoothing_level=None,
        smoothing_trend=None,
        smoothing_seasonal=None,
        initial_level=None,
        initial_trend=None,
        damping_trend=None,
        initial_seasons=None,
        use_boxcox=None,
        lamda=None,
        remove_bias=None,
        is_optimized=None,
    ):
        """
        Helper prediction function

        Parameters
        ----------
        h : int, optional
            The number of time steps to forecast ahead.
        """
        # Variable renames to alpha, beta, etc as this helps with following the
        # mathematical notation in general
        alpha = smoothing_level
        beta = smoothing_trend
        gamma = smoothing_seasonal
        phi = damping_trend

        # Start in sample and out of sample predictions
        data = self.endog
        damped = self.damped_trend
        has_seasonal = self.has_seasonal
        has_trend = self.has_trend
        trend = self.trend
        seasonal = self.seasonal
        m = self.seasonal_periods
        phi = phi if damped else 1.0
        if use_boxcox == "log":
            lamda = 0.0
            y = boxcox(data, 0.0)
        elif isinstance(use_boxcox, float):
            lamda = use_boxcox
            y = boxcox(data, lamda)
        elif use_boxcox:
            y, lamda = boxcox(data)
        else:
            lamda = None
            y = data.squeeze()
            if np.ndim(y) != 1:
                raise NotImplementedError("Only 1 dimensional data supported")
        y_alpha = np.zeros((self.nobs,))
        y_gamma = np.zeros((self.nobs,))
        alphac = 1 - alpha
        y_alpha[:] = alpha * y
        betac = 1 - beta if beta is not None else 0
        gammac = 1 - gamma if gamma is not None else 0
        if has_seasonal:
            y_gamma[:] = gamma * y
        lvls = np.zeros((self.nobs + h + 1,))
        b = np.zeros((self.nobs + h + 1,))
        s = np.zeros((self.nobs + h + m + 1,))
        lvls[0] = initial_level
        b[0] = initial_trend
        s[:m] = initial_seasons
        phi_h = (
            np.cumsum(np.repeat(phi, h + 1) ** np.arange(1, h + 1 + 1))
            if damped
            else np.arange(1, h + 1 + 1)
        )
        trended = {"mul": np.multiply, "add": np.add, None: lambda l, b: l}[
            trend
        ]
        detrend = {"mul": np.divide, "add": np.subtract, None: lambda l, b: 0}[
            trend
        ]
        dampen = {"mul": np.power, "add": np.multiply, None: lambda b, phi: 0}[
            trend
        ]
        nobs = self.nobs
        if seasonal == "mul":
            for i in range(1, nobs + 1):
                lvls[i] = y_alpha[i - 1] / s[i - 1] + (
                    alphac * trended(lvls[i - 1], dampen(b[i - 1], phi))
                )
                if has_trend:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + (
                        betac * dampen(b[i - 1], phi)
                    )
                s[i + m - 1] = y_gamma[i - 1] / trended(
                    lvls[i - 1], dampen(b[i - 1], phi)
                ) + (gammac * s[i - 1])
            _trend = b[1 : nobs + 1].copy()
            season = s[m : nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if has_trend:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            s[nobs + m - 1 :] = [
                s[(nobs - 1) + j % m] for j in range(h + 1 + 1)
            ]
            fitted = trend * s[:-m]
        elif seasonal == "add":
            for i in range(1, nobs + 1):
                lvls[i] = (
                    y_alpha[i - 1]
                    - (alpha * s[i - 1])
                    + (alphac * trended(lvls[i - 1], dampen(b[i - 1], phi)))
                )
                if has_trend:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + (
                        betac * dampen(b[i - 1], phi)
                    )
                s[i + m - 1] = (
                    y_gamma[i - 1]
                    - (gamma * trended(lvls[i - 1], dampen(b[i - 1], phi)))
                    + (gammac * s[i - 1])
                )
            _trend = b[1 : nobs + 1].copy()
            season = s[m : nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if has_trend:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            s[nobs + m - 1 :] = [
                s[(nobs - 1) + j % m] for j in range(h + 1 + 1)
            ]
            fitted = trend + s[:-m]
        else:
            for i in range(1, nobs + 1):
                lvls[i] = y_alpha[i - 1] + (
                    alphac * trended(lvls[i - 1], dampen(b[i - 1], phi))
                )
                if has_trend:
                    b[i] = (beta * detrend(lvls[i], lvls[i - 1])) + (
                        betac * dampen(b[i - 1], phi)
                    )
            _trend = b[1 : nobs + 1].copy()
            season = s[m : nobs + m].copy()
            lvls[nobs:] = lvls[nobs]
            if has_trend:
                b[:nobs] = dampen(b[:nobs], phi)
                b[nobs:] = dampen(b[nobs], phi_h)
            trend = trended(lvls, b)
            fitted = trend
        level = lvls[1 : nobs + 1].copy()
        if use_boxcox or use_boxcox == "log" or isinstance(use_boxcox, float):
            fitted = inv_boxcox(fitted, lamda)
        err = fitted[: -h - 1] - data
        sse = err.T @ err
        # (s0 + gamma) + (b0 + beta) + (l0 + alpha) + phi
        k = m * has_seasonal + 2 * has_trend + 2 + 1 * damped
        aic = self.nobs * np.log(sse / self.nobs) + k * 2
        dof_eff = self.nobs - k - 3
        if dof_eff > 0:
            aicc_penalty = (2 * (k + 2) * (k + 3)) / dof_eff
        else:
            aicc_penalty = np.inf
        aicc = aic + aicc_penalty
        bic = self.nobs * np.log(sse / self.nobs) + k * np.log(self.nobs)
        resid = data - fitted[: -h - 1]
        if remove_bias:
            fitted += resid.mean()
        self.params = {
            "smoothing_level": alpha,
            "smoothing_trend": beta,
            "smoothing_seasonal": gamma,
            "damping_trend": phi if damped else np.nan,
            "initial_level": lvls[0],
            "initial_trend": b[0] / phi if phi > 0 else 0,
            "initial_seasons": s[:m],
            "use_boxcox": use_boxcox,
            "lamda": lamda,
            "remove_bias": remove_bias,
        }

        # Format parameters into a DataFrame
        codes = ["alpha", "beta", "gamma", "l.0", "b.0", "phi"]
        codes += ["s.{0}".format(i) for i in range(m)]
        idx = [
            "smoothing_level",
            "smoothing_trend",
            "smoothing_seasonal",
            "initial_level",
            "initial_trend",
            "damping_trend",
        ]
        idx += ["initial_seasons.{0}".format(i) for i in range(m)]

        formatted = [alpha, beta, gamma, lvls[0], b[0], phi]
        formatted += s[:m].tolist()
        formatted = list(map(lambda v: np.nan if v is None else v, formatted))
        formatted = np.array(formatted)
        if is_optimized is None:
            optimized = np.zeros(len(codes), dtype=bool)
        else:
            optimized = is_optimized.astype(bool)
        included = [True, has_trend, has_seasonal, True, has_trend, damped]
        included += [True] * m
        formatted = pd.DataFrame(
            [[c, f, o] for c, f, o in zip(codes, formatted, optimized)],
            columns=["name", "param", "optimized"],
            index=idx,
        )
        formatted = formatted.loc[included]

        hwfit = HoltWintersResults(
            self,
            self.params,
            fittedfcast=fitted,
            fittedvalues=fitted[: -h - 1],
            fcastvalues=fitted[-h - 1 :],
            sse=sse,
            level=level,
            trend=_trend,
            season=season,
            aic=aic,
            bic=bic,
            aicc=aicc,
            resid=resid,
            k=k,
            params_formatted=formatted,
            optimized=optimized,
        )
        return HoltWintersResultsWrapper(hwfit)


class SimpleExpSmoothing(ExponentialSmoothing):
    """
    Simple Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.

    See Also
    --------
    ExponentialSmoothing
        Exponential smoothing with trend and seasonal components.
    Holt
        Exponential smoothing with a trend component.

    Notes
    -----
    This is a full implementation of the simple exponential smoothing as
    per [1]_.  `SimpleExpSmoothing` is a restricted version of
    :class:`ExponentialSmoothing`.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(
        self,
        endog,
        initialization_method=None,  # Future: 'estimated',
        initial_level=None,
    ):
        super().__init__(
            endog,
            initialization_method=initialization_method,
            initial_level=initial_level,
        )

    def fit(
        self,
        smoothing_level=None,
        *,
        optimized=True,
        start_params=None,
        initial_level=None,
        use_brute=True,
        use_boxcox=None,
        remove_bias=False,
        method=None,
        minimize_kwargs=None,
    ):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The smoothing_level value of the simple exponential smoothing, if
            the value is set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data.
        initial_level : float, optional
            Value to use when initializing the fitted level.
        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
            "least_squares" (also "ls"). basinhopping tries multiple starting
            values in an attempt to find a global minimizer in non-convex
            problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares. The valid keywords are optimizer specific.
            Consult SciPy's documentation for the full set of options.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the simple exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        return super().fit(
            smoothing_level=smoothing_level,
            optimized=optimized,
            start_params=start_params,
            initial_level=initial_level,
            use_brute=use_brute,
            remove_bias=remove_bias,
            use_boxcox=use_boxcox,
            method=method,
            minimize_kwargs=minimize_kwargs,
        )


class Holt(ExponentialSmoothing):
    """
    Holt's Exponential Smoothing

    Parameters
    ----------
    endog : array_like
        The time series to model.
    exponential : bool, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.

    See Also
    --------
    ExponentialSmoothing
        Exponential smoothing with trend and seasonal components.
    SimpleExpSmoothing
        Basic exponential smoothing with only a level component.

    Notes
    -----
    This is a full implementation of the Holt's exponential smoothing as
    per [1]_. `Holt` is a restricted version of :class:`ExponentialSmoothing`.

    See the notebook `Exponential Smoothing
    <../examples/notebooks/generated/exponential_smoothing.html>`__
    for an overview.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    @deprecate_kwarg("damped", "damped_trend")
    def __init__(
        self,
        endog,
        exponential=False,
        damped_trend=False,
        initialization_method=None,  # Future: 'estimated',
        initial_level=None,
        initial_trend=None,
    ):
        trend = "mul" if exponential else "add"
        super().__init__(
            endog,
            trend=trend,
            damped_trend=damped_trend,
            initialization_method=initialization_method,
            initial_level=initial_level,
            initial_trend=initial_trend,
        )

    @deprecate_kwarg("smoothing_slope", "smoothing_trend")
    @deprecate_kwarg("initial_slope", "initial_trend")
    @deprecate_kwarg("damping_slope", "damping_trend")
    def fit(
        self,
        smoothing_level=None,
        smoothing_trend=None,
        *,
        damping_trend=None,
        optimized=True,
        start_params=None,
        initial_level=None,
        initial_trend=None,
        use_brute=True,
        use_boxcox=None,
        remove_bias=False,
        method=None,
        minimize_kwargs=None,
    ):
        """
        Fit the model

        Parameters
        ----------
        smoothing_level : float, optional
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend :  float, optional
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        damping_trend : float, optional
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        optimized : bool, optional
            Estimate model parameters by maximizing the log-likelihood.
        start_params : ndarray, optional
            Starting values to used when optimizing the fit.  If not provided,
            starting values are determined using a combination of grid search
            and reasonable values based on the initial values of the data.
        initial_level : float, optional
            Value to use when initializing the fitted level.

            .. deprecated:: 0.12

               Set initial_level when constructing the model

        initial_trend : float, optional
            Value to use when initializing the fitted trend.

            .. deprecated:: 0.12

               Set initial_trend when constructing the model

        use_brute : bool, optional
            Search for good starting values using a brute force (grid)
            optimizer. If False, a naive set of starting values is used.
        use_boxcox : {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? If 'log'
            then apply the log. If float then use the value as lambda.
        remove_bias : bool, optional
            Remove bias from forecast values and fitted values by enforcing
            that the average residual is equal to zero.
        method : str, default "L-BFGS-B"
            The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
            "least_squares" (also "ls"). basinhopping tries multiple starting
            values in an attempt to find a global minimizer in non-convex
            problems, and so is slower than the others.
        minimize_kwargs : dict[str, Any]
            A dictionary of keyword arguments passed to SciPy's minimize
            function if method is one of "L-BFGS-B" (default), "TNC",
            "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
            or least_squares. The valid keywords are optimizer specific.
            Consult SciPy's documentation for the full set of options.

        Returns
        -------
        HoltWintersResults
            See statsmodels.tsa.holtwinters.HoltWintersResults.

        Notes
        -----
        This is a full implementation of the Holt's exponential smoothing as
        per [1].

        References
        ----------
        [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
            and practice. OTexts, 2014.
        """
        return super().fit(
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            damping_trend=damping_trend,
            optimized=optimized,
            start_params=start_params,
            initial_level=initial_level,
            initial_trend=initial_trend,
            use_brute=use_brute,
            use_boxcox=use_boxcox,
            remove_bias=remove_bias,
            method=method,
            minimize_kwargs=minimize_kwargs,
        )
