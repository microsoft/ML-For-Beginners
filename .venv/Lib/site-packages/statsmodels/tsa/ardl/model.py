from __future__ import annotations

from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from statsmodels.compat.python import Literal

from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
    ArrayLike1D,
    ArrayLike2D,
    Float64Array,
    NDArray,
)
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
)
from statsmodels.tsa.ar_model import (
    AROrderSelectionResults,
    AutoReg,
    AutoRegResults,
    sumofsq,
)
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = [
    "ARDL",
    "ARDLResults",
    "ardl_select_order",
    "ARDLOrderSelectionResults",
    "UECM",
    "UECMResults",
    "BoundsTestResult",
]


class BoundsTestResult(NamedTuple):
    stat: float
    crit_vals: pd.DataFrame
    p_values: pd.Series
    null: str
    alternative: str

    def __repr__(self):
        return f"""\
{self.__class__.__name__}
Stat: {self.stat:0.5f}
Upper P-value: {self.p_values["upper"]:0.3g}
Lower P-value: {self.p_values["lower"]:0.3g}
Null: {self.null}
Alternative: {self.alternative}
"""


_UECMOrder = Union[None, int, Dict[Hashable, Optional[int]]]

_ARDLOrder = Union[
    None,
    int,
    _UECMOrder,
    Sequence[int],
    Dict[Hashable, Union[int, Sequence[int], None]],
]

_INT_TYPES = (int, np.integer)


def _check_order(order: int | Sequence[int] | None, causal: bool) -> bool:
    if order is None:
        return True
    if isinstance(order, (int, np.integer)):
        if int(order) < int(causal):
            raise ValueError(
                f"integer orders must be at least {int(causal)} when causal "
                f"is {causal}."
            )
        return True
    for v in order:
        if not isinstance(v, (int, np.integer)):
            raise TypeError(
                "sequence orders must contain non-negative integer values"
            )
    order = [int(v) for v in order]
    if len(set(order)) != len(order) or min(order) < 0:
        raise ValueError(
            "sequence orders must contain distinct non-negative values"
        )
    if int(causal) and min(order) < 1:
        raise ValueError(
            "sequence orders must be strictly positive when causal is True"
        )
    return True


def _format_order(
    exog: ArrayLike2D, order: _ARDLOrder, causal: bool
) -> dict[Hashable, list[int]]:
    keys: list[Hashable]
    exog_order: dict[Hashable, int | Sequence[int] | None]
    if exog is None and order in (0, None):
        return {}
    if not isinstance(exog, pd.DataFrame):
        exog = array_like(exog, "exog", ndim=2, maxdim=2)
        keys = list(range(exog.shape[1]))
    else:
        keys = [col for col in exog.columns]
    if order is None:
        exog_order = {k: None for k in keys}
    elif isinstance(order, Mapping):
        exog_order = order
        missing = set(keys).difference(order.keys())
        extra = set(order.keys()).difference(keys)
        if extra:
            msg = (
                "order dictionary contains keys for exogenous "
                "variable(s) that are not contained in exog"
            )
            msg += " Extra keys: "
            msg += ", ".join(list(sorted([str(v) for v in extra]))) + "."
            raise ValueError(msg)
        if missing:
            msg = (
                "exog contains variables that are missing from the order "
                "dictionary.  Missing keys: "
            )
            msg += ", ".join([str(k) for k in missing]) + "."
            warnings.warn(msg, SpecificationWarning, stacklevel=2)

        for key in exog_order:
            _check_order(exog_order[key], causal)
    elif isinstance(order, _INT_TYPES):
        _check_order(order, causal)
        exog_order = {k: int(order) for k in keys}
    else:
        _check_order(order, causal)
        exog_order = {k: list(order) for k in keys}
    final_order: dict[Hashable, list[int]] = {}
    for key in exog_order:
        value = exog_order[key]
        if value is None:
            continue
        assert value is not None
        if isinstance(value, int):
            final_order[key] = list(range(int(causal), value + 1))
        else:
            final_order[key] = [int(lag) for lag in value]

    return final_order


class ARDL(AutoReg):
    r"""
    Autoregressive Distributed Lag (ARDL) Model

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If
        sequence[int], uses the ``order`` for all variables. If a dict,
        applies the lags series by series. If ``exog`` is anything other
        than a DataFrame, the keys are the column index of exog (e.g., 0,
        1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    The full specification of an ARDL is

    .. math ::

       Y_t = \delta_0 + \delta_1 t + \delta_2 t^2
             + \sum_{i=1}^{s-1} \gamma_i I_{[(\mod(t,s) + 1) = i]}
             + \sum_{j=1}^p \phi_j Y_{t-j}
             + \sum_{l=1}^k \sum_{m=0}^{o_l} \beta_{l,m} X_{l, t-m}
             + Z_t \lambda
             + \epsilon_t

    where :math:`\delta_\bullet` capture trends, :math:`\gamma_\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See the notebook `Autoregressive Distributed Lag Models
    <../examples/notebooks/generated/autoregressive_distributed_lag.html>`__
    for an overview.

    See Also
    --------
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.ardl.UECM
        Unconstrained Error Correction Model estimation
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import ARDL
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> ARDL(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> ARDL(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> ARDL(data.lrm, [1, 3], exog, exog_lags, causal=True)

    A dictionary can also be used to pass specific lags to include.
    Sequences hold the specific lags to include, while integers are expanded
    to include [0, 1, ..., lag]. If causal is False, then the 0-th lag is
    excluded.

    >>> ARDL(lrm, [1, 3], exog, {"lry": [0, 1], "ibo": [0, 1, 3], "ide": 2})

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> ARDL(lrma, 3, exoga, {0: [0, 1], 1: [0, 1, 3], 2: 2})
    """

    def __init__(
        self,
        endog: Sequence[float] | pd.Series | ArrayLike2D,
        lags: int | Sequence[int] | None,
        exog: ArrayLike2D | None = None,
        order: _ARDLOrder = 0,
        trend: Literal["n", "c", "ct", "ctt"] = "c",
        *,
        fixed: ArrayLike2D | None = None,
        causal: bool = False,
        seasonal: bool = False,
        deterministic: DeterministicProcess | None = None,
        hold_back: int | None = None,
        period: int | None = None,
        missing: Literal["none", "drop", "raise"] = "none",
    ) -> None:
        self._x = np.empty((0, 0))
        self._y = np.empty((0,))

        super().__init__(
            endog,
            lags,
            trend=trend,
            seasonal=seasonal,
            exog=exog,
            hold_back=hold_back,
            period=period,
            missing=missing,
            deterministic=deterministic,
            old_names=False,
        )
        # Reset hold back which was set in AutoReg.__init__
        self._causal = bool_like(causal, "causal", strict=True)
        self.data.orig_fixed = fixed
        if fixed is not None:
            fixed_arr = array_like(fixed, "fixed", ndim=2, maxdim=2)
            if fixed_arr.shape[0] != self.data.endog.shape[0] or not np.all(
                np.isfinite(fixed_arr)
            ):
                raise ValueError(
                    "fixed must be an (nobs, m) array where nobs matches the "
                    "number of observations in the endog variable, and all"
                    "values must be finite"
                )
            if isinstance(fixed, pd.DataFrame):
                self._fixed_names = list(fixed.columns)
            else:
                self._fixed_names = [
                    f"z.{i}" for i in range(fixed_arr.shape[1])
                ]
            self._fixed = fixed_arr
        else:
            self._fixed = np.empty((self.data.endog.shape[0], 0))
            self._fixed_names = []

        self._blocks: dict[str, np.ndarray] = {}
        self._names: dict[str, Sequence[str]] = {}

        # 1. Check and update order
        self._order = self._check_order(order)
        # 2. Construct Regressors
        self._y, self._x = self._construct_regressors(hold_back)
        # 3. Construct variable names
        self._endog_name, self._exog_names = self._construct_variable_names()
        self.data.param_names = self.data.xnames = self._exog_names
        self.data.ynames = self._endog_name

        self._causal = True
        if self._order:
            min_lags = [min(val) for val in self._order.values()]
            self._causal = min(min_lags) > 0
        self._results_class = ARDLResults
        self._results_wrapper = ARDLResultsWrapper

    @property
    def fixed(self) -> NDArray | pd.DataFrame | None:
        """The fixed data used to construct the model"""
        return self.data.orig_fixed

    @property
    def causal(self) -> bool:
        """Flag indicating that the ARDL is causal"""
        return self._causal

    @property
    def ar_lags(self) -> list[int] | None:
        """The autoregressive lags included in the model"""
        return None if not self._lags else self._lags

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables included in the model"""
        return self._order

    @property
    def ardl_order(self) -> tuple[int, ...]:
        """The order of the ARDL(p,q)"""
        ar_order = 0 if not self._lags else int(max(self._lags))
        ardl_order = [ar_order]
        for lags in self._order.values():
            if lags is not None:
                ardl_order.append(int(max(lags)))
        return tuple(ardl_order)

    def _setup_regressors(self) -> None:
        """Place holder to let AutoReg init complete"""
        self._y = np.empty((self.endog.shape[0] - self._hold_back, 0))

    @staticmethod
    def _format_exog(
        exog: ArrayLike2D, order: dict[Hashable, list[int]]
    ) -> dict[Hashable, np.ndarray]:
        """Transform exogenous variables and orders to regressors"""
        if not order:
            return {}
        max_order = 0
        for val in order.values():
            if val is not None:
                max_order = max(max(val), max_order)
        if not isinstance(exog, pd.DataFrame):
            exog = array_like(exog, "exog", ndim=2, maxdim=2)
        exog_lags = {}
        for key in order:
            if order[key] is None:
                continue
            if isinstance(exog, np.ndarray):
                assert isinstance(key, int)
                col = exog[:, key]
            else:
                col = exog[key]
            lagged_col = lagmat(col, max_order, original="in")
            lags = order[key]
            exog_lags[key] = lagged_col[:, lags]
        return exog_lags

    def _check_order(self, order: _ARDLOrder) -> dict[Hashable, list[int]]:
        """Validate and standardize the model order"""
        return _format_order(self.data.orig_exog, order, self._causal)

    def _fit(
        self,
        cov_type: str = "nonrobust",
        cov_kwds: dict[str, Any] = None,
        use_t: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._x.shape[1] == 0:
            return np.empty((0,)), np.empty((0, 0)), np.empty((0, 0))
        ols_mod = OLS(self._y, self._x)
        ols_res = ols_mod.fit(
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t
        )
        cov_params = ols_res.cov_params()
        use_t = ols_res.use_t
        if cov_type == "nonrobust" and not use_t:
            nobs = self._y.shape[0]
            k = self._x.shape[1]
            scale = nobs / (nobs - k)
            cov_params /= scale

        return ols_res.params, cov_params, ols_res.normalized_cov_params

    def fit(
        self,
        *,
        cov_type: str = "nonrobust",
        cov_kwds: dict[str, Any] = None,
        use_t: bool = True,
    ) -> ARDLResults:
        """
        Estimate the model parameters.

        Parameters
        ----------
        cov_type : str
            The covariance estimator to use. The most common choices are listed
            below.  Supports all covariance estimators that are available
            in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that inference should use the Student's t
            distribution that accounts for model degree of freedom.  If False,
            uses the normal distribution. If None, defers the choice to
            the cov_type. It also removes degree of freedom corrections from
            the covariance estimator when cov_type is 'nonrobust'.

        Returns
        -------
        ARDLResults
            Estimation results.

        See Also
        --------
        statsmodels.tsa.ar_model.AutoReg
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.OLS
            Ordinary Least Squares estimation.
        statsmodels.regression.linear_model.RegressionResults
            See ``get_robustcov_results`` for a detailed list of available
            covariance estimators and options.

        Notes
        -----
        Use ``OLS`` to estimate model parameters and to estimate parameter
        covariance.
        """
        params, cov_params, norm_cov_params = self._fit(
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t
        )
        res = ARDLResults(
            self, params, cov_params, norm_cov_params, use_t=use_t
        )
        return ARDLResultsWrapper(res)

    def _construct_regressors(
        self, hold_back: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        # TODO: Missing adjustment
        self._maxlag = max(self._lags) if self._lags else 0
        _endog_reg, _endog = lagmat(
            self.data.endog, self._maxlag, original="sep"
        )
        assert isinstance(_endog, np.ndarray)
        assert isinstance(_endog_reg, np.ndarray)
        self._endog_reg, self._endog = _endog_reg, _endog
        if self._endog_reg.shape[1] != len(self._lags):
            lag_locs = [lag - 1 for lag in self._lags]
            self._endog_reg = self._endog_reg[:, lag_locs]

        orig_exog = self.data.orig_exog
        self._exog = self._format_exog(orig_exog, self._order)

        exog_maxlag = 0
        for val in self._order.values():
            exog_maxlag = max(exog_maxlag, max(val) if val is not None else 0)
        self._maxlag = max(self._maxlag, exog_maxlag)

        self._deterministic_reg = self._deterministics.in_sample()
        self._blocks = {
            "endog": self._endog_reg,
            "exog": self._exog,
            "deterministic": self._deterministic_reg,
            "fixed": self._fixed,
        }
        x = [self._deterministic_reg, self._endog_reg]
        x += [ex for ex in self._exog.values()] + [self._fixed]
        reg = np.column_stack(x)
        if hold_back is None:
            self._hold_back = int(self._maxlag)
        if self._hold_back < self._maxlag:
            raise ValueError(
                "hold_back must be >= the maximum lag of the endog and exog "
                "variables"
            )
        reg = reg[self._hold_back :]
        if reg.shape[1] > reg.shape[0]:
            raise ValueError(
                f"The number of regressors ({reg.shape[1]}) including "
                "deterministics, lags of the endog, lags of the exogenous, "
                "and fixed regressors is larger than the sample available "
                f"for estimation ({reg.shape[0]})."
            )
        return self.data.endog[self._hold_back :], reg

    def _construct_variable_names(self):
        """Construct model variables names"""
        y_name = self.data.ynames
        endog_lag_names = [f"{y_name}.L{i}" for i in self._lags]

        exog = self.data.orig_exog
        exog_names = {}
        for key in self._order:
            if isinstance(exog, np.ndarray):
                base = f"x{key}"
            else:
                base = str(key)
            lags = self._order[key]
            exog_names[key] = [f"{base}.L{lag}" for lag in lags]

        self._names = {
            "endog": endog_lag_names,
            "exog": exog_names,
            "deterministic": self._deterministic_reg.columns,
            "fixed": self._fixed_names,
        }
        x_names = list(self._deterministic_reg.columns)
        x_names += endog_lag_names
        for key in exog_names:
            x_names += exog_names[key]
        x_names += self._fixed_names
        return y_name, x_names

    def _forecasting_x(
        self,
        start: int,
        end: int,
        num_oos: int,
        exog: ArrayLike2D | None,
        exog_oos: ArrayLike2D | None,
        fixed: ArrayLike2D | None,
        fixed_oos: ArrayLike2D | None,
    ) -> np.ndarray:
        """Construct exog matrix for forecasts"""

        def pad_x(x: np.ndarray, pad: int) -> np.ndarray:
            if pad == 0:
                return x
            k = x.shape[1]
            return np.vstack([np.full((pad, k), np.nan), x])

        pad = 0 if start >= self._hold_back else self._hold_back - start
        # Shortcut if all in-sample and no new data

        if (end + 1) < self.endog.shape[0] and exog is None and fixed is None:
            adjusted_start = max(start - self._hold_back, 0)
            return pad_x(
                self._x[adjusted_start : end + 1 - self._hold_back], pad
            )

        # If anything changed, rebuild x array
        exog = self.data.exog if exog is None else np.asarray(exog)
        if exog_oos is not None:
            exog = np.vstack([exog, np.asarray(exog_oos)[:num_oos]])
        fixed = self._fixed if fixed is None else np.asarray(fixed)
        if fixed_oos is not None:
            fixed = np.vstack([fixed, np.asarray(fixed_oos)[:num_oos]])
        det = self._deterministics.in_sample()
        if num_oos:
            oos_det = self._deterministics.out_of_sample(num_oos)
            det = pd.concat([det, oos_det], axis=0)
        endog = self.data.endog
        if num_oos:
            endog = np.hstack([endog, np.full(num_oos, np.nan)])
        x = [det]
        if self._lags:
            endog_reg = lagmat(endog, max(self._lags), original="ex")
            x.append(endog_reg[:, [lag - 1 for lag in self._lags]])
        if self.ardl_order[1:]:
            if isinstance(self.data.orig_exog, pd.DataFrame):
                exog = pd.DataFrame(exog, columns=self.data.orig_exog.columns)
            exog = self._format_exog(exog, self._order)
            x.extend([np.asarray(arr) for arr in exog.values()])
        if fixed.shape[1] > 0:
            x.append(fixed)
        _x = np.column_stack(x)
        _x[: self._hold_back] = np.nan
        return _x[start:]

    def predict(
        self,
        params: ArrayLike1D,
        start: int | str | dt.datetime | pd.Timestamp | None = None,
        end: int | str | dt.datetime | pd.Timestamp | None = None,
        dynamic: bool = False,
        exog: NDArray | pd.DataFrame | None = None,
        exog_oos: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
        fixed_oos: NDArray | pd.DataFrame | None = None,
    ):
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous
            variables. Must have the same number of columns as the exog
            used when the model was created, and at least as many rows as
            the number of out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(
            params, exog, exog_oos, start, end
        )

        def check_exog(arr, name, orig, exact):
            if isinstance(orig, pd.DataFrame):
                if not isinstance(arr, pd.DataFrame):
                    raise TypeError(
                        f"{name} must be a DataFrame when the original exog "
                        "was a DataFrame"
                    )
                if sorted(arr.columns) != sorted(self.data.orig_exog.columns):
                    raise ValueError(
                        f"{name} must have the same columns as the original "
                        "exog"
                    )
            else:
                arr = array_like(arr, name, ndim=2, optional=False)
            if arr.ndim != 2 or arr.shape[1] != orig.shape[1]:
                raise ValueError(
                    f"{name} must have the same number of columns as the "
                    f"original data, {orig.shape[1]}"
                )
            if exact and arr.shape[0] != orig.shape[0]:
                raise ValueError(
                    f"{name} must have the same number of rows as the "
                    f"original data ({n})."
                )
            return arr

        n = self.data.endog.shape[0]
        if exog is not None:
            exog = check_exog(exog, "exog", self.data.orig_exog, True)
        if exog_oos is not None:
            exog_oos = check_exog(
                exog_oos, "exog_oos", self.data.orig_exog, False
            )
        if fixed is not None:
            fixed = check_exog(fixed, "fixed", self._fixed, True)
        if fixed_oos is not None:
            fixed_oos = check_exog(
                np.asarray(fixed_oos), "fixed_oos", self._fixed, False
            )
        # The maximum number of 1-step predictions that can be made,
        # which depends on the model and lags
        if self._fixed.shape[1] or not self._causal:
            max_1step = 0
        else:
            max_1step = np.inf if not self._lags else min(self._lags)
            if self._order:
                min_exog = min([min(v) for v in self._order.values()])
                max_1step = min(max_1step, min_exog)
        if num_oos > max_1step:
            if self._order and exog_oos is None:
                raise ValueError(
                    "exog_oos must be provided when out-of-sample "
                    "observations require values of the exog not in the "
                    "original sample"
                )
            elif self._order and (exog_oos.shape[0] + max_1step) < num_oos:
                raise ValueError(
                    f"exog_oos must have at least {num_oos - max_1step} "
                    f"observations to produce {num_oos} forecasts based on "
                    "the model specification."
                )

            if self._fixed.shape[1] and fixed_oos is None:
                raise ValueError(
                    "fixed_oos must be provided when predicting "
                    "out-of-sample observations"
                )
            elif self._fixed.shape[1] and fixed_oos.shape[0] < num_oos:
                raise ValueError(
                    f"fixed_oos must have at least {num_oos} observations "
                    f"to produce {num_oos} forecasts."
                )
        # Extend exog_oos if fcast is valid for horizon but no exog_oos given
        if self.exog is not None and exog_oos is None and num_oos:
            exog_oos = np.full((num_oos, self.exog.shape[1]), np.nan)
            if isinstance(self.data.orig_exog, pd.DataFrame):
                exog_oos = pd.DataFrame(
                    exog_oos, columns=self.data.orig_exog.columns
                )
        x = self._forecasting_x(
            start, end, num_oos, exog, exog_oos, fixed, fixed_oos
        )
        if dynamic is False:
            dynamic_start = end + 1 - start
        else:
            dynamic_step = self._parse_dynamic(dynamic, start)
            dynamic_start = dynamic_step
            if start < self._hold_back:
                dynamic_start = max(dynamic_start, self._hold_back - start)

        fcasts = np.full(x.shape[0], np.nan)
        fcasts[:dynamic_start] = x[:dynamic_start] @ params
        offset = self._deterministic_reg.shape[1]
        for i in range(dynamic_start, fcasts.shape[0]):
            for j, lag in enumerate(self._lags):
                loc = i - lag
                if loc >= dynamic_start:
                    val = fcasts[loc]
                else:
                    # Actual data
                    val = self.endog[start + loc]
                x[i, offset + j] = val
            fcasts[i] = x[i] @ params
        return self._wrap_prediction(fcasts, start, end + 1 + num_oos, 0)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        lags: int | Sequence[int] | None = 0,
        order: _ARDLOrder = 0,
        trend: Literal["n", "c", "ct", "ctt"] = "n",
        *,
        causal: bool = False,
        seasonal: bool = False,
        deterministic: DeterministicProcess | None = None,
        hold_back: int | None = None,
        period: int | None = None,
        missing: Literal["none", "raise"] = "none",
    ) -> ARDL | "UECM":
        """
        Construct an ARDL from a formula

        Parameters
        ----------
        formula : str
            Formula with form dependent ~ independent | fixed. See Examples
            below.
        data : DataFrame
            DataFrame containing the variables in the formula.
        lags : {int, list[int]}
            The number of lags to include in the model if an integer or the
            list of lag indices to include.  For example, [1, 4] will only
            include lags 1 and 4 while lags=4 will include lags 1, 2, 3,
            and 4.
        order : {int, sequence[int], dict}
            If int, uses lags 0, 1, ..., order  for all exog variables. If
            sequence[int], uses the ``order`` for all variables. If a dict,
            applies the lags series by series. If ``exog`` is anything other
            than a DataFrame, the keys are the column index of exog (e.g., 0,
            1, ...). If a DataFrame, keys are column names.
        causal : bool, optional
            Whether to include lag 0 of exog variables.  If True, only
            includes lags 1, 2, ...
        trend : {'n', 'c', 't', 'ct'}, optional
            The trend to include in the model:

            * 'n' - No trend.
            * 'c' - Constant only.
            * 't' - Time trend only.
            * 'ct' - Constant and time trend.

            The default is 'c'.

        seasonal : bool, optional
            Flag indicating whether to include seasonal dummies in the model.
            If seasonal is True and trend includes 'c', then the first period
            is excluded from the seasonal terms.
        deterministic : DeterministicProcess, optional
            A deterministic process.  If provided, trend and seasonal are
            ignored. A warning is raised if trend is not "n" and seasonal
            is not False.
        hold_back : {None, int}, optional
            Initial observations to exclude from the estimation sample.  If
            None, then hold_back is equal to the maximum lag in the model.
            Set to a non-zero value to produce comparable models with
            different lag length.  For example, to compare the fit of a model
            with lags=3 and lags=1, set hold_back=3 which ensures that both
            models are estimated using observations 3,...,nobs. hold_back
            must be >= the maximum lag in the model.
        period : {None, int}, optional
            The period of the data. Only used if seasonal is True. This
            parameter can be omitted if using a pandas object for endog
            that contains a recognized frequency.
        missing : {"none", "drop", "raise"}, optional
            Available options are 'none', 'drop', and 'raise'. If 'none', no
            NaN checking is done. If 'drop', any observations with NaNs are
            dropped. If 'raise', an error is raised. Default is 'none'.

        Returns
        -------
        ARDL
            The ARDL model instance

        Examples
        --------
        A simple ARDL using the Danish data

        >>> from statsmodels.datasets.danish_data import load
        >>> from statsmodels.tsa.api import ARDL
        >>> data = load().data
        >>> mod = ARDL.from_formula("lrm ~ ibo", data, 2, 2)

        Fixed regressors can be specified using a |

        >>> mod = ARDL.from_formula("lrm ~ ibo | ide", data, 2, 2)
        """
        index = data.index
        fixed_formula = None
        if "|" in formula:
            formula, fixed_formula = formula.split("|")
            fixed_formula = fixed_formula.strip()
        mod = OLS.from_formula(formula + " -1", data)
        exog = mod.data.orig_exog
        exog.index = index
        endog = mod.data.orig_endog
        endog.index = index
        if fixed_formula is not None:
            endog_name = formula.split("~")[0].strip()
            fixed_formula = f"{endog_name} ~ {fixed_formula} - 1"
            mod = OLS.from_formula(fixed_formula, data)
            fixed: pd.DataFrame | None = mod.data.orig_exog
            fixed.index = index
        else:
            fixed = None
        return cls(
            endog,
            lags,
            exog,
            order,
            trend=trend,
            fixed=fixed,
            causal=causal,
            seasonal=seasonal,
            deterministic=deterministic,
            hold_back=hold_back,
            period=period,
            missing=missing,
        )


doc = Docstring(ARDL.predict.__doc__)
_predict_params = doc.extract_parameters(
    ["start", "end", "dynamic", "exog", "exog_oos", "fixed", "fixed_oos"], 8
)


class ARDLResults(AutoRegResults):
    """
    Class to hold results from fitting an ARDL model.

    Parameters
    ----------
    model : ARDL
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    use_t : bool
        Whether use_t was set in fit
    """

    _cache = {}  # for scale setter

    def __init__(
        self,
        model: ARDL,
        params: np.ndarray,
        cov_params: np.ndarray,
        normalized_cov_params: Float64Array | None = None,
        scale: float = 1.0,
        use_t: bool = False,
    ):
        super().__init__(
            model, params, normalized_cov_params, scale, use_t=use_t
        )
        self._cache = {}
        self._params = params
        self._nobs = model.nobs
        self._n_totobs = model.endog.shape[0]
        self._df_model = model.df_model
        self._ar_lags = model.ar_lags
        self._max_lag = 0
        if self._ar_lags:
            self._max_lag = max(self._ar_lags)
        self._hold_back = self.model.hold_back
        self.cov_params_default = cov_params

    @Appender(remove_parameters(ARDL.predict.__doc__, "params"))
    def predict(
        self,
        start: int | str | dt.datetime | pd.Timestamp | None = None,
        end: int | str | dt.datetime | pd.Timestamp | None = None,
        dynamic: bool = False,
        exog: NDArray | pd.DataFrame | None = None,
        exog_oos: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
        fixed_oos: NDArray | pd.DataFrame | None = None,
    ):
        return self.model.predict(
            self._params,
            start=start,
            end=end,
            dynamic=dynamic,
            exog=exog,
            exog_oos=exog_oos,
            fixed=fixed,
            fixed_oos=fixed_oos,
        )

    def forecast(
        self,
        steps: int = 1,
        exog: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
    ) -> np.ndarray | pd.Series:
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : {int, str, datetime}, default 1
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            steps must be an integer.
        exog : array_like, optional
            Exogenous values to use out-of-sample. Must have same number of
            columns as original exog data and at least `steps` rows
        fixed : array_like, optional
            Fixed values to use out-of-sample. Must have same number of
            columns as original fixed data and at least `steps` rows

        Returns
        -------
        array_like
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.

        See Also
        --------
        ARDLResults.predict
            In- and out-of-sample predictions
        ARDLResults.get_prediction
            In- and out-of-sample predictions and confidence intervals
        """
        start = self.model.data.orig_endog.shape[0]
        if isinstance(steps, (int, np.integer)):
            end = start + steps - 1
        else:
            end = steps
        return self.predict(
            start=start, end=end, dynamic=False, exog_oos=exog, fixed_oos=fixed
        )

    def _lag_repr(self) -> np.ndarray:
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        ar_lags = self._ar_lags if self._ar_lags is not None else []
        k_ar = len(ar_lags)
        ar_params = np.zeros(self._max_lag + 1)
        ar_params[0] = 1
        offset = self.model._deterministic_reg.shape[1]
        params = self._params[offset : offset + k_ar]
        for i, lag in enumerate(ar_lags):
            ar_params[lag] = -params[i]
        return ar_params

    def get_prediction(
        self,
        start: int | str | dt.datetime | pd.Timestamp | None = None,
        end: int | str | dt.datetime | pd.Timestamp | None = None,
        dynamic: bool = False,
        exog: NDArray | pd.DataFrame | None = None,
        exog_oos: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
        fixed_oos: NDArray | pd.DataFrame | None = None,
    ) -> np.ndarray | pd.Series:
        """
        Predictions and prediction intervals

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous variable.
            Must has the same number of columns as the exog used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        PredictionResults
            Prediction results with mean and prediction intervals
        """
        mean = self.predict(
            start=start,
            end=end,
            dynamic=dynamic,
            exog=exog,
            exog_oos=exog_oos,
            fixed=fixed,
            fixed_oos=fixed_oos,
        )
        mean_var = np.full_like(mean, fill_value=self.sigma2)
        mean_var[np.isnan(mean)] = np.nan
        start = 0 if start is None else start
        end = self.model._index[-1] if end is None else end
        _, _, oos, _ = self.model._get_prediction_index(start, end)
        if oos > 0:
            ar_params = self._lag_repr()
            ma = arma2ma(ar_params, np.ones(1), lags=oos)
            mean_var[-oos:] = self.sigma2 * np.cumsum(ma**2)
        if isinstance(mean, pd.Series):
            mean_var = pd.Series(mean_var, index=mean.index)

        return PredictionResults(mean, mean_var)

    @Substitution(predict_params=_predict_params)
    def plot_predict(
        self,
        start: int | str | dt.datetime | pd.Timestamp | None = None,
        end: int | str | dt.datetime | pd.Timestamp | None = None,
        dynamic: bool = False,
        exog: NDArray | pd.DataFrame | None = None,
        exog_oos: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
        fixed_oos: NDArray | pd.DataFrame | None = None,
        alpha: float = 0.05,
        in_sample: bool = True,
        fig: "matplotlib.figure.Figure" = None,
        figsize: tuple[int, int] | None = None,
    ) -> "matplotlib.figure.Figure":
        """
        Plot in- and out-of-sample predictions

        Parameters
        ----------\n%(predict_params)s
        alpha : {float, None}
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float]
            Tuple containing the figure size values.

        Returns
        -------
        Figure
            Figure handle containing the plot.
        """
        predictions = self.get_prediction(
            start=start,
            end=end,
            dynamic=dynamic,
            exog=exog,
            exog_oos=exog_oos,
            fixed=fixed,
            fixed_oos=fixed_oos,
        )
        return self._plot_predictions(
            predictions, start, end, alpha, in_sample, fig, figsize
        )

    def summary(self, alpha: float = 0.05) -> Summary:
        """
        Summarize the Model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals.

        Returns
        -------
        Summary
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        model = self.model

        title = model.__class__.__name__ + " Model Results"
        method = "Conditional MLE"
        # get sample
        start = self._hold_back
        if self.data.dates is not None:
            dates = self.data.dates
            sample = [dates[start].strftime("%m-%d-%Y")]
            sample += ["- " + dates[-1].strftime("%m-%d-%Y")]
        else:
            sample = [str(start), str(len(self.data.orig_endog))]
        model = self.model.__class__.__name__ + str(self.model.ardl_order)
        if self.model.seasonal:
            model = "Seas. " + model

        dep_name = str(self.model.endog_names)
        top_left = [
            ("Dep. Variable:", [dep_name]),
            ("Model:", [model]),
            ("Method:", [method]),
            ("Date:", None),
            ("Time:", None),
            ("Sample:", [sample[0]]),
            ("", [sample[1]]),
        ]

        top_right = [
            ("No. Observations:", [str(len(self.model.endog))]),
            ("Log Likelihood", ["%#5.3f" % self.llf]),
            ("S.D. of innovations", ["%#5.3f" % self.sigma2**0.5]),
            ("AIC", ["%#5.3f" % self.aic]),
            ("BIC", ["%#5.3f" % self.bic]),
            ("HQIC", ["%#5.3f" % self.hqic]),
        ]

        smry = Summary()
        smry.add_table_2cols(
            self, gleft=top_left, gright=top_right, title=title
        )
        smry.add_table_params(self, alpha=alpha, use_t=False)

        return smry


class ARDLResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(ARDLResultsWrapper, ARDLResults)


class ARDLOrderSelectionResults(AROrderSelectionResults):
    """
    Results from an ARDL order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model, ics, trend, seasonal, period):
        _ics = (((0,), (0, 0, 0)),)
        super().__init__(model, _ics, trend, seasonal, period)

        def _to_dict(d):
            return d[0], dict(d[1:])

        self._aic = pd.Series(
            {v[0]: _to_dict(k) for k, v in ics.items()}, dtype=object
        )
        self._aic.index.name = self._aic.name = "AIC"
        self._aic = self._aic.sort_index()

        self._bic = pd.Series(
            {v[1]: _to_dict(k) for k, v in ics.items()}, dtype=object
        )
        self._bic.index.name = self._bic.name = "BIC"
        self._bic = self._bic.sort_index()

        self._hqic = pd.Series(
            {v[2]: _to_dict(k) for k, v in ics.items()}, dtype=object
        )
        self._hqic.index.name = self._hqic.name = "HQIC"
        self._hqic = self._hqic.sort_index()

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables in the selected model"""
        return self._model.dl_lags


def ardl_select_order(
    endog: ArrayLike1D | ArrayLike2D,
    maxlag: int,
    exog: ArrayLike2D,
    maxorder: int | dict[Hashable, int],
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    *,
    fixed: ArrayLike2D | None = None,
    causal: bool = False,
    ic: Literal["aic", "bic"] = "bic",
    glob: bool = False,
    seasonal: bool = False,
    deterministic: DeterministicProcess | None = None,
    hold_back: int | None = None,
    period: int | None = None,
    missing: Literal["none", "raise"] = "none",
) -> ARDLOrderSelectionResults:
    r"""
    ARDL order selection

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    maxlag : int
        The maximum lag to consider for the endogenous variable.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    maxorder : {int, dict}
        If int, sets a common max lag length for all exog variables. If
        a dict, then sets individual lag length. They keys are column names
        if exog is a DataFrame or column indices otherwise.
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    ic : {"aic", "bic", "hqic"}
        The information criterion to use in model selection.
    glob : bool
        Whether to consider all possible submodels of the largest model
        or only if smaller order lags must be included if larger order
        lags are.  If ``True``, the number of model considered is of the
        order 2**(maxlag + k * maxorder) assuming maxorder is an int. This
        can be very large unless k and maxorder are bot relatively small.
        If False, the number of model considered is of the order
        maxlag*maxorder**k which may also be substantial when k and maxorder
        are large.
    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Returns
    -------
    ARDLSelectionResults
        A results holder containing the selected model and the complete set
        of information criteria for all models fit.
    """
    orig_hold_back = int_like(hold_back, "hold_back", optional=True)

    def compute_ics(y, x, df):
        if x.shape[1]:
            resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            resid = y
        nobs = resid.shape[0]
        sigma2 = 1.0 / nobs * sumofsq(resid)
        llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
        res = SimpleNamespace(
            nobs=nobs, df_model=df + x.shape[1], sigma2=sigma2, llf=llf
        )

        aic = call_cached_func(ARDLResults.aic, res)
        bic = call_cached_func(ARDLResults.bic, res)
        hqic = call_cached_func(ARDLResults.hqic, res)

        return aic, bic, hqic

    base = ARDL(
        endog,
        maxlag,
        exog,
        maxorder,
        trend,
        fixed=fixed,
        causal=causal,
        seasonal=seasonal,
        deterministic=deterministic,
        hold_back=hold_back,
        period=period,
        missing=missing,
    )
    hold_back = base.hold_back
    blocks = base._blocks
    always = np.column_stack([blocks["deterministic"], blocks["fixed"]])
    always = always[hold_back:]
    select = []
    iter_orders = []
    select.append(blocks["endog"][hold_back:])
    iter_orders.append(list(range(blocks["endog"].shape[1] + 1)))
    var_names = []
    for var in blocks["exog"]:
        block = blocks["exog"][var][hold_back:]
        select.append(block)
        iter_orders.append(list(range(block.shape[1] + 1)))
        var_names.append(var)
    y = base._y
    if always.shape[1]:
        pinv_always = np.linalg.pinv(always)
        for i in range(len(select)):
            x = select[i]
            select[i] = x - always @ (pinv_always @ x)
        y = y - always @ (pinv_always @ y)

    def perm_to_tuple(keys, perm):
        if perm == ():
            d = {k: 0 for k, _ in keys if k is not None}
            return (0,) + tuple((k, v) for k, v in d.items())
        d = defaultdict(list)
        y_lags = []
        for v in perm:
            key = keys[v]
            if key[0] is None:
                y_lags.append(key[1])
            else:
                d[key[0]].append(key[1])
        d = dict(d)
        if not y_lags or y_lags == [0]:
            y_lags = 0
        else:
            y_lags = tuple(y_lags)
        for key in keys:
            if key[0] not in d and key[0] is not None:
                d[key[0]] = None
        for key in d:
            if d[key] is not None:
                d[key] = tuple(d[key])
        return (y_lags,) + tuple((k, v) for k, v in d.items())

    always_df = always.shape[1]
    ics = {}
    if glob:
        ar_lags = base.ar_lags if base.ar_lags is not None else []
        keys = [(None, i) for i in ar_lags]
        for k, v in base._order.items():
            keys += [(k, i) for i in v]
        x = np.column_stack([a for a in select])
        all_columns = list(range(x.shape[1]))
        for i in range(x.shape[1]):
            for perm in combinations(all_columns, i):
                key = perm_to_tuple(keys, perm)
                ics[key] = compute_ics(y, x[:, perm], always_df)
    else:
        for io in product(*iter_orders):
            x = np.column_stack([a[:, : io[i]] for i, a in enumerate(select)])
            key = [io[0] if io[0] else None]
            for j, val in enumerate(io[1:]):
                var = var_names[j]
                if causal:
                    key.append((var, None if val == 0 else val))
                else:
                    key.append((var, val - 1 if val - 1 >= 0 else None))
            key = tuple(key)
            ics[key] = compute_ics(y, x, always_df)
    index = {"aic": 0, "bic": 1, "hqic": 2}[ic]
    lowest = np.inf
    for key in ics:
        val = ics[key][index]
        if val < lowest:
            lowest = val
            selected_order = key
    exog_order = {k: v for k, v in selected_order[1:]}
    model = ARDL(
        endog,
        selected_order[0],
        exog,
        exog_order,
        trend,
        fixed=fixed,
        causal=causal,
        seasonal=seasonal,
        deterministic=deterministic,
        hold_back=orig_hold_back,
        period=period,
        missing=missing,
    )

    return ARDLOrderSelectionResults(model, ics, trend, seasonal, period)


lags_descr = textwrap.wrap(
    "The number of lags of the endogenous variable to include in the model. "
    "Must be at least 1.",
    71,
)
lags_param = Parameter(name="lags", type="int", desc=lags_descr)
order_descr = textwrap.wrap(
    "If int, uses lags 0, 1, ..., order  for all exog variables. If a dict, "
    "applies the lags series by series. If ``exog`` is anything other than a "
    "DataFrame, the keys are the column index of exog (e.g., 0, 1, ...). If "
    "a DataFrame, keys are column names.",
    71,
)
order_param = Parameter(name="order", type="int, dict", desc=order_descr)

from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)


fit_doc = Docstring(ARDL.fit.__doc__)
fit_doc.replace_block(
    "Returns", [Parameter("", "UECMResults", ["Estimation results."])]
)

if fit_doc._ds is not None:
    see_also = fit_doc._ds["See Also"]
    see_also.insert(
        0,
        (
            [("statsmodels.tsa.ardl.ARDL", None)],
            ["Autoregressive distributed lag model estimation"],
        ),
    )
    fit_doc.replace_block("See Also", see_also)


class UECM(ARDL):
    r"""
    Unconstrained Error Correlation Model(UECM)

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {int, list[int]}
        The number of lags of the endogenous variable to include in the
        model. Must be at least 1.
    exog : array_like
        Exogenous variables to include in the model. Either a DataFrame or
        an 2-d array-like structure that can be converted to a NumPy array.
    order : {int, sequence[int], dict}
        If int, uses lags 0, 1, ..., order  for all exog variables. If a
        dict, applies the lags series by series. If ``exog`` is anything
        other than a DataFrame, the keys are the column index of exog
        (e.g., 0, 1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no NaN
        checking is done. If 'drop', any observations with NaNs are dropped.
        If 'raise', an error is raised. Default is 'none'.

    Notes
    -----
    The full specification of an UECM is

    .. math ::

       \Delta Y_t = \delta_0 + \delta_1 t + \delta_2 t^2
             + \sum_{i=1}^{s-1} \gamma_i I_{[(\mod(t,s) + 1) = i]}
             + \lambda_0 Y_{t-1} + \lambda_1 X_{1,t-1} + \ldots
             + \lambda_{k} X_{k,t-1}
             + \sum_{j=1}^{p-1} \phi_j \Delta Y_{t-j}
             + \sum_{l=1}^k \sum_{m=0}^{o_l-1} \beta_{l,m} \Delta X_{l, t-m}
             + Z_t \lambda
             + \epsilon_t

    where :math:`\delta_\bullet` capture trends, :math:`\gamma_\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See Also
    --------
    statsmodels.tsa.ardl.ARDL
        Autoregressive distributed lag model estimation
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    >>> from statsmodels.tsa.api import UECM
    >>> from statsmodels.datasets import danish_data
    >>> data = danish_data.load_pandas().data
    >>> lrm = data.lrm
    >>> exog = data[["lry", "ibo", "ide"]]

    A basic model where all variables have 3 lags included

    >>> UECM(data.lrm, 3, data[["lry", "ibo", "ide"]], 3)

    A dictionary can be used to pass custom lag orders

    >>> UECM(data.lrm, [1, 3], exog, {"lry": 1, "ibo": 3, "ide": 2})

    Setting causal removes the 0-th lag from the exogenous variables

    >>> exog_lags = {"lry": 1, "ibo": 3, "ide": 2}
    >>> UECM(data.lrm, 3, exog, exog_lags, causal=True)

    When using NumPy arrays, the dictionary keys are the column index.

    >>> import numpy as np
    >>> lrma = np.asarray(lrm)
    >>> exoga = np.asarray(exog)
    >>> UECM(lrma, 3, exoga, {0: 1, 1: 3, 2: 2})
    """

    def __init__(
        self,
        endog: ArrayLike1D | ArrayLike2D,
        lags: int | None,
        exog: ArrayLike2D | None = None,
        order: _UECMOrder = 0,
        trend: Literal["n", "c", "ct", "ctt"] = "c",
        *,
        fixed: ArrayLike2D | None = None,
        causal: bool = False,
        seasonal: bool = False,
        deterministic: DeterministicProcess | None = None,
        hold_back: int | None = None,
        period: int | None = None,
        missing: Literal["none", "drop", "raise"] = "none",
    ) -> None:
        super().__init__(
            endog,
            lags,
            exog,
            order,
            trend=trend,
            fixed=fixed,
            seasonal=seasonal,
            causal=causal,
            hold_back=hold_back,
            period=period,
            missing=missing,
            deterministic=deterministic,
        )
        self._results_class = UECMResults
        self._results_wrapper = UECMResultsWrapper

    def _check_lags(
        self, lags: int | Sequence[int] | None, hold_back: int | None
    ) -> tuple[list[int], int]:
        """Check lags value conforms to requirement"""
        if not (isinstance(lags, _INT_TYPES) or lags is None):
            raise TypeError("lags must be an integer or None")
        return super()._check_lags(lags, hold_back)

    def _check_order(self, order: _ARDLOrder):
        """Check order conforms to requirement"""
        if isinstance(order, Mapping):
            for k, v in order.items():
                if not isinstance(v, _INT_TYPES) and v is not None:
                    raise TypeError(
                        "order values must be positive integers or None"
                    )
        elif not (isinstance(order, _INT_TYPES) or order is None):
            raise TypeError(
                "order must be None, a positive integer, or a dict "
                "containing positive integers or None"
            )
        # TODO: Check order is >= 1
        order = super()._check_order(order)
        if not order:
            raise ValueError(
                "Model must contain at least one exogenous variable"
            )
        for key, val in order.items():
            if val == [0]:
                raise ValueError(
                    "All included exog variables must have a lag length >= 1"
                )
        return order

    def _construct_variable_names(self):
        """Construct model variables names"""
        endog = self.data.orig_endog
        if isinstance(endog, pd.Series):
            y_base = endog.name or "y"
        elif isinstance(endog, pd.DataFrame):
            y_base = endog.squeeze().name or "y"
        else:
            y_base = "y"
        y_name = f"D.{y_base}"
        # 1. Deterministics
        x_names = list(self._deterministic_reg.columns)
        # 2. Levels
        x_names.append(f"{y_base}.L1")
        orig_exog = self.data.orig_exog
        exog_pandas = isinstance(orig_exog, pd.DataFrame)
        dexog_names = []
        for key, val in self._order.items():
            if val is not None:
                if exog_pandas:
                    x_name = f"{key}.L1"
                else:
                    x_name = f"x{key}.L1"
                x_names.append(x_name)
                lag_base = x_name[:-1]
                for lag in val[:-1]:
                    dexog_names.append(f"D.{lag_base}{lag}")
        # 3. Lagged endog
        y_lags = max(self._lags) if self._lags else 0
        dendog_names = [f"{y_name}.L{lag}" for lag in range(1, y_lags)]
        x_names.extend(dendog_names)
        x_names.extend(dexog_names)
        x_names.extend(self._fixed_names)
        return y_name, x_names

    def _construct_regressors(
        self, hold_back: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct and format model regressors"""
        # 1. Endogenous and endogenous lags
        self._maxlag = max(self._lags) if self._lags else 0
        dendog = np.full_like(self.data.endog, np.nan)
        dendog[1:] = np.diff(self.data.endog, axis=0)
        dlag = max(0, self._maxlag - 1)
        self._endog_reg, self._endog = lagmat(dendog, dlag, original="sep")
        # 2. Deterministics
        self._deterministic_reg = self._deterministics.in_sample()
        # 3. Levels
        orig_exog = self.data.orig_exog
        exog_pandas = isinstance(orig_exog, pd.DataFrame)
        lvl = np.full_like(self.data.endog, np.nan)
        lvl[1:] = self.data.endog[:-1]
        lvls = [lvl.copy()]
        for key, val in self._order.items():
            if val is not None:
                if exog_pandas:
                    loc = orig_exog.columns.get_loc(key)
                else:
                    loc = key
                lvl[1:] = self.data.exog[:-1, loc]
                lvls.append(lvl.copy())
        self._levels = np.column_stack(lvls)

        # 4. exog Lags
        if exog_pandas:
            dexog = orig_exog.diff()
        else:
            dexog = np.full_like(self.data.exog, np.nan)
            dexog[1:] = np.diff(orig_exog, axis=0)
        adj_order = {}
        for key, val in self._order.items():
            val = None if (val is None or val == [1]) else val[:-1]
            adj_order[key] = val
        self._exog = self._format_exog(dexog, adj_order)

        self._blocks = {
            "deterministic": self._deterministic_reg,
            "levels": self._levels,
            "endog": self._endog_reg,
            "exog": self._exog,
            "fixed": self._fixed,
        }
        blocks = [self._endog]
        for key, val in self._blocks.items():
            if key != "exog":
                blocks.append(np.asarray(val))
            else:
                for subval in val.values():
                    blocks.append(np.asarray(subval))
        y = blocks[0]
        reg = np.column_stack(blocks[1:])
        exog_maxlag = 0
        for val in self._order.values():
            exog_maxlag = max(exog_maxlag, max(val) if val is not None else 0)
        self._maxlag = max(self._maxlag, exog_maxlag)
        # Must be at least 1 since the endog is differenced
        self._maxlag = max(self._maxlag, 1)
        if hold_back is None:
            self._hold_back = int(self._maxlag)
        if self._hold_back < self._maxlag:
            raise ValueError(
                "hold_back must be >= the maximum lag of the endog and exog "
                "variables"
            )
        reg = reg[self._hold_back :]
        if reg.shape[1] > reg.shape[0]:
            raise ValueError(
                f"The number of regressors ({reg.shape[1]}) including "
                "deterministics, lags of the endog, lags of the exogenous, "
                "and fixed regressors is larger than the sample available "
                f"for estimation ({reg.shape[0]})."
            )
        return np.squeeze(y)[self._hold_back :], reg

    @Appender(str(fit_doc))
    def fit(
        self,
        *,
        cov_type: str = "nonrobust",
        cov_kwds: dict[str, Any] = None,
        use_t: bool = True,
    ) -> UECMResults:
        params, cov_params, norm_cov_params = self._fit(
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t
        )
        res = UECMResults(
            self, params, cov_params, norm_cov_params, use_t=use_t
        )
        return UECMResultsWrapper(res)

    @classmethod
    def from_ardl(
        cls, ardl: ARDL, missing: Literal["none", "drop", "raise"] = "none"
    ):
        """
        Construct a UECM from an ARDL model

        Parameters
        ----------
        ardl : ARDL
            The ARDL model instance
        missing : {"none", "drop", "raise"}, default "none"
            How to treat missing observations.

        Returns
        -------
        UECM
            The UECM model instance

        Notes
        -----
        The lag requirements for a UECM are stricter than for an ARDL.
        Any variable that is included in the UECM must have a lag length
        of at least 1. Additionally, the included lags must be contiguous
        starting at 0 if non-causal or 1 if causal.
        """
        err = (
            "UECM can only be created from ARDL models that include all "
            "{var_typ} lags up to the maximum lag in the model."
        )
        uecm_lags = {}
        dl_lags = ardl.dl_lags
        for key, val in dl_lags.items():
            max_val = max(val)
            if len(dl_lags[key]) < (max_val + int(not ardl.causal)):
                raise ValueError(err.format(var_typ="exogenous"))
            uecm_lags[key] = max_val
        if ardl.ar_lags is None:
            ar_lags = None
        else:
            max_val = max(ardl.ar_lags)
            if len(ardl.ar_lags) != max_val:
                raise ValueError(err.format(var_typ="endogenous"))
            ar_lags = max_val

        return cls(
            ardl.data.orig_endog,
            ar_lags,
            ardl.data.orig_exog,
            uecm_lags,
            trend=ardl.trend,
            fixed=ardl.fixed,
            seasonal=ardl.seasonal,
            hold_back=ardl.hold_back,
            period=ardl.period,
            causal=ardl.causal,
            missing=missing,
            deterministic=ardl.deterministic,
        )

    def predict(
        self,
        params: ArrayLike1D,
        start: int | str | dt.datetime | pd.Timestamp | None = None,
        end: int | str | dt.datetime | pd.Timestamp | None = None,
        dynamic: bool = False,
        exog: NDArray | pd.DataFrame | None = None,
        exog_oos: NDArray | pd.DataFrame | None = None,
        fixed: NDArray | pd.DataFrame | None = None,
        fixed_oos: NDArray | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        In-sample prediction and out-of-sample forecasting.

        Parameters
        ----------
        params : array_like
            The fitted model parameters.
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out-of-sample prediction. Default is the last observation in
            the sample. Unlike standard python slices, end is inclusive so
            that all the predictions [start, start+1, ..., end-1, end] are
            returned.
        dynamic : {bool, int, str, datetime, Timestamp}, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Prior to this observation, true endogenous values
            will be used for prediction; starting with this observation and
            continuing through the end of prediction, forecasted endogenous
            values will be used instead. Datetime-like objects are not
            interpreted as offsets. They are instead used to find the index
            location of `dynamic` which is then used to to compute the offset.
        exog : array_like
            A replacement exogenous array.  Must have the same shape as the
            exogenous data array used when the model was created.
        exog_oos : array_like
            An array containing out-of-sample values of the exogenous
            variables. Must have the same number of columns as the exog
            used when the model was created, and at least as many rows as
            the number of out-of-sample forecasts.
        fixed : array_like
            A replacement fixed array.  Must have the same shape as the
            fixed data array used when the model was created.
        fixed_oos : array_like
            An array containing out-of-sample values of the fixed variables.
            Must have the same number of columns as the fixed used when the
            model was created, and at least as many rows as the number of
            out-of-sample forecasts.

        Returns
        -------
        predictions : {ndarray, Series}
            Array of out of in-sample predictions and / or out-of-sample
            forecasts.
        """
        if dynamic is not False:
            raise NotImplementedError("dynamic forecasts are not supported")
        params, exog, exog_oos, start, end, num_oos = self._prepare_prediction(
            params, exog, exog_oos, start, end
        )
        if num_oos != 0:
            raise NotImplementedError(
                "Out-of-sample forecasts are not supported"
            )
        pred = np.full(self.endog.shape[0], np.nan)
        pred[-self._x.shape[0] :] = self._x @ params
        return pred[start : end + 1]

    @classmethod
    @Appender(from_formula_doc.__str__().replace("ARDL", "UECM"))
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        lags: int | Sequence[int] | None = 0,
        order: _ARDLOrder = 0,
        trend: Literal["n", "c", "ct", "ctt"] = "n",
        *,
        causal: bool = False,
        seasonal: bool = False,
        deterministic: DeterministicProcess | None = None,
        hold_back: int | None = None,
        period: int | None = None,
        missing: Literal["none", "raise"] = "none",
    ) -> UECM:
        return super().from_formula(
            formula,
            data,
            lags,
            order,
            trend,
            causal=causal,
            seasonal=seasonal,
            deterministic=deterministic,
            hold_back=hold_back,
            period=period,
            missing=missing,
        )


class UECMResults(ARDLResults):
    """
    Class to hold results from fitting an UECM model.

    Parameters
    ----------
    model : UECM
        Reference to the model that is fit.
    params : ndarray
        The fitted parameters from the AR Model.
    cov_params : ndarray
        The estimated covariance matrix of the model parameters.
    normalized_cov_params : ndarray
        The array inv(dot(x.T,x)) where x contains the regressors in the
        model.
    scale : float, optional
        An estimate of the scale of the model.
    """

    _cache: dict[str, Any] = {}  # for scale setter

    def _ci_wrap(
        self, val: np.ndarray, name: str = ""
    ) -> NDArray | pd.Series | pd.DataFrame:
        if not isinstance(self.model.data, PandasData):
            return val
        ndet = self.model._blocks["deterministic"].shape[1]
        nlvl = self.model._blocks["levels"].shape[1]
        lbls = self.model.exog_names[: (ndet + nlvl)]
        for i in range(ndet, ndet + nlvl):
            lbl = lbls[i]
            if lbl.endswith(".L1"):
                lbls[i] = lbl[:-3]
        if val.ndim == 2:
            return pd.DataFrame(val, columns=lbls, index=lbls)
        return pd.Series(val, index=lbls, name=name)

    @cache_readonly
    def ci_params(self) -> np.ndarray | pd.Series:
        """Parameters of normalized cointegrating relationship"""
        ndet = self.model._blocks["deterministic"].shape[1]
        nlvl = self.model._blocks["levels"].shape[1]
        base = np.asarray(self.params)[ndet]
        return self._ci_wrap(self.params[: ndet + nlvl] / base, "ci_params")

    @cache_readonly
    def ci_bse(self) -> np.ndarray | pd.Series:
        """Standard Errors of normalized cointegrating relationship"""
        bse = np.sqrt(np.diag(self.ci_cov_params()))
        return self._ci_wrap(bse, "ci_bse")

    @cache_readonly
    def ci_tvalues(self) -> np.ndarray | pd.Series:
        """T-values of normalized cointegrating relationship"""
        ndet = self.model._blocks["deterministic"].shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tvalues = np.asarray(self.ci_params) / np.asarray(self.ci_bse)
            tvalues[ndet] = np.nan
        return self._ci_wrap(tvalues, "ci_tvalues")

    @cache_readonly
    def ci_pvalues(self) -> np.ndarray | pd.Series:
        """P-values of normalized cointegrating relationship"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.ci_tvalues)))
        return self._ci_wrap(pvalues, "ci_pvalues")

    def ci_conf_int(self, alpha: float = 0.05) -> Float64Array | pd.DataFrame:
        alpha = float_like(alpha, "alpha")

        if self.use_t:
            q = stats.t(self.df_resid).ppf(1 - alpha / 2)
        else:
            q = stats.norm().ppf(1 - alpha / 2)
        p = self.ci_params
        se = self.ci_bse
        out = [p - q * se, p + q * se]
        if not isinstance(p, pd.Series):
            return np.column_stack(out)

        df = pd.concat(out, axis=1)
        df.columns = ["lower", "upper"]

        return df

    def ci_summary(self, alpha: float = 0.05) -> Summary:
        def _ci(alpha=alpha):
            return np.asarray(self.ci_conf_int(alpha))

        smry = Summary()
        ndet = self.model._blocks["deterministic"].shape[1]
        nlvl = self.model._blocks["levels"].shape[1]
        exog_names = list(self.model.exog_names)[: (ndet + nlvl)]

        model = SimpleNamespace(
            endog_names=self.model.endog_names, exog_names=exog_names
        )
        data = SimpleNamespace(
            params=self.ci_params,
            bse=self.ci_bse,
            tvalues=self.ci_tvalues,
            pvalues=self.ci_pvalues,
            conf_int=_ci,
            model=model,
        )
        tab = summary_params(data)
        tab.title = "Cointegrating Vector"
        smry.tables.append(tab)

        return smry

    @cache_readonly
    def ci_resids(self) -> np.ndarray | pd.Series:
        d = self.model._blocks["deterministic"]
        exog = self.model.data.orig_exog
        is_pandas = isinstance(exog, pd.DataFrame)
        exog = exog if is_pandas else self.model.exog
        cols = [np.asarray(d), self.model.endog]
        for key, value in self.model.dl_lags.items():
            if value is not None:
                if is_pandas:
                    cols.append(np.asarray(exog[key]))
                else:
                    cols.append(exog[:, key])
        ci_x = np.column_stack(cols)
        resids = ci_x @ self.ci_params
        if not isinstance(self.model.data, PandasData):
            return resids
        index = self.model.data.orig_endog.index
        return pd.Series(resids, index=index, name="ci_resids")

    def ci_cov_params(self) -> Float64Array | pd.DataFrame:
        """Covariance of normalized of cointegrating relationship"""
        ndet = self.model._blocks["deterministic"].shape[1]
        nlvl = self.model._blocks["levels"].shape[1]
        loc = list(range(ndet + nlvl))
        cov = self.cov_params()
        cov_a = np.asarray(cov)
        ci_cov = cov_a[np.ix_(loc, loc)]
        m = ci_cov.shape[0]
        params = np.asarray(self.params)[: ndet + nlvl]
        base = params[ndet]
        d = np.zeros((m, m))
        for i in range(m):
            if i == ndet:
                continue
            d[i, i] = 1 / base
            d[i, ndet] = -params[i] / (base**2)
        ci_cov = d @ ci_cov @ d.T
        return self._ci_wrap(ci_cov)

    def _lag_repr(self):
        """Returns poly repr of an AR, (1  -phi1 L -phi2 L^2-...)"""
        # TODO

    def bounds_test(
        self,
        case: Literal[1, 2, 3, 4, 5],
        cov_type: str = "nonrobust",
        cov_kwds: dict[str, Any] = None,
        use_t: bool = True,
        asymptotic: bool = True,
        nsim: int = 100_000,
        seed: int
        | Sequence[int]
        | np.random.RandomState
        | np.random.Generator
        | None = None,
    ):
        r"""
        Cointegration bounds test of Pesaran, Shin, and Smith

        Parameters
        ----------
        case : {1, 2, 3, 4, 5}
            One of the cases covered in the PSS test.
        cov_type : str
            The covariance estimator to use. The asymptotic distribution of
            the PSS test has only been established in the homoskedastic case,
            which is the default.

            The most common choices are listed below.  Supports all covariance
            estimators that are available in ``OLS.fit``.

            * 'nonrobust' - The class OLS covariance estimator that assumes
              homoskedasticity.
            * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
              (or Eiker-Huber-White) covariance estimator. `HC0` is the
              standard implementation.  The other make corrections to improve
              the finite sample performance of the heteroskedasticity robust
              covariance estimator.
            * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
              estimation. Supports cov_kwds.

              - `maxlags` integer (required) : number of lags to use.
              - `kernel` callable or str (optional) : kernel
                  currently available kernels are ['bartlett', 'uniform'],
                  default is Bartlett.
              - `use_correction` bool (optional) : If true, use small sample
                  correction.
        cov_kwds : dict, optional
            A dictionary of keyword arguments to pass to the covariance
            estimator. `nonrobust` and `HC#` do not support cov_kwds.
        use_t : bool, optional
            A flag indicating that small-sample corrections should be applied
            to the covariance estimator.
        asymptotic : bool
            Flag indicating whether to use asymptotic critical values which
            were computed by simulation (True, default) or to simulate a
            sample-size specific set of critical values. Tables are only
            available for up to 10 components in the cointegrating
            relationship, so if more variables are included then simulation
            is always used. The simulation computed the test statistic under
            and assumption that the residuals are homoskedastic.
        nsim : int
            Number of simulations to run when computing exact critical values.
            Only used if ``asymptotic`` is ``True``.
        seed : {None, int, sequence[int], RandomState, Generator}, optional
            Seed to use when simulating critical values. Must be provided if
            reproducible critical value and p-values are required when
            ``asymptotic`` is ``False``.

        Returns
        -------
        BoundsTestResult
            Named tuple containing ``stat``, ``crit_vals``, ``p_values``,
            ``null` and ``alternative``. The statistic is the F-type
            test statistic favored in PSS.

        Notes
        -----
        The PSS bounds test has 5 cases which test the coefficients on the
        level terms in the model

        .. math::

           \Delta Y_{t}=\delta_{0} + \delta_{1}t + Z_{t-1}\beta
                        + \sum_{j=0}^{P}\Delta X_{t-j}\Gamma + \epsilon_{t}

        where :math:`Z_{t-1}` contains both :math:`Y_{t-1}` and
        :math:`X_{t-1}`.

        The cases determine which deterministic terms are included in the
        model and which are tested as part of the test.

        Cases:

        1. No deterministic terms
        2. Constant included in both the model and the test
        3. Constant included in the model but not in the test
        4. Constant and trend included in the model, only trend included in
           the test
        5. Constant and trend included in the model, neither included in the
           test

        The test statistic is a Wald-type quadratic form test that all of the
        coefficients in :math:`\beta` are 0 along with any included
        deterministic terms, which depends on the case. The statistic returned
        is an F-type test statistic which is the standard quadratic form test
        statistic divided by the number of restrictions.

        References
        ----------
        .. [*] Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing
           approaches to the analysis of level relationships. Journal of
           applied econometrics, 16(3), 289-326.
        """
        model = self.model
        trend: Literal["n", "c", "ct"]
        if case == 1:
            trend = "n"
        elif case in (2, 3):
            trend = "c"
        else:
            trend = "ct"
        order = {key: max(val) for key, val in model._order.items()}
        uecm = UECM(
            model.data.endog,
            max(model.ar_lags),
            model.data.orig_exog,
            order=order,
            causal=model.causal,
            trend=trend,
        )
        res = uecm.fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        cov = res.cov_params()
        nvar = len(res.model.ardl_order)
        if case == 1:
            rest = np.arange(nvar)
        elif case == 2:
            rest = np.arange(nvar + 1)
        elif case == 3:
            rest = np.arange(1, nvar + 1)
        elif case == 4:
            rest = np.arange(1, nvar + 2)
        elif case == 5:
            rest = np.arange(2, nvar + 2)
        r = np.zeros((rest.shape[0], cov.shape[1]))
        for i, loc in enumerate(rest):
            r[i, loc] = 1
        vcv = r @ cov @ r.T
        coef = r @ res.params
        stat = coef.T @ np.linalg.inv(vcv) @ coef / r.shape[0]
        k = nvar
        if asymptotic and k <= 10:
            cv = pss_critical_values.crit_vals
            key = (k, case)
            upper = cv[key + (True,)]
            lower = cv[key + (False,)]
            crit_vals = pd.DataFrame(
                {"lower": lower, "upper": upper},
                index=pss_critical_values.crit_percentiles,
            )
            crit_vals.index.name = "percentile"
            p_values = pd.Series(
                {
                    "lower": _pss_pvalue(stat, k, case, False),
                    "upper": _pss_pvalue(stat, k, case, True),
                }
            )
        else:
            nobs = res.resid.shape[0]
            crit_vals, p_values = _pss_simulate(
                stat, k, case, nobs=nobs, nsim=nsim, seed=seed
            )

        return BoundsTestResult(
            stat,
            crit_vals,
            p_values,
            "No Cointegration",
            "Possible Cointegration",
        )


def _pss_pvalue(stat: float, k: int, case: int, i1: bool) -> float:
    key = (k, case, i1)
    large_p = pss_critical_values.large_p[key]
    small_p = pss_critical_values.small_p[key]
    threshold = pss_critical_values.stat_star[key]
    log_stat = np.log(stat)
    p = small_p if stat > threshold else large_p
    x = [log_stat**i for i in range(len(p))]
    return 1 - stats.norm.cdf(x @ np.array(p))


def _pss_simulate(
    stat: float,
    k: int,
    case: Literal[1, 2, 3, 4, 5],
    nobs: int,
    nsim: int,
    seed: int
    | Sequence[int]
    | np.random.RandomState
    | np.random.Generator
    | None,
) -> tuple[pd.DataFrame, pd.Series]:
    rs: np.random.RandomState | np.random.Generator
    if not isinstance(seed, np.random.RandomState):
        rs = np.random.default_rng(seed)
    else:
        assert isinstance(seed, np.random.RandomState)
        rs = seed

    def _vectorized_ols_resid(rhs, lhs):
        rhs_t = np.transpose(rhs, [0, 2, 1])
        xpx = np.matmul(rhs_t, rhs)
        xpy = np.matmul(rhs_t, lhs)
        b = np.linalg.solve(xpx, xpy)
        return np.squeeze(lhs - np.matmul(rhs, b))

    block_size = 100_000_000 // (8 * nobs * k)
    remaining = nsim
    loc = 0
    f_upper = np.empty(nsim)
    f_lower = np.empty(nsim)
    while remaining > 0:
        to_do = min(remaining, block_size)
        e = rs.standard_normal((to_do, nobs + 1, k))

        y = np.cumsum(e[:, :, :1], axis=1)
        x_upper = np.cumsum(e[:, :, 1:], axis=1)
        x_lower = e[:, :, 1:]
        lhs = np.diff(y, axis=1)
        if case in (2, 3):
            rhs = np.empty((to_do, nobs, k + 1))
            rhs[:, :, -1] = 1
        elif case in (4, 5):
            rhs = np.empty((to_do, nobs, k + 2))
            rhs[:, :, -2] = np.arange(nobs, dtype=float)
            rhs[:, :, -1] = 1
        else:
            rhs = np.empty((to_do, nobs, k))
        rhs[:, :, :1] = y[:, :-1]
        rhs[:, :, 1:k] = x_upper[:, :-1]

        u = _vectorized_ols_resid(rhs, lhs)
        df = rhs.shape[1] - rhs.shape[2]
        s2 = (u**2).sum(1) / df

        if case in (3, 4):
            rhs_r = rhs[:, :, -1:]
        elif case == 5:  # case 5
            rhs_r = rhs[:, :, -2:]
        if case in (3, 4, 5):
            ur = _vectorized_ols_resid(rhs_r, lhs)
            nrest = rhs.shape[-1] - rhs_r.shape[-1]
        else:
            ur = np.squeeze(lhs)
            nrest = rhs.shape[-1]

        f = ((ur**2).sum(1) - (u**2).sum(1)) / nrest
        f /= s2
        f_upper[loc : loc + to_do] = f

        # Lower
        rhs[:, :, 1:k] = x_lower[:, :-1]
        u = _vectorized_ols_resid(rhs, lhs)
        s2 = (u**2).sum(1) / df

        if case in (3, 4):
            rhs_r = rhs[:, :, -1:]
        elif case == 5:  # case 5
            rhs_r = rhs[:, :, -2:]
        if case in (3, 4, 5):
            ur = _vectorized_ols_resid(rhs_r, lhs)
            nrest = rhs.shape[-1] - rhs_r.shape[-1]
        else:
            ur = np.squeeze(lhs)
            nrest = rhs.shape[-1]

        f = ((ur**2).sum(1) - (u**2).sum(1)) / nrest
        f /= s2
        f_lower[loc : loc + to_do] = f

        loc += to_do
        remaining -= to_do

    crit_percentiles = pss_critical_values.crit_percentiles
    crit_vals = pd.DataFrame(
        {
            "lower": np.percentile(f_lower, crit_percentiles),
            "upper": np.percentile(f_upper, crit_percentiles),
        },
        index=crit_percentiles,
    )
    crit_vals.index.name = "percentile"
    p_values = pd.Series(
        {"lower": (stat < f_lower).mean(), "upper": (stat < f_upper).mean()}
    )
    return crit_vals, p_values


class UECMResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        tsa_model.TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(UECMResultsWrapper, UECMResults)
