r"""
Implementation of the Theta forecasting method of

Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition
approach to forecasting. International journal of forecasting, 16(4), 521-530.

and updates in

Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method. International
Journal of Forecasting, 19(2), 287-290.

Fioruci, J. A., Pellegrini, T. R., Louzada, F., & Petropoulos, F. (2015).
The optimized theta method. arXiv preprint arXiv:1503.03529.
"""
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa.deterministic import DeterministicTerm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import (
    ExponentialSmoothing,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend, freq_to_period

if TYPE_CHECKING:
    import matplotlib.figure


def extend_index(steps: int, index: pd.Index) -> pd.Index:
    return DeterministicTerm._extend_index(index, steps)


class ThetaModel:
    r"""
    The Theta forecasting model of Assimakopoulos and Nikolopoulos (2000)

    Parameters
    ----------
    endog : array_like, 1d
        The data to forecast.
    period : int, default None
        The period of the data that is used in the seasonality test and
        adjustment. If None then the period is determined from y's index,
        if available.
    deseasonalize : bool, default True
        A flag indicating whether the deseasonalize the data. If True and
        use_test is True, the data is only deseasonalized if the null of no
        seasonal component is rejected.
    use_test : bool, default True
        A flag indicating whether test the period-th autocorrelation. If this
        test rejects using a size of 10%, then decomposition is used. Set to
        False to skip the test.
    method : {"auto", "additive", "multiplicative"}, default "auto"
        The model used for the seasonal decomposition. "auto" uses a
        multiplicative if y is non-negative and all estimated seasonal
        components are positive. If either of these conditions is False,
        then it uses an additive decomposition.
    difference : bool, default False
        A flag indicating to difference the data before testing for
        seasonality.

    See Also
    --------
    statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
        Exponential smoothing parameter estimation and forecasting
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA parameter estimation and forecasting

    Notes
    -----
    The Theta model forecasts the future as a weighted combination of two
    Theta lines.  This class supports combinations of models with two
    thetas: 0 and a user-specified choice (default 2). The forecasts are
    then

    .. math::

       \hat{X}_{T+h|T} = \frac{\theta-1}{\theta} b_0
                         \left[h - 1 + \frac{1}{\alpha}
                         - \frac{(1-\alpha)^T}{\alpha} \right]
                         + \tilde{X}_{T+h|T}

    where :math:`\tilde{X}_{T+h|T}` is the SES forecast of the endogenous
    variable using the parameter :math:`\alpha`. :math:`b_0` is the
    slope of a time trend line fitted to X using the terms 0, 1, ..., T-1.

    The model is estimated in steps:

    1. Test for seasonality
    2. Deseasonalize if seasonality detected
    3. Estimate :math:`\alpha` by fitting a SES model to the data and
       :math:`b_0` by OLS.
    4. Forecast the series
    5. Reseasonalize if the data was deseasonalized.

    The seasonality test examines where the autocorrelation at the
    seasonal period is different from zero. The seasonality is then
    removed using a seasonal decomposition with a multiplicative trend.
    If the seasonality estimate is non-positive then an additive trend
    is used instead. The default deseasonalizing method can be changed
    using the options.

    References
    ----------
    .. [1] Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a
       decomposition approach to forecasting. International Journal of
       Forecasting, 16(4), 521-530.
    .. [2] Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
       International Journal of Forecasting, 19(2), 287-290.
    .. [3] Fioruci, J. A., Pellegrini, T. R., Louzada, F., & Petropoulos, F.
       (2015). The optimized theta method. arXiv preprint arXiv:1503.03529.
    """

    def __init__(
        self,
        endog,
        *,
        period: Optional[int] = None,
        deseasonalize: bool = True,
        use_test: bool = True,
        method: str = "auto",
        difference: bool = False
    ) -> None:
        self._y = array_like(endog, "endog", ndim=1)
        if isinstance(endog, pd.DataFrame):
            self.endog_orig = endog.iloc[:, 0]
        else:
            self.endog_orig = endog
        self._period = int_like(period, "period", optional=True)
        self._deseasonalize = bool_like(deseasonalize, "deseasonalize")
        self._use_test = (
            bool_like(use_test, "use_test") and self._deseasonalize
        )
        self._diff = bool_like(difference, "difference")
        self._method = string_like(
            method,
            "model",
            options=("auto", "additive", "multiplicative", "mul", "add"),
        )
        if self._method == "auto":
            self._method = "mul" if self._y.min() > 0 else "add"
        if self._period is None and self._deseasonalize:
            idx = getattr(endog, "index", None)
            pfreq = None
            if idx is not None:
                pfreq = getattr(idx, "freq", None)
                if pfreq is None:
                    pfreq = getattr(idx, "inferred_freq", None)
            if pfreq is not None:
                self._period = freq_to_period(pfreq)
            else:
                raise ValueError(
                    "You must specify a period or endog must be a "
                    "pandas object with a DatetimeIndex with "
                    "a freq not set to None"
                )

        self._has_seasonality = self._deseasonalize

    def _test_seasonality(self) -> None:
        y = self._y
        if self._diff:
            y = np.diff(y)
        rho = acf(y, nlags=self._period, fft=True)
        nobs = y.shape[0]
        stat = nobs * rho[-1] ** 2 / np.sum(rho[:-1] ** 2)
        # CV is 10% from a chi2(1), 1.645**2
        self._has_seasonality = stat > 2.705543454095404

    def _deseasonalize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        y = self._y
        if not self._has_seasonality:
            return self._y, np.empty(0)

        res = seasonal_decompose(y, model=self._method, period=self._period)
        if res.seasonal.min() <= 0:
            self._method = "add"
            res = seasonal_decompose(y, model="add", period=self._period)
            return y - res.seasonal, res.seasonal[: self._period]
        else:
            return y / res.seasonal, res.seasonal[: self._period]

    def fit(
        self, use_mle: bool = False, disp: bool = False
    ) -> "ThetaModelResults":
        r"""
        Estimate model parameters.

        Parameters
        ----------
        use_mle : bool, default False
            Estimate the parameters using MLE by fitting an ARIMA(0,1,1) with
            a drift.  If False (the default), estimates parameters using OLS
            of a constant and a time-trend and by fitting a SES to the model
            data.
        disp : bool, default True
            Display iterative output from fitting the model.

        Notes
        -----
        When using MLE, the parameters are estimated from the ARIMA(0,1,1)

        .. math::

           X_t = X_{t-1} + b_0 + (\alpha-1)\epsilon_{t-1} + \epsilon_t

        When estimating the model using 2-step estimation, the model
        parameters are estimated using the OLS regression

        .. math::

           X_t = a_0 + b_0 (t-1) + \eta_t

        and the SES

        .. math::

           \tilde{X}_{t+1} = \alpha X_{t} + (1-\alpha)\tilde{X}_{t}

        Returns
        -------
        ThetaModelResult
            Model results and forecasting
        """
        if self._deseasonalize and self._use_test:
            self._test_seasonality()
        y, seasonal = self._deseasonalize_data()
        if use_mle:
            mod = SARIMAX(y, order=(0, 1, 1), trend="c")
            res = mod.fit(disp=disp)
            params = np.asarray(res.params)
            alpha = params[1] + 1
            if alpha > 1:
                alpha = 0.9998
                res = mod.fit_constrained({"ma.L1": alpha - 1})
                params = np.asarray(res.params)
            b0 = params[0]
            sigma2 = params[-1]
            one_step = res.forecast(1) - b0
        else:
            ct = add_trend(y, "ct", prepend=True)[:, :2]
            ct[:, 1] -= 1
            _, b0 = np.linalg.lstsq(ct, y, rcond=None)[0]
            res = ExponentialSmoothing(
                y, initial_level=y[0], initialization_method="known"
            ).fit(disp=disp)
            alpha = res.params[0]
            sigma2 = None
            one_step = res.forecast(1)
        return ThetaModelResults(
            b0, alpha, sigma2, one_step, seasonal, use_mle, self
        )

    @property
    def deseasonalize(self) -> bool:
        """Whether to deseasonalize the data"""
        return self._deseasonalize

    @property
    def period(self) -> int:
        """The period of the seasonality"""
        return self._period

    @property
    def use_test(self) -> bool:
        """Whether to test the data for seasonality"""
        return self._use_test

    @property
    def difference(self) -> bool:
        """Whether the data is differenced in the seasonality test"""
        return self._diff

    @property
    def method(self) -> str:
        """The method used to deseasonalize the data"""
        return self._method


class ThetaModelResults:
    """
    Results class from estimated Theta Models.

    Parameters
    ----------
    b0 : float
        The estimated trend slope.
    alpha : float
        The estimated SES parameter.
    sigma2 : float
        The estimated residual variance from the SES/IMA model.
    one_step : float
        The one-step forecast from the SES.
    seasonal : ndarray
        An array of estimated seasonal terms.
    use_mle : bool
        A flag indicating that the parameters were estimated using MLE.
    model : ThetaModel
        The model used to produce the results.
    """

    def __init__(
        self,
        b0: float,
        alpha: float,
        sigma2: Optional[float],
        one_step: float,
        seasonal: np.ndarray,
        use_mle: bool,
        model: ThetaModel,
    ) -> None:
        self._b0 = b0
        self._alpha = alpha
        self._sigma2 = sigma2
        self._one_step = one_step
        self._nobs = model.endog_orig.shape[0]
        self._model = model
        self._seasonal = seasonal
        self._use_mle = use_mle

    @property
    def params(self) -> pd.Series:
        """The forecasting model parameters"""
        return pd.Series([self._b0, self._alpha], index=["b0", "alpha"])

    @property
    def sigma2(self) -> float:
        """The estimated residual variance"""
        if self._sigma2 is None:
            mod = SARIMAX(self.model._y, order=(0, 1, 1), trend="c")
            res = mod.fit(disp=False)
            self._sigma2 = np.asarray(res.params)[-1]
        assert self._sigma2 is not None
        return self._sigma2

    @property
    def model(self) -> ThetaModel:
        """The model used to produce the results"""
        return self._model

    def forecast(self, steps: int = 1, theta: float = 2) -> pd.Series:
        r"""
        Forecast the model for a given theta

        Parameters
        ----------
        steps : int
            The number of steps ahead to compute the forecast components.
        theta : float
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.

        Returns
        -------
        Series
            A Series containing the forecasts

        Notes
        -----
        The forecast is computed as

        .. math::

           \hat{X}_{T+h|T} = \frac{\theta-1}{\theta} b_0
                             \left[h - 1 + \frac{1}{\alpha}
                             - \frac{(1-\alpha)^T}{\alpha} \right]
                             + \tilde{X}_{T+h|T}

        where :math:`\tilde{X}_{T+h|T}` is the SES forecast of the endogenous
        variable using the parameter :math:`\alpha`. :math:`b_0` is the
        slope of a time trend line fitted to X using the terms 0, 1, ..., T-1.

        This expression follows from [1]_ and [2]_ when the combination
        weights are restricted to be (theta-1)/theta and 1/theta. This nests
        the original implementation when theta=2 and the two weights are both
        1/2.

        References
        ----------
        .. [1] Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method.
           International Journal of Forecasting, 19(2), 287-290.
        .. [2] Fioruci, J. A., Pellegrini, T. R., Louzada, F., & Petropoulos,
           F. (2015). The optimized theta method. arXiv preprint
           arXiv:1503.03529.
        """

        steps = int_like(steps, "steps")
        if steps < 1:
            raise ValueError("steps must be a positive integer")
        theta = float_like(theta, "theta")
        if theta < 1:
            raise ValueError("theta must be a float >= 1")
        thresh = 4.0 / np.finfo(np.double).eps
        trend_weight = (theta - 1) / theta if theta < thresh else 1.0
        comp = self.forecast_components(steps=steps)
        fcast = trend_weight * comp.trend + np.asarray(comp.ses)
        # Re-seasonalize if needed
        if self.model.deseasonalize:
            seasonal = np.asarray(comp.seasonal)
            if self.model.method.startswith("mul"):
                fcast *= seasonal
            else:
                fcast += seasonal
        fcast.name = "forecast"

        return fcast

    def forecast_components(self, steps: int = 1) -> pd.DataFrame:
        r"""
        Compute the three components of the Theta model forecast

        Parameters
        ----------
        steps : int
            The number of steps ahead to compute the forecast components.

        Returns
        -------
        DataFrame
            A DataFrame with three columns: trend, ses and seasonal containing
            the forecast values of each of the three components.

        Notes
        -----
        For a given value of :math:`\theta`, the deseasonalized forecast is
        `fcast = w * trend + ses` where :math:`w = \frac{theta - 1}{theta}`.
        The reseasonalized forecasts are then `seasonal * fcast` if the
        seasonality is multiplicative or `seasonal + fcast` if the seasonality
        is additive.
        """
        steps = int_like(steps, "steps")
        if steps < 1:
            raise ValueError("steps must be a positive integer")
        alpha = self._alpha
        b0 = self._b0
        nobs = self._nobs
        h = np.arange(1, steps + 1, dtype=np.float64) - 1
        if alpha > 0:
            h += 1 / alpha - ((1 - alpha) ** nobs / alpha)
        trend = b0 * h
        ses = self._one_step * np.ones(steps)
        if self.model.method.startswith("add"):
            season = np.zeros(steps)
        else:
            season = np.ones(steps)
        # Re-seasonalize
        if self.model.deseasonalize:
            seasonal = self._seasonal
            period = self.model.period
            oos_idx = nobs + np.arange(steps)
            seasonal_locs = oos_idx % period
            if seasonal.shape[0]:
                season[:] = seasonal[seasonal_locs]
        index = getattr(self.model.endog_orig, "index", None)
        if index is None:
            index = pd.RangeIndex(0, self.model.endog_orig.shape[0])
        index = extend_index(steps, index)

        df = pd.DataFrame(
            {"trend": trend, "ses": ses, "seasonal": season}, index=index
        )
        return df

    def summary(self) -> Summary:
        """
        Summarize the model

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
        smry = Summary()

        model_name = type(model).__name__
        title = model_name + " Results"
        method = "MLE" if self._use_mle else "OLS/SES"

        is_series = isinstance(model.endog_orig, pd.Series)
        index = getattr(model.endog_orig, "index", None)
        if is_series and isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            sample = [index[0].strftime("%m-%d-%Y")]
            sample += ["- " + index[-1].strftime("%m-%d-%Y")]
        else:
            sample = [str(0), str(model.endog_orig.shape[0])]

        dep_name = getattr(model.endog_orig, "name", "endog") or "endog"
        top_left = [
            ("Dep. Variable:", [dep_name]),
            ("Method:", [method]),
            ("Date:", None),
            ("Time:", None),
            ("Sample:", [sample[0]]),
            ("", [sample[1]]),
        ]
        method = (
            "Multiplicative" if model.method.startswith("mul") else "Additive"
        )
        top_right = [
            ("No. Observations:", [str(self._nobs)]),
            ("Deseasonalized:", [str(model.deseasonalize)]),
        ]

        if model.deseasonalize:
            top_right.extend(
                [
                    ("Deseas. Method:", [method]),
                    ("Period:", [str(model.period)]),
                    ("", [""]),
                    ("", [""]),
                ]
            )
        else:
            top_right.extend([("", [""])] * 4)

        smry.add_table_2cols(
            self, gleft=top_left, gright=top_right, title=title
        )
        table_fmt = {"data_fmts": ["%s", "%#0.4g"], "data_aligns": "r"}

        data = np.asarray(self.params)[:, None]
        st = SimpleTable(
            data,
            ["Parameters", "Estimate"],
            list(self.params.index),
            title="Parameter Estimates",
            txt_fmt=table_fmt,
        )
        smry.tables.append(st)

        return smry

    def prediction_intervals(
        self, steps: int = 1, theta: float = 2, alpha: float = 0.05
    ) -> pd.DataFrame:
        r"""
        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : float, default 0.05
            Significance level for the confidence intervals.

        Returns
        -------
        DataFrame
            DataFrame with columns lower and upper

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\sigma^2(1 + (h-1)(1 + (\alpha-1)^2)`. The prediction interval
        assumes that innovations are normally distributed.
        """
        model_alpha = self.params.iloc[1]
        sigma2_h = (
            1 + np.arange(steps) * (1 + (model_alpha - 1) ** 2)
        ) * self.sigma2
        sigma_h = np.sqrt(sigma2_h)
        quantile = stats.norm.ppf(alpha / 2)
        predictions = self.forecast(steps, theta)
        return pd.DataFrame(
            {
                "lower": predictions + sigma_h * quantile,
                "upper": predictions + sigma_h * -quantile,
            }
        )

    def plot_predict(
        self,
        steps: int = 1,
        theta: float = 2,
        alpha: Optional[float] = 0.05,
        in_sample: bool = False,
        fig: Optional["matplotlib.figure.Figure"] = None,
        figsize: Tuple[float, float] = None,
    ) -> "matplotlib.figure.Figure":
        r"""
        Plot forecasts, prediction intervals and in-sample values

        Parameters
        ----------
        steps : int, default 1
            The number of steps ahead to compute the forecast components.
        theta : float, default 2
            The theta value to use when computing the weight to combine
            the trend and the SES forecasts.
        alpha : {float, None}, default 0.05
            The tail probability not covered by the confidence interval. Must
            be in (0, 1). Confidence interval is constructed assuming normally
            distributed shocks. If None, figure will not show the confidence
            interval.
        in_sample : bool, default False
            Flag indicating whether to include the in-sample period in the
            plot.
        fig : Figure, default None
            An existing figure handle. If not provided, a new figure is
            created.
        figsize: tuple[float, float], default None
            Tuple containing the figure size.

        Returns
        -------
        Figure
            Figure handle containing the plot.

        Notes
        -----
        The variance of the h-step forecast is assumed to follow from the
        integrated Moving Average structure of the Theta model, and so is
        :math:`\sigma^2(\alpha^2 + (h-1))`. The prediction interval assumes
        that innovations are normally distributed.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig

        _import_mpl()
        fig = create_mpl_fig(fig, figsize)
        assert fig is not None
        predictions = self.forecast(steps, theta)
        pred_index = predictions.index

        ax = fig.add_subplot(111)
        nobs = self.model.endog_orig.shape[0]
        index = pd.Index(np.arange(nobs))
        if in_sample:
            if isinstance(self.model.endog_orig, pd.Series):
                index = self.model.endog_orig.index
            ax.plot(index, self.model.endog_orig)
        ax.plot(pred_index, predictions)
        if alpha is not None:
            pi = self.prediction_intervals(steps, theta, alpha)
            label = "{0:.0%} confidence interval".format(1 - alpha)
            ax.fill_between(
                pred_index,
                pi["lower"],
                pi["upper"],
                color="gray",
                alpha=0.5,
                label=label,
            )

        ax.legend(loc="best", frameon=False)
        fig.tight_layout(pad=1.0)

        return fig
