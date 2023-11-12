import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import (
    boxcox,
    rv_continuous,
    rv_discrete,
)
from scipy.stats.distributions import rv_frozen

from statsmodels.base.data import PandasData
from statsmodels.base.model import Results
from statsmodels.base.wrapper import (
    ResultsWrapper,
    populate_wrapper,
    union_dicts,
)


class HoltWintersResults(Results):
    """
    Results from fitting Exponential Smoothing models.

    Parameters
    ----------
    model : ExponentialSmoothing instance
        The fitted model instance.
    params : dict
        All the parameters for the Exponential Smoothing model.
    sse : float
        The sum of squared errors.
    aic : float
        The Akaike information criterion.
    aicc : float
        AIC with a correction for finite sample sizes.
    bic : float
        The Bayesian information criterion.
    optimized : bool
        Flag indicating whether the model parameters were optimized to fit
        the data.
    level : ndarray
        An array of the levels values that make up the fitted values.
    trend : ndarray
        An array of the trend values that make up the fitted values.
    season : ndarray
        An array of the seasonal values that make up the fitted values.
    params_formatted : pd.DataFrame
        DataFrame containing all parameters, their short names and a flag
        indicating whether the parameter's value was optimized to fit the data.
    resid : ndarray
        An array of the residuals of the fittedvalues and actual values.
    k : int
        The k parameter used to remove the bias in AIC, BIC etc.
    fittedvalues : ndarray
        An array of the fitted values. Fitted by the Exponential Smoothing
        model.
    fittedfcast : ndarray
        An array of both the fitted values and forecast values.
    fcastvalues : ndarray
        An array of the forecast values forecast by the Exponential Smoothing
        model.
    mle_retvals : {None, scipy.optimize.optimize.OptimizeResult}
        Optimization results if the parameters were optimized to fit the data.
    """

    def __init__(
        self,
        model,
        params,
        sse,
        aic,
        aicc,
        bic,
        optimized,
        level,
        trend,
        season,
        params_formatted,
        resid,
        k,
        fittedvalues,
        fittedfcast,
        fcastvalues,
        mle_retvals=None,
    ):
        self.data = model.data
        super().__init__(model, params)
        self._model = model
        self._sse = sse
        self._aic = aic
        self._aicc = aicc
        self._bic = bic
        self._optimized = optimized
        self._level = level
        self._trend = trend
        self._season = season
        self._params_formatted = params_formatted
        self._fittedvalues = fittedvalues
        self._fittedfcast = fittedfcast
        self._fcastvalues = fcastvalues
        self._resid = resid
        self._k = k
        self._mle_retvals = mle_retvals

    @property
    def aic(self):
        """
        The Akaike information criterion.
        """
        return self._aic

    @property
    def aicc(self):
        """
        AIC with a correction for finite sample sizes.
        """
        return self._aicc

    @property
    def bic(self):
        """
        The Bayesian information criterion.
        """
        return self._bic

    @property
    def sse(self):
        """
        The sum of squared errors between the data and the fittted value.
        """
        return self._sse

    @property
    def model(self):
        """
        The model used to produce the results instance.
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def level(self):
        """
        An array of the levels values that make up the fitted values.
        """
        return self._level

    @property
    def optimized(self):
        """
        Flag indicating if model parameters were optimized to fit the data.
        """
        return self._optimized

    @property
    def trend(self):
        """
        An array of the trend values that make up the fitted values.
        """
        return self._trend

    @property
    def season(self):
        """
        An array of the seasonal values that make up the fitted values.
        """
        return self._season

    @property
    def params_formatted(self):
        """
        DataFrame containing all parameters

        Contains short names and a flag indicating whether the parameter's
        value was optimized to fit the data.
        """
        return self._params_formatted

    @property
    def fittedvalues(self):
        """
        An array of the fitted values
        """
        return self._fittedvalues

    @property
    def fittedfcast(self):
        """
        An array of both the fitted values and forecast values.
        """
        return self._fittedfcast

    @property
    def fcastvalues(self):
        """
        An array of the forecast values
        """
        return self._fcastvalues

    @property
    def resid(self):
        """
        An array of the residuals of the fittedvalues and actual values.
        """
        return self._resid

    @property
    def k(self):
        """
        The k parameter used to remove the bias in AIC, BIC etc.
        """
        return self._k

    @property
    def mle_retvals(self):
        """
        Optimization results if the parameters were optimized to fit the data.
        """
        return self._mle_retvals

    @mle_retvals.setter
    def mle_retvals(self, value):
        self._mle_retvals = value

    def predict(self, start=None, end=None):
        """
        In-sample prediction and out-of-sample forecasting

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Can also be a date string to
            parse or a datetime type. However, if the dates index does not
            have a fixed frequency, end must be an integer index if you
            want out of sample prediction. Default is the last observation in
            the sample.

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts.
        """
        return self.model.predict(self.params, start, end)

    def forecast(self, steps=1):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of out of sample forecasts from the end of the
            sample.

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts
        """
        try:
            freq = getattr(self.model._index, "freq", 1)
            if not isinstance(freq, int) and isinstance(
                self.model._index, (pd.DatetimeIndex, pd.PeriodIndex)
            ):
                start = self.model._index[-1] + freq
                end = self.model._index[-1] + steps * freq
            else:
                start = self.model._index.shape[0]
                end = start + steps - 1
            return self.model.predict(self.params, start=start, end=end)
        except AttributeError:
            # May occur when the index does not have a freq
            return self.model._predict(h=steps, **self.params).fcastvalues

    def summary(self):
        """
        Summarize the fitted Model

        Returns
        -------
        smry : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary
        from statsmodels.iolib.table import SimpleTable

        model = self.model
        title = model.__class__.__name__ + " Model Results"

        dep_variable = "endog"
        orig_endog = self.model.data.orig_endog
        if isinstance(orig_endog, pd.DataFrame):
            dep_variable = orig_endog.columns[0]
        elif isinstance(orig_endog, pd.Series):
            dep_variable = orig_endog.name
        seasonal_periods = (
            None
            if self.model.seasonal is None
            else self.model.seasonal_periods
        )
        lookup = {
            "add": "Additive",
            "additive": "Additive",
            "mul": "Multiplicative",
            "multiplicative": "Multiplicative",
            None: "None",
        }
        transform = self.params["use_boxcox"]
        box_cox_transform = True if transform else False
        box_cox_coeff = (
            transform if isinstance(transform, str) else self.params["lamda"]
        )
        if isinstance(box_cox_coeff, float):
            box_cox_coeff = "{:>10.5f}".format(box_cox_coeff)
        top_left = [
            ("Dep. Variable:", [dep_variable]),
            ("Model:", [model.__class__.__name__]),
            ("Optimized:", [str(np.any(self.optimized))]),
            ("Trend:", [lookup[self.model.trend]]),
            ("Seasonal:", [lookup[self.model.seasonal]]),
            ("Seasonal Periods:", [str(seasonal_periods)]),
            ("Box-Cox:", [str(box_cox_transform)]),
            ("Box-Cox Coeff.:", [str(box_cox_coeff)]),
        ]

        top_right = [
            ("No. Observations:", [str(len(self.model.endog))]),
            ("SSE", ["{:5.3f}".format(self.sse)]),
            ("AIC", ["{:5.3f}".format(self.aic)]),
            ("BIC", ["{:5.3f}".format(self.bic)]),
            ("AICC", ["{:5.3f}".format(self.aicc)]),
            ("Date:", None),
            ("Time:", None),
        ]

        smry = Summary()
        smry.add_table_2cols(
            self, gleft=top_left, gright=top_right, title=title
        )
        formatted = self.params_formatted  # type: pd.DataFrame

        def _fmt(x):
            abs_x = np.abs(x)
            scale = 1
            if np.isnan(x):
                return f"{str(x):>20}"
            if abs_x != 0:
                scale = int(np.log10(abs_x))
            if scale > 4 or scale < -3:
                return "{:>20.5g}".format(x)
            dec = min(7 - scale, 7)
            fmt = "{{:>20.{0}f}}".format(dec)
            return fmt.format(x)

        tab = []
        for _, vals in formatted.iterrows():
            tab.append(
                [
                    _fmt(vals.iloc[1]),
                    "{0:>20}".format(vals.iloc[0]),
                    "{0:>20}".format(str(bool(vals.iloc[2]))),
                ]
            )
        params_table = SimpleTable(
            tab,
            headers=["coeff", "code", "optimized"],
            title="",
            stubs=list(formatted.index),
        )

        smry.tables.append(params_table)

        return smry

    def simulate(
        self,
        nsimulations,
        anchor=None,
        repetitions=1,
        error="add",
        random_errors=None,
        random_state=None,
    ):
        r"""
        Random simulations using the state space formulation.

        Parameters
        ----------
        nsimulations : int
            The number of simulation steps.
        anchor : int, str, or datetime, optional
            First period for simulation. The simulation will be conditional on
            all existing datapoints prior to the `anchor`.  Type depends on the
            index of the given `endog` in the model. Two special cases are the
            strings 'start' and 'end'. `start` refers to beginning the
            simulation at the first period of the sample, and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'end'.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
        error : {"add", "mul", "additive", "multiplicative"}, optional
            Error model for state space formulation. Default is ``"add"``.
        random_errors : optional
            Specifies how the random errors should be obtained. Can be one of
            the following:

            * ``None``: Random normally distributed values with variance
              estimated from the fit errors drawn from numpy's standard
              RNG (can be seeded with the `random_state` argument). This is the
              default option.
            * A distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm``: Fits the distribution function to the fit
              errors and draws from the fitted distribution.
              Note the difference between ``scipy.stats.norm`` and
              ``scipy.stats.norm()``, the latter one is a frozen distribution
              function.
            * A frozen distribution function from ``scipy.stats``, e.g.
              ``scipy.stats.norm(scale=2)``: Draws from the frozen distribution
              function.
            * A ``np.ndarray`` with shape (`nsimulations`, `repetitions`): Uses
              the given values as random errors.
            * ``"bootstrap"``: Samples the random errors from the fit errors.

        random_state : int or np.random.RandomState, optional
            A seed for the random number generator or a
            ``np.random.RandomState`` object. Only used if `random_errors` is
            ``None``. Default is ``None``.

        Returns
        -------
        sim : pd.Series, pd.DataFrame or np.ndarray
            An ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame`` of simulated
            values.
            If the original data was a ``pd.Series`` or ``pd.DataFrame``, `sim`
            will be a ``pd.Series`` if `repetitions` is 1, and a
            ``pd.DataFrame`` of shape (`nsimulations`, `repetitions`) else.
            Otherwise, if `repetitions` is 1, a ``np.ndarray`` of shape
            (`nsimulations`,) is returned, and if `repetitions` is not 1 a
            ``np.ndarray`` of shape (`nsimulations`, `repetitions`) is
            returned.

        Notes
        -----
        The simulation is based on the state space model of the Holt-Winter's
        methods. The state space model assumes that the true value at time
        :math:`t` is randomly distributed around the prediction value.
        If using the additive error model, this means:

        .. math::

            y_t &= \hat{y}_{t|t-1} + e_t\\
            e_t &\sim \mathcal{N}(0, \sigma^2)

        Using the multiplicative error model:

        .. math::

            y_t &= \hat{y}_{t|t-1} \cdot (1 + e_t)\\
            e_t &\sim \mathcal{N}(0, \sigma^2)

        Inserting these equations into the smoothing equation formulation leads
        to the state space equations. The notation used here follows
        [1]_.

        Additionally,

        .. math::

           B_t &= b_{t-1} \circ_d \phi\\
           L_t &= l_{t-1} \circ_b B_t\\
           S_t &= s_{t-m}\\
           Y_t &= L_t \circ_s S_t,

        where :math:`\circ_d` is the operation linking trend and damping
        parameter (multiplication if the trend is additive, power if the trend
        is multiplicative), :math:`\circ_b` is the operation linking level and
        trend (addition if the trend is additive, multiplication if the trend
        is multiplicative), and :math:`\circ_s` is the operation linking
        seasonality to the rest.

        The state space equations can then be formulated as

        .. math::

           y_t &= Y_t + \eta \cdot e_t\\
           l_t &= L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
           b_t &= B_t + \beta \cdot (M_e \cdot B_t + \kappa_b) \cdot e_t\\
           s_t &= S_t + \gamma \cdot (M_e \cdot S_t + \kappa_s) \cdot e_t\\

        with

        .. math::

           \eta &= \begin{cases}
                       Y_t\quad\text{if error is multiplicative}\\
                       1\quad\text{else}
                   \end{cases}\\
           M_e &= \begin{cases}
                       1\quad\text{if error is multiplicative}\\
                       0\quad\text{else}
                   \end{cases}\\

        and, when using the additive error model,

        .. math::

           \kappa_l &= \begin{cases}
                       \frac{1}{S_t}\quad
                       \text{if seasonality is multiplicative}\\
                       1\quad\text{else}
                   \end{cases}\\
           \kappa_b &= \begin{cases}
                       \frac{\kappa_l}{l_{t-1}}\quad
                       \text{if trend is multiplicative}\\
                       \kappa_l\quad\text{else}
                   \end{cases}\\
           \kappa_s &= \begin{cases}
                       \frac{1}{L_t}\quad\text{if seasonality is
                                               multiplicative}\\
                       1\quad\text{else}
                   \end{cases}

        When using the multiplicative error model

        .. math::

           \kappa_l &= \begin{cases}
                       0\quad
                       \text{if seasonality is multiplicative}\\
                       S_t\quad\text{else}
                   \end{cases}\\
           \kappa_b &= \begin{cases}
                       \frac{\kappa_l}{l_{t-1}}\quad
                       \text{if trend is multiplicative}\\
                       \kappa_l + l_{t-1}\quad\text{else}
                   \end{cases}\\
           \kappa_s &= \begin{cases}
                       0\quad\text{if seasonality is multiplicative}\\
                       L_t\quad\text{else}
                   \end{cases}

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) *Forecasting:
           principles and practice*, 2nd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp2. Accessed on February 28th 2020.
        """

        # check inputs
        if error in ["additive", "multiplicative"]:
            error = {"additive": "add", "multiplicative": "mul"}[error]
        if error not in ["add", "mul"]:
            raise ValueError("error must be 'add' or 'mul'!")

        # Get the starting location
        if anchor is None or anchor == "end":
            start_idx = self.model.nobs
        elif anchor == "start":
            start_idx = 0
        else:
            start_idx, _, _ = self.model._get_index_loc(anchor)
            if isinstance(start_idx, slice):
                start_idx = start_idx.start
        if start_idx < 0:
            start_idx += self.model.nobs
        if start_idx > self.model.nobs:
            raise ValueError("Cannot anchor simulation outside of the sample.")

        # get Holt-Winters settings and parameters
        trend = self.model.trend
        damped = self.model.damped_trend
        seasonal = self.model.seasonal
        use_boxcox = self.params["use_boxcox"]
        lamda = self.params["lamda"]
        alpha = self.params["smoothing_level"]
        beta = self.params["smoothing_trend"]
        gamma = self.params["smoothing_seasonal"]
        phi = self.params["damping_trend"]
        # if model has no seasonal component, use 1 as period length
        m = max(self.model.seasonal_periods, 1)
        n_params = (
            2
            + 2 * self.model.has_trend
            + (m + 1) * self.model.has_seasonal
            + damped
        )
        mul_seasonal = seasonal == "mul"
        mul_trend = trend == "mul"
        mul_error = error == "mul"

        # define trend, damping and seasonality operations
        if mul_trend:
            op_b = np.multiply
            op_d = np.power
            neutral_b = 1
        else:
            op_b = np.add
            op_d = np.multiply
            neutral_b = 0
        if mul_seasonal:
            op_s = np.multiply
            neutral_s = 1
        else:
            op_s = np.add
            neutral_s = 0

        # set initial values
        level = self.level
        _trend = self.trend
        season = self.season
        # (notation as in https://otexts.com/fpp2/ets.html)
        y = np.empty((nsimulations, repetitions))
        # lvl instead of l because of E741
        lvl = np.empty((nsimulations + 1, repetitions))
        b = np.empty((nsimulations + 1, repetitions))
        s = np.empty((nsimulations + m, repetitions))
        # the following uses python's index wrapping
        if start_idx == 0:
            lvl[-1, :] = self.params["initial_level"]
            b[-1, :] = self.params["initial_trend"]
        else:
            lvl[-1, :] = level[start_idx - 1]
            b[-1, :] = _trend[start_idx - 1]
        if 0 <= start_idx and start_idx <= m:
            initial_seasons = self.params["initial_seasons"]
            _s = np.concatenate(
                (initial_seasons[start_idx:], season[:start_idx])
            )
            s[-m:, :] = np.tile(_s, (repetitions, 1)).T
        else:
            s[-m:, :] = np.tile(
                season[start_idx - m : start_idx], (repetitions, 1)
            ).T

        # set neutral values for unused features
        if trend is None:
            b[:, :] = neutral_b
            phi = 1
            beta = 0
        if seasonal is None:
            s[:, :] = neutral_s
            gamma = 0
        if not damped:
            phi = 1

        # calculate residuals for error covariance estimation
        if use_boxcox:
            fitted = boxcox(self.fittedvalues, lamda)
        else:
            fitted = self.fittedvalues
        if error == "add":
            resid = self.model._y - fitted
        else:
            resid = (self.model._y - fitted) / fitted
        sigma = np.sqrt(np.sum(resid**2) / (len(resid) - n_params))

        # get random error eps
        if isinstance(random_errors, np.ndarray):
            if random_errors.shape != (nsimulations, repetitions):
                raise ValueError(
                    "If random_errors is an ndarray, it must have shape "
                    "(nsimulations, repetitions)"
                )
            eps = random_errors
        elif random_errors == "bootstrap":
            eps = np.random.choice(
                resid, size=(nsimulations, repetitions), replace=True
            )
        elif random_errors is None:
            if random_state is None:
                eps = np.random.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
                eps = rng.randn(nsimulations, repetitions) * sigma
            elif isinstance(random_state, np.random.RandomState):
                eps = random_state.randn(nsimulations, repetitions) * sigma
            else:
                raise ValueError(
                    "Argument random_state must be None, an integer, "
                    "or an instance of np.random.RandomState"
                )
        elif isinstance(random_errors, (rv_continuous, rv_discrete)):
            params = random_errors.fit(resid)
            eps = random_errors.rvs(*params, size=(nsimulations, repetitions))
        elif isinstance(random_errors, rv_frozen):
            eps = random_errors.rvs(size=(nsimulations, repetitions))
        else:
            raise ValueError("Argument random_errors has unexpected value!")

        for t in range(nsimulations):
            b0 = op_d(b[t - 1, :], phi)
            l0 = op_b(lvl[t - 1, :], b0)
            s0 = s[t - m, :]
            y0 = op_s(l0, s0)
            if error == "add":
                eta = 1
                kappa_l = 1 / s0 if mul_seasonal else 1
                kappa_b = kappa_l / lvl[t - 1, :] if mul_trend else kappa_l
                kappa_s = 1 / l0 if mul_seasonal else 1
            else:
                eta = y0
                kappa_l = 0 if mul_seasonal else s0
                kappa_b = (
                    kappa_l / lvl[t - 1, :]
                    if mul_trend
                    else kappa_l + lvl[t - 1, :]
                )
                kappa_s = 0 if mul_seasonal else l0

            y[t, :] = y0 + eta * eps[t, :]
            lvl[t, :] = l0 + alpha * (mul_error * l0 + kappa_l) * eps[t, :]
            b[t, :] = b0 + beta * (mul_error * b0 + kappa_b) * eps[t, :]
            s[t, :] = s0 + gamma * (mul_error * s0 + kappa_s) * eps[t, :]

        if use_boxcox:
            y = inv_boxcox(y, lamda)

        sim = np.atleast_1d(np.squeeze(y))
        if y.shape[0] == 1 and y.size > 1:
            sim = sim[None, :]
        # Wrap data / squeeze where appropriate
        if not isinstance(self.model.data, PandasData):
            return sim

        _, _, _, index = self.model._get_prediction_index(
            start_idx, start_idx + nsimulations - 1
        )
        if repetitions == 1:
            sim = pd.Series(sim, index=index, name=self.model.endog_names)
        else:
            sim = pd.DataFrame(sim, index=index)

        return sim


class HoltWintersResultsWrapper(ResultsWrapper):
    _attrs = {
        "fittedvalues": "rows",
        "level": "rows",
        "resid": "rows",
        "season": "rows",
        "trend": "rows",
        "slope": "rows",
    }
    _wrap_attrs = union_dicts(ResultsWrapper._wrap_attrs, _attrs)
    _methods = {"predict": "dates", "forecast": "dates"}
    _wrap_methods = union_dicts(ResultsWrapper._wrap_methods, _methods)


populate_wrapper(HoltWintersResultsWrapper, HoltWintersResults)
