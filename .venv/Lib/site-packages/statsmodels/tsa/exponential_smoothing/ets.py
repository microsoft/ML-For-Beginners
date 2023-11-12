r"""
ETS models for time series analysis.

The ETS models are a family of time series models. They can be seen as a
generalization of simple exponential smoothing to time series that contain
trends and seasonalities. Additionally, they have an underlying state space
model.

An ETS model is specified by an error type (E; additive or multiplicative), a
trend type (T; additive or multiplicative, both damped or undamped, or none),
and a seasonality type (S; additive or multiplicative or none).
The following gives a very short summary, a more thorough introduction can be
found in [1]_.

Denote with :math:`\circ_b` the trend operation (addition or
multiplication), with :math:`\circ_d` the operation linking trend and dampening
factor :math:`\phi` (multiplication if trend is additive, power if trend is
multiplicative), and with :math:`\circ_s` the seasonality operation (addition
or multiplication).
Furthermore, let :math:`\ominus` be the respective inverse operation
(subtraction or division).

With this, it is possible to formulate the ETS models as a forecast equation
and 3 smoothing equations. The former is used to forecast observations, the
latter are used to update the internal state.

.. math::

    \hat{y}_{t|t-1} &= (l_{t-1} \circ_b (b_{t-1}\circ_d \phi))\circ_s s_{t-m}\\
    l_{t} &= \alpha (y_{t} \ominus_s s_{t-m})
             + (1 - \alpha) (l_{t-1} \circ_b (b_{t-1} \circ_d \phi))\\
    b_{t} &= \beta/\alpha (l_{t} \ominus_b l_{t-1})
             + (1 - \beta/\alpha) b_{t-1}\\
    s_{t} &= \gamma (y_t \ominus_s (l_{t-1} \circ_b (b_{t-1}\circ_d\phi))
             + (1 - \gamma) s_{t-m}

The notation here follows [1]_; :math:`l_t` denotes the level at time
:math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m` is the
number of seasonal periods, and :math:`\phi` a trend damping factor.
The parameters :math:`\alpha, \beta, \gamma` are the smoothing parameters,
which are called ``smoothing_level``, ``smoothing_trend``, and
``smoothing_seasonal``, respectively.

Note that the formulation above as forecast and smoothing equation does not
distinguish different error models -- it is the same for additive and
multiplicative errors. But the different error models lead to different
likelihood models, and therefore will lead to different fit results.

The error models specify how the true values :math:`y_t` are updated. In the
additive error model,

.. math::

    y_t = \hat{y}_{t|t-1} + e_t,

in the multiplicative error model,

.. math::

    y_t = \hat{y}_{t|t-1}\cdot (1 + e_t).

Using these error models, it is possible to formulate state space equations for
the ETS models:

.. math::

   y_t &= Y_t + \eta \cdot e_t\\
   l_t &= L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
   b_t &= B_t + \beta \cdot (M_e \cdot B_t + \kappa_b) \cdot e_t\\
   s_t &= S_t + \gamma \cdot (M_e \cdot S_t+\kappa_s)\cdot e_t\\

with

.. math::

   B_t &= b_{t-1} \circ_d \phi\\
   L_t &= l_{t-1} \circ_b B_t\\
   S_t &= s_{t-m}\\
   Y_t &= L_t \circ_s S_t,

and

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
               \frac{1}{L_t}\quad\text{if seasonality is multiplicative}\\
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

When fitting an ETS model, the parameters :math:`\alpha, \beta`, \gamma,
\phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}` are
selected as maximizers of log likelihood.

References
----------
.. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
   principles and practice*, 3rd edition, OTexts: Melbourne,
   Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""

from collections import OrderedDict
import contextlib
import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen

from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    int_like,
    string_like,
)
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
    _initialization_simple,
    _initialization_heuristic,
)
from statsmodels.tsa.tsatools import freq_to_period

# Implementation details:

# * The smoothing equations are implemented only for models having all
#   components (trend, dampening, seasonality). When using other models, the
#   respective parameters (smoothing and initial parameters) are set to values
#   that lead to the reduced model (often zero).
#   The internal model is needed for smoothing (called from fit and loglike),
#   forecasts, and simulations.
# * Somewhat related to above: There are 2 sets of parameters: model/external
#   params, and internal params.
#   - model params are all parameters necessary for a model, and are for
#     example passed as argument to the likelihood function or as start_params
#     to fit
#   - internal params are what is used internally in the smoothing equations
# * Regarding fitting, bounds, fixing parameters, and internal parameters, the
#   overall workflow is the following:
#   - get start parameters in the form of external parameters (includes fixed
#     parameters)
#   - transform external parameters to internal parameters, bounding all that
#     are missing -> now we have some missing parameters, but potentially also
#     some user-specified bounds
#   - set bounds for fixed parameters
#   - make sure that starting parameters are within bounds
#   - set up the constraint bounds and function
# * Since the traditional bounds are nonlinear for beta and gamma, if no bounds
#   are given, we internally use beta_star and gamma_star for fitting
# * When estimating initial level and initial seasonal values, one of them has
#   to be removed in order to have a well posed problem. I am solving this by
#   fixing the last initial seasonal value to 0 (for additive seasonality) or 1
#   (for multiplicative seasonality).
#   For the additive models, this means I have to subtract the last initial
#   seasonal value from all initial seasonal values and add it to the initial
#   level; for the multiplicative models I do the same with division and
#   multiplication


class ETSModel(base.StateSpaceMLEModel):
    r"""
    ETS models.

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    error : str, optional
        The error model. "add" (default) or "mul".
    trend : str or None, optional
        The trend component model. "add", "mul", or None (default).
    damped_trend : bool, optional
        Whether or not an included trend component is damped. Default is
        False.
    seasonal : str, optional
        The seasonality model. "add", "mul", or None (default).
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle for seasonal
        (Holt-Winters) models. For example, 4 for quarterly data with an
        annual cycle or 7 for daily data with a weekly cycle. Required if
        `seasonal` is not None.
    initialization_method : str, optional
        Method for initialization of the state space model. One of:

        * 'estimated' (default)
        * 'heuristic'
        * 'known'

        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_trend` and `initial_seasonal` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.  Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Only used if initialization is 'known'.
    initial_trend : float, optional
        The initial trend component. Only used if initialization is 'known'.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal_periods`.
        Only used if initialization is 'known'.
    bounds : dict or None, optional
        A dictionary with parameter names as keys and the respective bounds
        intervals as values (lists/tuples/arrays).
        The available parameter names are, depending on the model and
        initialization method:

        * "smoothing_level"
        * "smoothing_trend"
        * "smoothing_seasonal"
        * "damping_trend"
        * "initial_level"
        * "initial_trend"
        * "initial_seasonal.0", ..., "initial_seasonal.<m-1>"

        The default option is ``None``, in which case the traditional
        (nonlinear) bounds as described in [1]_ are used.

    Notes
    -----
    The ETS models are a family of time series models. They can be seen as a
    generalization of simple exponential smoothing to time series that contain
    trends and seasonalities. Additionally, they have an underlying state
    space model.

    An ETS model is specified by an error type (E; additive or multiplicative),
    a trend type (T; additive or multiplicative, both damped or undamped, or
    none), and a seasonality type (S; additive or multiplicative or none).
    The following gives a very short summary, a more thorough introduction can
    be found in [1]_.

    Denote with :math:`\circ_b` the trend operation (addition or
    multiplication), with :math:`\circ_d` the operation linking trend and
    dampening factor :math:`\phi` (multiplication if trend is additive, power
    if trend is multiplicative), and with :math:`\circ_s` the seasonality
    operation (addition or multiplication). Furthermore, let :math:`\ominus`
    be the respective inverse operation (subtraction or division).

    With this, it is possible to formulate the ETS models as a forecast
    equation and 3 smoothing equations. The former is used to forecast
    observations, the latter are used to update the internal state.

    .. math::

        \hat{y}_{t|t-1} &= (l_{t-1} \circ_b (b_{t-1}\circ_d \phi))
                           \circ_s s_{t-m}\\
        l_{t} &= \alpha (y_{t} \ominus_s s_{t-m})
                 + (1 - \alpha) (l_{t-1} \circ_b (b_{t-1} \circ_d \phi))\\
        b_{t} &= \beta/\alpha (l_{t} \ominus_b l_{t-1})
                 + (1 - \beta/\alpha) b_{t-1}\\
        s_{t} &= \gamma (y_t \ominus_s (l_{t-1} \circ_b (b_{t-1}\circ_d\phi))
                 + (1 - \gamma) s_{t-m}

    The notation here follows [1]_; :math:`l_t` denotes the level at time
    :math:`t`, `b_t` the trend, and `s_t` the seasonal component. :math:`m`
    is the number of seasonal periods, and :math:`\phi` a trend damping
    factor. The parameters :math:`\alpha, \beta, \gamma` are the smoothing
    parameters, which are called ``smoothing_level``, ``smoothing_trend``, and
    ``smoothing_seasonal``, respectively.

    Note that the formulation above as forecast and smoothing equation does
    not distinguish different error models -- it is the same for additive and
    multiplicative errors. But the different error models lead to different
    likelihood models, and therefore will lead to different fit results.

    The error models specify how the true values :math:`y_t` are
    updated. In the additive error model,

    .. math::

        y_t = \hat{y}_{t|t-1} + e_t,

    in the multiplicative error model,

    .. math::

        y_t = \hat{y}_{t|t-1}\cdot (1 + e_t).

    Using these error models, it is possible to formulate state space
    equations for the ETS models:

    .. math::

       y_t &= Y_t + \eta \cdot e_t\\
       l_t &= L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
       b_t &= B_t + \beta \cdot (M_e \cdot B_t + \kappa_b) \cdot e_t\\
       s_t &= S_t + \gamma \cdot (M_e \cdot S_t+\kappa_s)\cdot e_t\\

    with

    .. math::

       B_t &= b_{t-1} \circ_d \phi\\
       L_t &= l_{t-1} \circ_b B_t\\
       S_t &= s_{t-m}\\
       Y_t &= L_t \circ_s S_t,

    and

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
                   \frac{1}{L_t}\quad\text{if seasonality is multiplicative}\\
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

    When fitting an ETS model, the parameters :math:`\alpha, \beta`, \gamma,
    \phi` and the initial states `l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}` are
    selected as maximizers of log likelihood.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
       principles and practice*, 3rd edition, OTexts: Melbourne,
       Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
    """

    def __init__(
        self,
        endog,
        error="add",
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
        initialization_method="estimated",
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        bounds=None,
        dates=None,
        freq=None,
        missing="none",
    ):

        super().__init__(
            endog, exog=None, dates=dates, freq=freq, missing=missing
        )

        # MODEL DEFINITION
        # ================
        options = ("add", "mul", "additive", "multiplicative")
        # take first three letters of option -> either "add" or "mul"
        self.error = string_like(error, "error", options=options)[:3]
        self.trend = string_like(
            trend, "trend", options=options, optional=True
        )
        if self.trend is not None:
            self.trend = self.trend[:3]
        self.damped_trend = bool_like(damped_trend, "damped_trend")
        self.seasonal = string_like(
            seasonal, "seasonal", options=options, optional=True
        )
        if self.seasonal is not None:
            self.seasonal = self.seasonal[:3]

        self.has_trend = self.trend is not None
        self.has_seasonal = self.seasonal is not None

        if self.has_seasonal:
            self.seasonal_periods = int_like(
                seasonal_periods, "seasonal_periods", optional=True
            )
            if seasonal_periods is None:
                self.seasonal_periods = freq_to_period(self._index_freq)
            if self.seasonal_periods <= 1:
                raise ValueError("seasonal_periods must be larger than 1.")
        else:
            # in case the model has no seasonal component, we internally handle
            # this as if it had an additive seasonal component with
            # seasonal_periods=1, but restrict the smoothing parameter to 0 and
            # set the initial seasonal to 0.
            self.seasonal_periods = 1

        # reject invalid models
        if np.any(self.endog <= 0) and (
            self.error == "mul"
            or self.trend == "mul"
            or self.seasonal == "mul"
        ):
            raise ValueError(
                "endog must be strictly positive when using "
                "multiplicative error, trend or seasonal components."
            )
        if self.damped_trend and not self.has_trend:
            raise ValueError("Can only dampen the trend component")

        # INITIALIZATION METHOD
        # =====================
        self.set_initialization_method(
            initialization_method,
            initial_level,
            initial_trend,
            initial_seasonal,
        )

        # BOUNDS
        # ======
        self.set_bounds(bounds)

        # SMOOTHER
        # ========
        if self.trend == "add" or self.trend is None:
            if self.seasonal == "add" or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_add_add
            else:
                self._smoothing_func = smooth._ets_smooth_add_mul
        else:
            if self.seasonal == "add" or self.seasonal is None:
                self._smoothing_func = smooth._ets_smooth_mul_add
            else:
                self._smoothing_func = smooth._ets_smooth_mul_mul

    def set_initialization_method(
        self,
        initialization_method,
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
    ):
        """
        Sets a new initialization method for the state space model.

        Parameters
        ----------
        initialization_method : str, optional
            Method for initialization of the state space model. One of:

            * 'estimated' (default)
            * 'heuristic'
            * 'known'

            If 'known' initialization is used, then `initial_level` must be
            passed, as well as `initial_trend` and `initial_seasonal` if
            applicable.
            'heuristic' uses a heuristic based on the data to estimate initial
            level, trend, and seasonal state. 'estimated' uses the same
            heuristic as initial guesses, but then estimates the initial states
            as part of the fitting process. Default is 'estimated'.
        initial_level : float, optional
            The initial level component. Only used if initialization is
            'known'.
        initial_trend : float, optional
            The initial trend component. Only used if initialization is
            'known'.
        initial_seasonal : array_like, optional
            The initial seasonal component. An array of length
            `seasonal_periods`. Only used if initialization is 'known'.
        """
        self.initialization_method = string_like(
            initialization_method,
            "initialization_method",
            options=("estimated", "known", "heuristic"),
        )
        if self.initialization_method == "known":
            if initial_level is None:
                raise ValueError(
                    "`initial_level` argument must be provided"
                    ' when initialization method is set to "known".'
                )
            if self.has_trend and initial_trend is None:
                raise ValueError(
                    "`initial_trend` argument must be provided"
                    " for models with a trend component when"
                    ' initialization method is set to "known".'
                )
            if self.has_seasonal and initial_seasonal is None:
                raise ValueError(
                    "`initial_seasonal` argument must be provided"
                    " for models with a seasonal component when"
                    ' initialization method is set to "known".'
                )
        elif self.initialization_method == "heuristic":
            (
                initial_level,
                initial_trend,
                initial_seasonal,
            ) = _initialization_heuristic(
                self.endog,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            )
        elif self.initialization_method == "estimated":
            if self.nobs < 10 + 2 * (self.seasonal_periods // 2):
                (
                    initial_level,
                    initial_trend,
                    initial_seasonal,
                ) = _initialization_simple(
                    self.endog,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )
            else:
                (
                    initial_level,
                    initial_trend,
                    initial_seasonal,
                ) = _initialization_heuristic(
                    self.endog,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )
        if not self.has_trend:
            initial_trend = 0
        if not self.has_seasonal:
            initial_seasonal = 0
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal

        # we also have to reset the params index dictionaries
        self._internal_params_index = OrderedDict(
            zip(self._internal_param_names, np.arange(self._k_params_internal))
        )
        self._params_index = OrderedDict(
            zip(self.param_names, np.arange(self.k_params))
        )

    def set_bounds(self, bounds):
        """
        Set bounds for parameter estimation.

        Parameters
        ----------
        bounds : dict or None, optional
            A dictionary with parameter names as keys and the respective bounds
            intervals as values (lists/tuples/arrays).
            The available parameter names are in ``self.param_names``.
            The default option is ``None``, in which case the traditional
            (nonlinear) bounds as described in [1]_ are used.

        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        if bounds is None:
            self.bounds = {}
        else:
            if not isinstance(bounds, (dict, OrderedDict)):
                raise ValueError("bounds must be a dictionary")
            for key in bounds:
                if key not in self.param_names:
                    raise ValueError(
                        f"Invalid key: {key} in bounds dictionary"
                    )
                bounds[key] = array_like(
                    bounds[key], f"bounds[{key}]", shape=(2,)
                )
            self.bounds = bounds

    @staticmethod
    def prepare_data(data):
        """
        Prepare data for use in the state space representation
        """
        endog = np.array(data.orig_endog, order="C")
        if endog.ndim != 1:
            raise ValueError("endog must be 1-dimensional")
        if endog.dtype != np.double:
            endog = np.asarray(data.orig_endog, order="C", dtype=float)
        return endog, None

    @property
    def nobs_effective(self):
        return self.nobs

    @property
    def k_endog(self):
        return 1

    @property
    def short_name(self):
        name = "".join(
            [
                str(s)[0].upper()
                for s in [self.error, self.trend, self.seasonal]
            ]
        )
        if self.damped_trend:
            name = name[0:2] + "d" + name[2]
        return name

    @property
    def _param_names(self):
        param_names = ["smoothing_level"]
        if self.has_trend:
            param_names += ["smoothing_trend"]
        if self.has_seasonal:
            param_names += ["smoothing_seasonal"]
        if self.damped_trend:
            param_names += ["damping_trend"]

        # Initialization
        if self.initialization_method == "estimated":
            param_names += ["initial_level"]
            if self.has_trend:
                param_names += ["initial_trend"]
            if self.has_seasonal:
                param_names += [
                    f"initial_seasonal.{i}"
                    for i in range(self.seasonal_periods)
                ]
        return param_names

    @property
    def state_names(self):
        names = ["level"]
        if self.has_trend:
            names += ["trend"]
        if self.has_seasonal:
            names += ["seasonal"]
        return names

    @property
    def initial_state_names(self):
        names = ["initial_level"]
        if self.has_trend:
            names += ["initial_trend"]
        if self.has_seasonal:
            names += [
                f"initial_seasonal.{i}" for i in range(self.seasonal_periods)
            ]
        return names

    @property
    def _smoothing_param_names(self):
        return [
            "smoothing_level",
            "smoothing_trend",
            "smoothing_seasonal",
            "damping_trend",
        ]

    @property
    def _internal_initial_state_names(self):
        param_names = [
            "initial_level",
            "initial_trend",
        ]
        param_names += [
            f"initial_seasonal.{i}" for i in range(self.seasonal_periods)
        ]
        return param_names

    @property
    def _internal_param_names(self):
        return self._smoothing_param_names + self._internal_initial_state_names

    @property
    def _k_states(self):
        return 1 + int(self.has_trend) + int(self.has_seasonal)  # level

    @property
    def _k_states_internal(self):
        return 2 + self.seasonal_periods

    @property
    def _k_smoothing_params(self):
        return self._k_states + int(self.damped_trend)

    @property
    def _k_initial_states(self):
        return (
            1
            + int(self.has_trend)
            + +int(self.has_seasonal) * self.seasonal_periods
        )

    @property
    def k_params(self):
        k = self._k_smoothing_params
        if self.initialization_method == "estimated":
            k += self._k_initial_states
        return k

    @property
    def _k_params_internal(self):
        return 4 + 2 + self.seasonal_periods

    def _internal_params(self, params):
        """
        Converts a parameter array passed from outside to the internally used
        full parameter array.
        """
        # internal params that are not needed are all set to zero, except phi,
        # which is one
        internal = np.zeros(self._k_params_internal, dtype=params.dtype)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            internal[internal_idx] = params[i]
        if not self.damped_trend:
            internal[3] = 1  # phi is 4th parameter
        if self.initialization_method != "estimated":
            internal[4] = self.initial_level
            internal[5] = self.initial_trend
            if np.isscalar(self.initial_seasonal):
                internal[6:] = self.initial_seasonal
            else:
                # See GH 7893
                internal[6:] = self.initial_seasonal[::-1]
        return internal

    def _model_params(self, internal):
        """
        Converts internal parameters to model parameters
        """
        params = np.empty(self.k_params)
        for i, name in enumerate(self.param_names):
            internal_idx = self._internal_params_index[name]
            params[i] = internal[internal_idx]
        return params

    @property
    def _seasonal_index(self):
        return 1 + int(self.has_trend)

    def _get_states(self, xhat):
        states = np.empty((self.nobs, self._k_states))
        all_names = ["level", "trend", "seasonal"]
        for i, name in enumerate(self.state_names):
            idx = all_names.index(name)
            states[:, i] = xhat[:, idx]
        return states

    def _get_internal_states(self, states, params):
        """
        Converts a state matrix/dataframe to the (nobs, 2+m) matrix used
        internally
        """
        internal_params = self._internal_params(params)
        if isinstance(states, (pd.Series, pd.DataFrame)):
            states = states.values
        internal_states = np.zeros((self.nobs, 2 + self.seasonal_periods))
        internal_states[:, 0] = states[:, 0]
        if self.has_trend:
            internal_states[:, 1] = states[:, 1]
        if self.has_seasonal:
            for j in range(self.seasonal_periods):
                internal_states[j:, 2 + j] = states[
                    0 : self.nobs - j, self._seasonal_index
                ]
                internal_states[0:j, 2 + j] = internal_params[6 : 6 + j][::-1]
        return internal_states

    @property
    def _default_start_params(self):
        return {
            "smoothing_level": 0.1,
            "smoothing_trend": 0.01,
            "smoothing_seasonal": 0.01,
            "damping_trend": 0.98,
        }

    @property
    def _start_params(self):
        """
        Default start params in the format of external parameters.
        This should not be called directly, but by calling
        ``self.start_params``.
        """
        params = []
        for p in self._smoothing_param_names:
            if p in self.param_names:
                params.append(self._default_start_params[p])

        if self.initialization_method == "estimated":
            lvl_idx = len(params)
            params += [self.initial_level]
            if self.has_trend:
                params += [self.initial_trend]
            if self.has_seasonal:
                # we have to adapt the seasonal values a bit to make sure the
                # problem is well posed (see implementation notes above)
                initial_seasonal = self.initial_seasonal
                if self.seasonal == "mul":
                    params[lvl_idx] *= initial_seasonal[-1]
                    initial_seasonal /= initial_seasonal[-1]
                else:
                    params[lvl_idx] += initial_seasonal[-1]
                    initial_seasonal -= initial_seasonal[-1]
                params += initial_seasonal.tolist()

        return np.array(params)

    def _convert_and_bound_start_params(self, params):
        """
        This converts start params to internal params, sets internal-only
        parameters as bounded, sets bounds for fixed parameters, and then makes
        sure that all start parameters are within the specified bounds.
        """
        internal_params = self._internal_params(params)
        # set bounds for missing and fixed
        for p in self._internal_param_names:
            idx = self._internal_params_index[p]
            if p not in self.param_names:
                # any missing parameters are set to the value they got from the
                # call to _internal_params
                self.bounds[p] = [internal_params[idx]] * 2
            elif self._has_fixed_params and p in self._fixed_params:
                self.bounds[p] = [self._fixed_params[p]] * 2
            # make sure everything is within bounds
            if p in self.bounds:
                internal_params[idx] = np.clip(
                    internal_params[idx]
                    + 1e-3,  # try not to start on boundary
                    *self.bounds[p],
                )
        return internal_params

    def _setup_bounds(self):
        # By default, we are using the traditional constraints for the
        # smoothing parameters if nothing else is specified
        #
        #    0 <     alpha     < 1
        #    0 <   beta/alpha  < 1
        #    0 < gamma + alpha < 1
        #  0.8 <      phi      < 0.98
        #
        # For initial states, no bounds are the default setting.
        #
        # Since the bounds for beta and gamma are not in the simple form of a
        # constant interval, we will use the parameters beta_star=beta/alpha
        # and gamma_star=gamma+alpha during fitting.

        lb = np.zeros(self._k_params_internal) + 1e-4
        ub = np.ones(self._k_params_internal) - 1e-4

        # other bounds for phi and initial states
        lb[3], ub[3] = 0.8, 0.98
        if self.initialization_method == "estimated":
            lb[4:-1] = -np.inf
            ub[4:-1] = np.inf
            # fix the last initial_seasonal to 0 or 1, otherwise the equation
            # is underdetermined
            if self.seasonal == "mul":
                lb[-1], ub[-1] = 1, 1
            else:
                lb[-1], ub[-1] = 0, 0

        # set lb and ub for parameters with bounds
        for p in self._internal_param_names:
            idx = self._internal_params_index[p]
            if p in self.bounds:
                lb[idx], ub[idx] = self.bounds[p]

        return [(lb[i], ub[i]) for i in range(self._k_params_internal)]

    def fit(
        self,
        start_params=None,
        maxiter=1000,
        full_output=True,
        disp=True,
        callback=None,
        return_params=False,
        **kwargs,
    ):
        r"""
        Fit an ETS model by maximizing log-likelihood.

        Log-likelihood is a function of the model parameters :math:`\alpha,
        \beta, \gamma, \phi` (depending on the chosen model), and, if
        `initialization_method` was set to `'estimated'` in the constructor,
        also the initial states :math:`l_{-1}, b_{-1}, s_{-1}, \ldots, s_{-m}`.

        The fit is performed using the L-BFGS algorithm.

        Parameters
        ----------
        start_params : array_like, optional
            Initial values for parameters that will be optimized. If this is
            ``None``, default values will be used.
            The length of this depends on the chosen model. This should contain
            the parameters in the following order, skipping parameters that do
            not exist in the chosen model.

            * `smoothing_level` (:math:`\alpha`)
            * `smoothing_trend` (:math:`\beta`)
            * `smoothing_seasonal` (:math:`\gamma`)
            * `damping_trend` (:math:`\phi`)

            If ``initialization_method`` was set to ``'estimated'`` (the
            default), additionally, the parameters

            * `initial_level` (:math:`l_{-1}`)
            * `initial_trend` (:math:`l_{-1}`)
            * `initial_seasonal.0` (:math:`s_{-1}`)
            * ...
            * `initial_seasonal.<m-1>` (:math:`s_{-m}`)

            also have to be specified.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : bool, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        results : ETSResults
        """

        if start_params is None:
            start_params = self.start_params
        else:
            start_params = np.asarray(start_params)

        if self._has_fixed_params and len(self._free_params_index) == 0:
            final_params = np.asarray(list(self._fixed_params.values()))
            mlefit = Bunch(
                params=start_params, mle_retvals=None, mle_settings=None
            )
        else:
            internal_start_params = self._convert_and_bound_start_params(
                start_params
            )
            bounds = self._setup_bounds()

            # check if we need to use the starred parameters
            use_beta_star = "smoothing_trend" not in self.bounds
            if use_beta_star:
                internal_start_params[1] /= internal_start_params[0]
            use_gamma_star = "smoothing_seasonal" not in self.bounds
            if use_gamma_star:
                internal_start_params[2] /= 1 - internal_start_params[0]

            # check if we have fixed parameters and remove them from the
            # parameter vector
            is_fixed = np.zeros(self._k_params_internal, dtype=int)
            fixed_values = np.empty_like(internal_start_params)
            params_without_fixed = []
            kwargs["bounds"] = []
            for i in range(self._k_params_internal):
                if bounds[i][0] == bounds[i][1]:
                    is_fixed[i] = True
                    fixed_values[i] = bounds[i][0]
                else:
                    params_without_fixed.append(internal_start_params[i])
                    kwargs["bounds"].append(bounds[i])
            params_without_fixed = np.asarray(params_without_fixed)

            # pre-allocate memory for smoothing results
            yhat = np.zeros(self.nobs)
            xhat = np.zeros((self.nobs, self._k_states_internal))

            kwargs["approx_grad"] = True
            with self.use_internal_loglike():
                mlefit = super().fit(
                    params_without_fixed,
                    fargs=(
                        yhat,
                        xhat,
                        is_fixed,
                        fixed_values,
                        use_beta_star,
                        use_gamma_star,
                    ),
                    method="lbfgs",
                    maxiter=maxiter,
                    full_output=full_output,
                    disp=disp,
                    callback=callback,
                    skip_hessian=True,
                    **kwargs,
                )
            # convert params back
            # first, insert fixed params
            fitted_params = np.empty_like(internal_start_params)
            idx_without_fixed = 0
            for i in range(self._k_params_internal):
                if is_fixed[i]:
                    fitted_params[i] = fixed_values[i]
                else:
                    fitted_params[i] = mlefit.params[idx_without_fixed]
                    idx_without_fixed += 1

            if use_beta_star:
                fitted_params[1] *= fitted_params[0]
            if use_gamma_star:
                fitted_params[2] *= 1 - fitted_params[0]
            final_params = self._model_params(fitted_params)

        if return_params:
            return final_params
        else:
            result = self.smooth(final_params)
            result.mlefit = mlefit
            result.mle_retvals = mlefit.mle_retvals
            result.mle_settings = mlefit.mle_settings
            return result

    def _loglike_internal(
        self,
        params,
        yhat,
        xhat,
        is_fixed=None,
        fixed_values=None,
        use_beta_star=False,
        use_gamma_star=False,
    ):
        """
        Log-likelihood function to be called from fit to avoid reallocation of
        memory.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m]). If there are no fixed values this must
            be in the format of internal parameters. Otherwise the fixed values
            are skipped.
        yhat : np.ndarray
            Array of size (n,) where fitted values will be written to.
        xhat : np.ndarray
            Array of size (n, _k_states_internal) where fitted states will be
            written to.
        is_fixed : np.ndarray or None
            Boolean array indicating values which are fixed during fitting.
            This must have the full length of internal parameters.
        fixed_values : np.ndarray or None
            Array of fixed values (arbitrary values for non-fixed parameters)
            This must have the full length of internal parameters.
        use_beta_star : boolean
            Whether to internally use beta_star as parameter
        use_gamma_star : boolean
            Whether to internally use gamma_star as parameter
        """
        if np.iscomplexobj(params):
            data = np.asarray(self.endog, dtype=complex)
        else:
            data = self.endog

        if is_fixed is None:
            is_fixed = np.zeros(self._k_params_internal, dtype=int)
            fixed_values = np.empty(
                self._k_params_internal, dtype=params.dtype
            )

        self._smoothing_func(
            params,
            data,
            yhat,
            xhat,
            is_fixed,
            fixed_values,
            use_beta_star,
            use_gamma_star,
        )
        res = self._residuals(yhat, data=data)
        logL = -self.nobs / 2 * (np.log(2 * np.pi * np.mean(res ** 2)) + 1)
        if self.error == "mul":
            # GH-7331: in some cases, yhat can become negative, so that a
            # multiplicative model is no longer well-defined. To avoid these
            # parameterizations, we clip negative values to very small positive
            # values so that the log-transformation yields very large negative
            # values.
            yhat[yhat <= 0] = 1 / (1e-8 * (1 + np.abs(yhat[yhat <= 0])))
            logL -= np.sum(np.log(yhat))
        return logL

    @contextlib.contextmanager
    def use_internal_loglike(self):
        external_loglike = self.loglike
        self.loglike = self._loglike_internal
        try:
            yield
        finally:
            self.loglike = external_loglike

    def loglike(self, params, **kwargs):
        r"""
        Log-likelihood of model.

        Parameters
        ----------
        params : np.ndarray of np.float
            Model parameters: (alpha, beta, gamma, phi, l[-1],
            b[-1], s[-1], ..., s[-m])

        Notes
        -----
        The log-likelihood of a exponential smoothing model is [1]_:

        .. math::

           l(\theta, x_0|y) = - \frac{n}{2}(\log(2\pi s^2) + 1)
                              - \sum\limits_{t=1}^n \log(k_t)

        with

        .. math::

           s^2 = \frac{1}{n}\sum\limits_{t=1}^n \frac{(\hat{y}_t - y_t)^2}{k_t}

        where :math:`k_t = 1` for the additive error model and :math:`k_t =
        y_t` for the multiplicative error model.

        References
        ----------
        .. [1] J. K. Ord, A. B. Koehler R. D. and Snyder (1997). Estimation and
           Prediction for a Class of Dynamic Nonlinear Statistical Models.
           *Journal of the American Statistical Association*, 92(440),
           1621-1629
        """
        params = self._internal_params(np.asarray(params))
        yhat = np.zeros(self.nobs, dtype=params.dtype)
        xhat = np.zeros(
            (self.nobs, self._k_states_internal), dtype=params.dtype
        )
        return self._loglike_internal(np.asarray(params), yhat, xhat)

    def _residuals(self, yhat, data=None):
        """Calculates residuals of a prediction"""
        if data is None:
            data = self.endog
        if self.error == "mul":
            return (data - yhat) / yhat
        else:
            return data - yhat

    def _smooth(self, params):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters

        Returns
        -------
        yhat : pd.Series or np.ndarray
            Predicted values from exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.Series``, else a ``np.ndarray``.
        xhat : pd.DataFrame or np.ndarray
            Internal states of exponential smoothing. If original data was a
            ``pd.Series``, returns a ``pd.DataFrame``, else a ``np.ndarray``.
        """
        internal_params = self._internal_params(params)
        yhat = np.zeros(self.nobs)
        xhat = np.zeros((self.nobs, self._k_states_internal))
        is_fixed = np.zeros(self._k_params_internal, dtype=int)
        fixed_values = np.empty(self._k_params_internal, dtype=params.dtype)
        self._smoothing_func(
            internal_params, self.endog, yhat, xhat, is_fixed, fixed_values
        )

        # remove states that are only internal
        states = self._get_states(xhat)

        if self.use_pandas:
            _, _, _, index = self._get_prediction_index(0, self.nobs - 1)
            yhat = pd.Series(yhat, index=index)
            statenames = ["level"]
            if self.has_trend:
                statenames += ["trend"]
            if self.has_seasonal:
                statenames += ["seasonal"]
            states = pd.DataFrame(states, index=index, columns=statenames)
        return yhat, states

    def smooth(self, params, return_raw=False):
        """
        Exponential smoothing with given parameters

        Parameters
        ----------
        params : array_like
            Model parameters
        return_raw : bool, optional
            Whether to return only the state space results or the full results
            object. Default is ``False``.

        Returns
        -------
        result : ETSResultsWrapper or tuple
            If ``return_raw=False``, returns a ETSResultsWrapper
            object. Otherwise a tuple of arrays or pandas objects, depending on
            the format of the endog data.
        """
        params = np.asarray(params)
        results = self._smooth(params)
        return self._wrap_results(params, results, return_raw)

    @property
    def _res_classes(self):
        return {"fit": (ETSResults, ETSResultsWrapper)}

    def hessian(
        self, params, approx_centered=False, approx_complex_step=True, **kwargs
    ):
        r"""
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the hessian.
        approx_centered : bool
            Whether to use a centered scheme for finite difference
            approximation
        approx_complex_step : bool
            Whether to use complex step differentiation for approximation

        Returns
        -------
        hessian : ndarray
            Hessian matrix evaluated at `params`

        Notes
        -----
        This is a numerical approximation.
        """
        method = kwargs.get("method", "approx")

        if method == "approx":
            if approx_complex_step:
                hessian = self._hessian_complex_step(params, **kwargs)
            else:
                hessian = self._hessian_finite_difference(
                    params, approx_centered=approx_centered, **kwargs
                )
        else:
            raise NotImplementedError("Invalid Hessian calculation method.")

        return hessian

    def score(
        self, params, approx_centered=False, approx_complex_step=True, **kwargs
    ):
        method = kwargs.get("method", "approx")

        if method == "approx":
            if approx_complex_step:
                score = self._score_complex_step(params, **kwargs)
            else:
                score = self._score_finite_difference(
                    params, approx_centered=approx_centered, **kwargs
                )
        else:
            raise NotImplementedError("Invalid score method.")

        return score

    def update(params, *args, **kwargs):
        # Dummy method to make methods copied from statespace.MLEModel work
        ...


class ETSResults(base.StateSpaceMLEResults):
    """
    Results from an error, trend, seasonal (ETS) exponential smoothing model
    """
    def __init__(self, model, params, results):
        yhat, xhat = results
        self._llf = model.loglike(params)
        self._residuals = model._residuals(yhat)
        self._fittedvalues = yhat
        # scale is concentrated in this model formulation and corresponds to
        # mean squared residuals, see docstring of model.loglike
        scale = np.mean(self._residuals ** 2)
        super().__init__(model, params, scale=scale)

        # get model definition
        model_definition_attrs = [
            "short_name",
            "error",
            "trend",
            "seasonal",
            "damped_trend",
            "has_trend",
            "has_seasonal",
            "seasonal_periods",
            "initialization_method",
        ]
        for attr in model_definition_attrs:
            setattr(self, attr, getattr(model, attr))
        self.param_names = [
            "%s (fixed)" % name if name in self.fixed_params else name
            for name in (self.model.param_names or [])
        ]

        # get fitted states and parameters
        internal_params = self.model._internal_params(params)
        self.states = xhat
        if self.model.use_pandas:
            states = self.states.iloc
        else:
            states = self.states
        self.initial_state = np.zeros(model._k_initial_states)

        self.level = states[:, 0]
        self.initial_level = internal_params[4]
        self.initial_state[0] = self.initial_level
        self.alpha = self.params[0]
        self.smoothing_level = self.alpha
        if self.has_trend:
            self.slope = states[:, 1]
            self.initial_trend = internal_params[5]
            self.initial_state[1] = self.initial_trend
            self.beta = self.params[1]
            self.smoothing_trend = self.beta
        if self.has_seasonal:
            self.season = states[:, self.model._seasonal_index]
            # See GH 7893
            self.initial_seasonal = internal_params[6:][::-1]
            self.initial_state[
                self.model._seasonal_index :
            ] = self.initial_seasonal
            self.gamma = self.params[self.model._seasonal_index]
            self.smoothing_seasonal = self.gamma
        if self.damped_trend:
            self.phi = internal_params[3]
            self.damping_trend = self.phi

        # degrees of freedom of model
        k_free_params = self.k_params - len(self.fixed_params)
        self.df_model = k_free_params + 1

        # standardized forecasting error
        self.mean_resid = np.mean(self.resid)
        self.scale_resid = np.std(self.resid, ddof=1)
        self.standardized_forecasts_error = (
            self.resid - self.mean_resid
        ) / self.scale_resid

        # Setup covariance matrix notes dictionary
        # For now, only support "approx"
        if not hasattr(self, "cov_kwds"):
            self.cov_kwds = {}
        self.cov_type = "approx"

        # Setup the cache
        self._cache = {}

        # Handle covariance matrix calculation
        self._cov_approx_complex_step = True
        self._cov_approx_centered = False
        approx_type_str = "complex-step"
        try:
            self._rank = None
            if self.k_params == 0:
                self.cov_params_default = np.zeros((0, 0))
                self._rank = 0
                self.cov_kwds["description"] = "No parameters estimated."
            else:
                self.cov_params_default = self.cov_params_approx
                self.cov_kwds["description"] = descriptions["approx"].format(
                    approx_type=approx_type_str
                )
        except np.linalg.LinAlgError:
            self._rank = 0
            k_params = len(self.params)
            self.cov_params_default = np.zeros((k_params, k_params)) * np.nan
            self.cov_kwds["cov_type"] = (
                "Covariance matrix could not be calculated: singular."
                " information matrix."
            )

    @cache_readonly
    def nobs_effective(self):
        return self.nobs

    @cache_readonly
    def fittedvalues(self):
        return self._fittedvalues

    @cache_readonly
    def resid(self):
        return self._residuals

    @cache_readonly
    def llf(self):
        """
        log-likelihood function evaluated at the fitted params
        """
        return self._llf

    def _get_prediction_params(self, start_idx):
        """
        Returns internal parameter representation of smoothing parameters and
        "initial" states for prediction/simulation, that is the states just
        before the first prediction/simulation step.
        """
        internal_params = self.model._internal_params(self.params)
        if start_idx == 0:
            return internal_params
        else:
            internal_states = self.model._get_internal_states(
                self.states, self.params
            )
            start_state = np.empty(6 + self.seasonal_periods)
            start_state[0:4] = internal_params[0:4]
            start_state[4:] = internal_states[start_idx - 1, :]
            return start_state

    def _relative_forecast_variance(self, steps):
        """
        References
        ----------
        .. [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
           principles and practice*, 3rd edition, OTexts: Melbourne,
           Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
        """
        h = steps
        alpha = self.smoothing_level
        if self.has_trend:
            beta = self.smoothing_trend
        if self.has_seasonal:
            gamma = self.smoothing_seasonal
            m = self.seasonal_periods
            k = np.asarray((h - 1) / m, dtype=int)
        if self.damped_trend:
            phi = self.damping_trend
        model = self.model.short_name
        if model == "ANN":
            return 1 + alpha ** 2 * (h - 1)
        elif model == "AAN":
            return 1 + (h - 1) * (
                alpha ** 2 + alpha * beta * h + beta ** 2 * h / 6 * (2 * h - 1)
            )
        elif model == "AAdN":
            return (
                1
                + alpha ** 2 * (h - 1)
                + (
                    (beta * phi * h)
                    / ((1 - phi) ** 2)
                    * (2 * alpha * (1 - phi) + beta * phi)
                )
                - (
                    (beta * phi * (1 - phi ** h))
                    / ((1 - phi) ** 2 * (1 - phi ** 2))
                    * (
                        2 * alpha * (1 - phi ** 2)
                        + beta * phi * (1 + 2 * phi - phi ** h)
                    )
                )
            )
        elif model == "ANA":
            return 1 + alpha ** 2 * (h - 1) + gamma * k * (2 * alpha + gamma)
        elif model == "AAA":
            return (
                1
                + (h - 1)
                * (
                    alpha ** 2
                    + alpha * beta * h
                    + (beta ** 2) / 6 * h * (2 * h - 1)
                )
                + gamma * k * (2 * alpha + gamma + beta * m * (k + 1))
            )
        elif model == "AAdA":
            return (
                1
                + alpha ** 2 * (h - 1)
                + gamma * k * (2 * alpha + gamma)
                + (beta * phi * h)
                / ((1 - phi) ** 2)
                * (2 * alpha * (1 - phi) + beta * phi)
                - (
                    (beta * phi * (1 - phi ** h))
                    / ((1 - phi) ** 2 * (1 - phi ** 2))
                    * (
                        2 * alpha * (1 - phi ** 2)
                        + beta * phi * (1 + 2 * phi - phi ** h)
                    )
                )
                + (
                    (2 * beta * gamma * phi)
                    / ((1 - phi) * (1 - phi ** m))
                    * (k * (1 - phi ** m) - phi ** m * (1 - phi ** (m * k)))
                )
            )
        else:
            raise NotImplementedError

    def simulate(
        self,
        nsimulations,
        anchor=None,
        repetitions=1,
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
            simulation at the first period of the sample (i.e. using the
            initial values as simulation anchor), and `end` refers to
            beginning the simulation at the first period after the sample.
            Integer values can run from 0 to `nobs`, or can be negative to
            apply negative indexing. Finally, if a date/time index was provided
            to the model, then this argument can be a date string to parse or a
            datetime type. Default is 'start'.
            Note: `anchor` corresponds to the observation right before the
            `start` observation in the `predict` method.
        repetitions : int, optional
            Number of simulated paths to generate. Default is 1 simulated path.
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
        """

        r"""
        Implementation notes
        --------------------
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

           B_t = b_{t-1} \circ_d \phi\\
           L_t = l_{t-1} \circ_b B_t\\
           S_t = s_{t-m}\\
           Y_t = L_t \circ_s S_t,

        where :math:`\circ_d` is the operation linking trend and damping
        parameter (multiplication if the trend is additive, power if the trend
        is multiplicative), :math:`\circ_b` is the operation linking level and
        trend (addition if the trend is additive, multiplication if the trend
        is multiplicative), and :math:'\circ_s` is the operation linking
        seasonality to the rest.

        The state space equations can then be formulated as

        .. math::

           y_t = Y_t + \eta \cdot e_t\\
           l_t = L_t + \alpha \cdot (M_e \cdot L_t + \kappa_l) \cdot e_t\\
           b_t = B_t + \beta \cdot (M_e \cdot B_t+\kappa_b) \cdot e_t\\
           s_t = S_t + \gamma \cdot (M_e \cdot S_t + \kappa_s) \cdot e_t\\

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
                       \frac{1}{L_t}\quad
                       \text{if seasonality is multiplicative}\\
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
        # Get the starting location
        start_idx = self._get_prediction_start_index(anchor)

        # set initial values and obtain parameters
        start_params = self._get_prediction_params(start_idx)
        x = np.zeros((nsimulations, self.model._k_states_internal))
        # is fixed and fixed values are dummy arguments
        is_fixed = np.zeros(len(start_params), dtype=int)
        fixed_values = np.zeros_like(start_params)
        (
            alpha,
            beta_star,
            gamma_star,
            phi,
            m,
            _,
        ) = smooth._initialize_ets_smooth(
            start_params, x, is_fixed, fixed_values
        )
        beta = alpha * beta_star
        gamma = (1 - alpha) * gamma_star
        # make x a 3 dimensional matrix: first dimension is nsimulations
        # (number of steps), next is number of states, innermost is repetitions
        nstates = x.shape[1]
        x = np.tile(np.reshape(x, (nsimulations, nstates, 1)), repetitions)
        y = np.empty((nsimulations, repetitions))

        # get random error eps
        sigma = np.sqrt(self.scale)
        if isinstance(random_errors, np.ndarray):
            if random_errors.shape != (nsimulations, repetitions):
                raise ValueError(
                    "If random is an ndarray, it must have shape "
                    "(nsimulations, repetitions)!"
                )
            eps = random_errors
        elif random_errors == "bootstrap":
            eps = np.random.choice(
                self.resid, size=(nsimulations, repetitions), replace=True
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
            params = random_errors.fit(self.resid)
            eps = random_errors.rvs(*params, size=(nsimulations, repetitions))
        elif isinstance(random_errors, rv_frozen):
            eps = random_errors.rvs(size=(nsimulations, repetitions))
        else:
            raise ValueError("Argument random_errors has unexpected value!")

        # get model settings
        mul_seasonal = self.seasonal == "mul"
        mul_trend = self.trend == "mul"
        mul_error = self.error == "mul"

        # define trend, damping and seasonality operations
        if mul_trend:
            op_b = np.multiply
            op_d = np.power
        else:
            op_b = np.add
            op_d = np.multiply
        if mul_seasonal:
            op_s = np.multiply
        else:
            op_s = np.add

        # x translation:
        # - x[t, 0, :] is level[t]
        # - x[t, 1, :] is trend[t]
        # - x[t, 2, :] is seasonal[t]
        # - x[t, 3, :] is seasonal[t-1]
        # - x[t, 2+j, :] is seasonal[t-j]
        # - similarly: x[t-1, 2+m-1, :] is seasonal[t-m]
        for t in range(nsimulations):
            B = op_d(x[t - 1, 1, :], phi)
            L = op_b(x[t - 1, 0, :], B)
            S = x[t - 1, 2 + m - 1, :]
            Y = op_s(L, S)
            if self.error == "add":
                eta = 1
                kappa_l = 1 / S if mul_seasonal else 1
                kappa_b = kappa_l / x[t - 1, 0, :] if mul_trend else kappa_l
                kappa_s = 1 / L if mul_seasonal else 1
            else:
                eta = Y
                kappa_l = 0 if mul_seasonal else S
                kappa_b = (
                    kappa_l / x[t - 1, 0, :]
                    if mul_trend
                    else kappa_l + x[t - 1, 0, :]
                )
                kappa_s = 0 if mul_seasonal else L

            y[t, :] = Y + eta * eps[t, :]
            x[t, 0, :] = L + alpha * (mul_error * L + kappa_l) * eps[t, :]
            x[t, 1, :] = B + beta * (mul_error * B + kappa_b) * eps[t, :]
            x[t, 2, :] = S + gamma * (mul_error * S + kappa_s) * eps[t, :]
            # update seasonals by shifting previous seasonal right
            x[t, 3:, :] = x[t - 1, 2:-1, :]

        # Wrap data / squeeze where appropriate
        if repetitions > 1:
            names = ["simulation.%d" % num for num in range(repetitions)]
        else:
            names = "simulation"
        return self.model._wrap_data(
            y, start_idx, start_idx + nsimulations - 1, names=names
        )

    def forecast(self, steps=1):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int, str, or datetime, optional
            If an integer, the number of steps to forecast from the end of the
            sample. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency, steps
            must be an integer. Default

        Returns
        -------
        forecast : ndarray
            Array of out of sample forecasts. A (steps x k_endog) array.
        """
        return self._forecast(steps, "end")

    def _forecast(self, steps, anchor):
        """
        Dynamic prediction/forecasting
        """
        # forecast is the same as simulation without errors
        return self.simulate(
            steps, anchor=anchor, random_errors=np.zeros((steps, 1))
        )

    def _handle_prediction_index(self, start, dynamic, end, index):
        if start is None:
            start = 0

        # Handle start, end, dynamic
        start, end, out_of_sample, _ = self.model._get_prediction_index(
            start, end, index
        )
        # if end was outside of the sample, it is now the last point in the
        # sample
        if start > end + out_of_sample + 1:
            raise ValueError(
                "Prediction start cannot lie outside of the sample."
            )

        # Handle `dynamic`
        if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
            dynamic, _, _ = self.model._get_index_loc(dynamic)
            # Convert to offset relative to start
            dynamic = dynamic - start
        elif isinstance(dynamic, bool):
            if dynamic:
                dynamic = 0
            else:
                dynamic = end + 1 - start

        # start : index of first predicted value
        # dynamic : offset to first dynamically predicted value
        #     -> if dynamic == 0, only dynamic simulations
        if dynamic == 0:
            start_smooth = None
            end_smooth = None
            nsmooth = 0
            start_dynamic = start
        else:
            # dynamic simulations from start + dynamic
            start_smooth = start
            end_smooth = min(start + dynamic - 1, end)
            nsmooth = max(end_smooth - start_smooth + 1, 0)
            start_dynamic = start + dynamic
        # anchor for simulations is one before start_dynamic
        if start_dynamic == 0:
            anchor_dynamic = "start"
        else:
            anchor_dynamic = start_dynamic - 1
        # end is last point in sample, out_of_sample gives number of
        # simulations out of sample
        end_dynamic = end + out_of_sample
        ndynamic = end_dynamic - start_dynamic + 1
        return (
            start,
            end,
            start_smooth,
            end_smooth,
            anchor_dynamic,
            start_dynamic,
            end_dynamic,
            nsmooth,
            ndynamic,
            index,
        )

    def predict(self, start=None, end=None, dynamic=False, index=None):
        """
        In-sample prediction and out-of-sample forecasting

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
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.

        Returns
        -------
        forecast : array_like or pd.Series.
            Array of out of in-sample predictions and / or out-of-sample
            forecasts. An (npredict,) array. If original data was a pd.Series
            or DataFrame, a pd.Series is returned.
        """

        (
            start,
            end,
            start_smooth,
            end_smooth,
            anchor_dynamic,
            _,
            end_dynamic,
            nsmooth,
            ndynamic,
            index,
        ) = self._handle_prediction_index(start, dynamic, end, index)

        y = np.empty(nsmooth + ndynamic)

        # In sample nondynamic prediction: smoothing
        if nsmooth > 0:
            y[0:nsmooth] = self.fittedvalues[start_smooth : end_smooth + 1]

        # Out of sample/dynamic prediction: forecast
        if ndynamic > 0:
            y[nsmooth:] = self._forecast(ndynamic, anchor_dynamic)

        # when we are doing out of sample only prediction, start > end + 1, and
        # we only want to output beginning at start
        if start > end + 1:
            ndiscard = start - (end + 1)
            y = y[ndiscard:]

        # Wrap data / squeeze where appropriate
        return self.model._wrap_data(y, start, end_dynamic)

    def get_prediction(
        self,
        start=None,
        end=None,
        dynamic=False,
        index=None,
        method=None,
        simulate_repetitions=1000,
        **simulate_kwargs,
    ):
        """
        Calculates mean prediction and prediction intervals.

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
            want out of sample prediction. Default is the last observation in
            the sample.
        dynamic : bool, int, str, or datetime, optional
            Integer offset relative to `start` at which to begin dynamic
            prediction. Can also be an absolute date string to parse or a
            datetime type (these are not interpreted as offsets).
            Prior to this observation, true endogenous values will be used for
            prediction; starting with this observation and continuing through
            the end of prediction, forecasted endogenous values will be used
            instead.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        method : str or None, optional
            Method to use for calculating prediction intervals. 'exact'
            (default, if available) or 'simulated'.
        simulate_repetitions : int, optional
            Number of simulation repetitions for calculating prediction
            intervals when ``method='simulated'``. Default is 1000.
        **simulate_kwargs :
            Additional arguments passed to the ``simulate`` method.

        Returns
        -------
        PredictionResults
            Predicted mean values and prediction intervals
        """
        return PredictionResultsWrapper(
            PredictionResults(
                self,
                start,
                end,
                dynamic,
                index,
                method,
                simulate_repetitions,
                **simulate_kwargs,
            )
        )

    def summary(self, alpha=0.05, start=None):
        """
        Summarize the fitted model

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals. Default is 0.05.
        start : int, optional
            Integer of the start observation. Default is 0.

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        model_name = f"ETS({self.short_name})"

        summary = super().summary(
            alpha=alpha,
            start=start,
            title="ETS Results",
            model_name=model_name,
        )

        if self.model.initialization_method != "estimated":
            params = np.array(self.initial_state)
            if params.ndim > 1:
                params = params[0]
            names = self.model.initial_state_names
            param_header = [
                "initialization method: %s" % self.model.initialization_method
            ]
            params_stubs = names
            params_data = [
                [forg(params[i], prec=4)] for i in range(len(params))
            ]

            initial_state_table = SimpleTable(
                params_data, param_header, params_stubs, txt_fmt=fmt_params
            )
            summary.tables.insert(-1, initial_state_table)

        return summary


class ETSResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        "fittedvalues": "rows",
        "level": "rows",
        "resid": "rows",
        "season": "rows",
        "slope": "rows",
    }
    _wrap_attrs = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_attrs, _attrs
    )
    _methods = {"predict": "dates", "forecast": "dates"}
    _wrap_methods = wrap.union_dicts(
        tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods
    )


wrap.populate_wrapper(ETSResultsWrapper, ETSResults)


class PredictionResults:
    """
    ETS mean prediction and prediction intervals

    Parameters
    ----------
    results : ETSResults
        Model estimation results.
    start : int, str, or datetime, optional
        Zero-indexed observation number at which to start forecasting,
        i.e., the first forecast is start. Can also be a date string to
        parse or a datetime type. Default is the the zeroth observation.
    end : int, str, or datetime, optional
        Zero-indexed observation number at which to end forecasting, i.e.,
        the last forecast is end. Can also be a date string to
        parse or a datetime type. However, if the dates index does not
        have a fixed frequency, end must be an integer index if you
        want out of sample prediction. Default is the last observation in
        the sample.
    dynamic : bool, int, str, or datetime, optional
        Integer offset relative to `start` at which to begin dynamic
        prediction. Can also be an absolute date string to parse or a
        datetime type (these are not interpreted as offsets).
        Prior to this observation, true endogenous values will be used for
        prediction; starting with this observation and continuing through
        the end of prediction, forecasted endogenous values will be used
        instead.
    index : pd.Index, optional
        Optionally an index to associate the predicted results to. If None,
        an attempt is made to create an index for the predicted results
        from the model's index or model's row labels.
    method : str or None, optional
        Method to use for calculating prediction intervals. 'exact' (default,
        if available) or 'simulated'.
    simulate_repetitions : int, optional
        Number of simulation repetitions for calculating prediction intervals.
        Default is 1000.
    **simulate_kwargs :
        Additional arguments passed to the ``simulate`` method.
    """

    def __init__(
        self,
        results,
        start=None,
        end=None,
        dynamic=False,
        index=None,
        method=None,
        simulate_repetitions=1000,
        **simulate_kwargs,
    ):
        self.use_pandas = results.model.use_pandas

        if method is None:
            exact_available = ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"]
            if results.model.short_name in exact_available:
                method = "exact"
            else:
                method = "simulated"
        self.method = method

        (
            start,
            end,
            start_smooth,
            _,
            anchor_dynamic,
            start_dynamic,
            end_dynamic,
            nsmooth,
            ndynamic,
            index,
        ) = results._handle_prediction_index(start, dynamic, end, index)

        self.predicted_mean = results.predict(
            start=start, end=end_dynamic, dynamic=dynamic, index=index
        )
        self.row_labels = self.predicted_mean.index
        self.endog = np.empty(nsmooth + ndynamic) * np.nan
        if nsmooth > 0:
            self.endog[0: (end - start + 1)] = results.data.endog[
                start: (end + 1)
            ]
        self.model = Bunch(
            data=results.model.data.__class__(
                endog=self.endog, predict_dates=self.row_labels
            )
        )

        if self.method == "simulated":

            sim_results = []
            # first, perform "non-dynamic" simulations, i.e. simulations of
            # only one step, based on the previous step
            if nsmooth > 1:
                if start_smooth == 0:
                    anchor = "start"
                else:
                    anchor = start_smooth - 1
                for i in range(nsmooth):
                    sim_results.append(
                        results.simulate(
                            1,
                            anchor=anchor,
                            repetitions=simulate_repetitions,
                            **simulate_kwargs,
                        )
                    )
                    # anchor
                    anchor = start_smooth + i
            if ndynamic:
                sim_results.append(
                    results.simulate(
                        ndynamic,
                        anchor=anchor_dynamic,
                        repetitions=simulate_repetitions,
                        **simulate_kwargs,
                    )
                )
            if sim_results and isinstance(sim_results[0], pd.DataFrame):
                self.simulation_results = pd.concat(sim_results, axis=0)
            else:
                self.simulation_results = np.concatenate(sim_results, axis=0)
            self.forecast_variance = self.simulation_results.var(1)
        else:  # method == 'exact'
            steps = np.ones(ndynamic + nsmooth)
            if ndynamic > 0:
                steps[
                    (start_dynamic - min(start_dynamic, start)):
                    ] = range(1, ndynamic + 1)
            # when we are doing out of sample only prediction,
            # start > end + 1, and
            # we only want to output beginning at start
            if start > end + 1:
                ndiscard = start - (end + 1)
                steps = steps[ndiscard:]
            self.forecast_variance = (
                results.mse * results._relative_forecast_variance(steps)
            )

    @property
    def var_pred_mean(self):
        """The variance of the predicted mean"""
        return self.forecast_variance

    def pred_int(self, alpha=0.05):
        """
        Calculates prediction intervals by performing multiple simulations.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the prediction interval. Default is
            0.05, that is, a 95% prediction interval.
        """

        if self.method == "simulated":
            simulated_upper_pi = np.quantile(
                self.simulation_results, 1 - alpha / 2, axis=1
            )
            simulated_lower_pi = np.quantile(
                self.simulation_results, alpha / 2, axis=1
            )
            pred_int = np.vstack((simulated_lower_pi, simulated_upper_pi)).T
        else:
            q = norm.ppf(1 - alpha / 2)
            half_interval_size = q * np.sqrt(self.forecast_variance)
            pred_int = np.vstack(
                (
                    self.predicted_mean - half_interval_size,
                    self.predicted_mean + half_interval_size,
                )
            ).T

        if self.use_pandas:
            pred_int = pd.DataFrame(pred_int, index=self.row_labels)
            names = [
                f"lower PI (alpha={alpha:f})",
                f"upper PI (alpha={alpha:f})",
            ]
            pred_int.columns = names
        return pred_int

    def summary_frame(self, endog=0, alpha=0.05):
        pred_int = np.asarray(self.pred_int(alpha=alpha))
        to_include = {}
        to_include["mean"] = self.predicted_mean
        if self.method == "simulated":
            to_include["mean_numerical"] = np.mean(
                self.simulation_results, axis=1
            )
        to_include["pi_lower"] = pred_int[:, 0]
        to_include["pi_upper"] = pred_int[:, 1]

        res = pd.DataFrame(
            to_include, index=self.row_labels, columns=list(to_include.keys())
        )
        return res


class PredictionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        "predicted_mean": "dates",
        "simulation_results": "dates",
        "endog": "dates",
    }
    _wrap_attrs = wrap.union_dicts(_attrs)

    _methods = {}
    _wrap_methods = wrap.union_dicts(_methods)


wrap.populate_wrapper(PredictionResultsWrapper, PredictionResults)  # noqa:E305
