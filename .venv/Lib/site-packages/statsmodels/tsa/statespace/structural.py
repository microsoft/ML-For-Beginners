"""
Univariate structural time series models

TODO: tests: "** On entry to DLASCL, parameter number  4 had an illegal value"

Author: Chad Fulton
License: Simplified-BSD
"""

from warnings import warn

import numpy as np

from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat

from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
    companion_matrix, constrain_stationary_univariate,
    unconstrain_stationary_univariate, prepare_exog)

_mask_map = {
    1: 'irregular',
    2: 'fixed intercept',
    3: 'deterministic constant',
    6: 'random walk',
    7: 'local level',
    8: 'fixed slope',
    11: 'deterministic trend',
    14: 'random walk with drift',
    15: 'local linear deterministic trend',
    31: 'local linear trend',
    27: 'smooth trend',
    26: 'random trend'
}


class UnobservedComponents(MLEModel):
    r"""
    Univariate unobserved components time series model

    These are also known as structural time series models, and decompose a
    (univariate) time series into trend, seasonal, cyclical, and irregular
    components.

    Parameters
    ----------

    endog : array_like
        The observed time-series process :math:`y`
    level : {bool, str}, optional
        Whether or not to include a level component. Default is False. Can also
        be a string specification of the level / trend component; see Notes
        for available model specification strings.
    trend : bool, optional
        Whether or not to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal : {int, None}, optional
        The period of the seasonal component, if any. Default is None.
    freq_seasonal : {list[dict], None}, optional.
        Whether (and how) to model seasonal component(s) with trig. functions.
        If specified, there is one dictionary for each frequency-domain
        seasonal component.  Each dictionary must have the key, value pair for
        'period' -- integer and may have a key, value pair for
        'harmonics' -- integer. If 'harmonics' is not specified in any of the
        dictionaries, it defaults to the floor of period/2.
    cycle : bool, optional
        Whether or not to include a cycle component. Default is False.
    autoregressive : {int, None}, optional
        The order of the autoregressive component. Default is None.
    exog : {array_like, None}, optional
        Exogenous variables.
    irregular : bool, optional
        Whether or not to include an irregular component. Default is False.
    stochastic_level : bool, optional
        Whether or not any level component is stochastic. Default is False.
    stochastic_trend : bool, optional
        Whether or not any trend component is stochastic. Default is False.
    stochastic_seasonal : bool, optional
        Whether or not any seasonal component is stochastic. Default is True.
    stochastic_freq_seasonal : list[bool], optional
        Whether or not each seasonal component(s) is (are) stochastic.  Default
        is True for each component.  The list should be of the same length as
        freq_seasonal.
    stochastic_cycle : bool, optional
        Whether or not any cycle component is stochastic. Default is False.
    damped_cycle : bool, optional
        Whether or not the cycle component is damped. Default is False.
    cycle_period_bounds : tuple, optional
        A tuple with lower and upper allowed bounds for the period of the
        cycle. If not provided, the following default bounds are used:
        (1) if no date / time information is provided, the frequency is
        constrained to be between zero and :math:`\pi`, so the period is
        constrained to be in [0.5, infinity].
        (2) If the date / time information is provided, the default bounds
        allow the cyclical component to be between 1.5 and 12 years; depending
        on the frequency of the endogenous variable, this will imply different
        specific bounds.
    mle_regression : bool, optional
        Whether or not to estimate regression coefficients by maximum likelihood
        as one of hyperparameters. Default is True.
        If False, the regression coefficients are estimated by recursive OLS,
        included in the state vector.
    use_exact_diffuse : bool, optional
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).

    See Also
    --------
    statsmodels.tsa.statespace.structural.UnobservedComponentsResults
    statsmodels.tsa.statespace.mlemodel.MLEModel

    Notes
    -----

    These models take the general form (see [1]_ Chapter 3.2 for all details)

    .. math::

        y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
    seasonal component, :math:`c_t` refers to the cycle, and
    :math:`\varepsilon_t` is the irregular. The modeling details of these
    components are given below.

    **Trend**

    The trend component is a dynamic extension of a regression model that
    includes an intercept and linear time-trend. It can be written:

    .. math::

        \mu_t = \mu_{t-1} + \beta_{t-1} + \eta_{t-1} \\
        \beta_t = \beta_{t-1} + \zeta_{t-1}

    where the level is a generalization of the intercept term that can
    dynamically vary across time, and the trend is a generalization of the
    time-trend such that the slope can dynamically vary across time.

    Here :math:`\eta_t \sim N(0, \sigma_\eta^2)` and
    :math:`\zeta_t \sim N(0, \sigma_\zeta^2)`.

    For both elements (level and trend), we can consider models in which:

    - The element is included vs excluded (if the trend is included, there must
      also be a level included).
    - The element is deterministic vs stochastic (i.e. whether or not the
      variance on the error term is confined to be zero or not)

    The only additional parameters to be estimated via MLE are the variances of
    any included stochastic components.

    The level/trend components can be specified using the boolean keyword
    arguments `level`, `stochastic_level`, `trend`, etc., or all at once as a
    string argument to `level`. The following table shows the available
    model specifications:

    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Model name                       | Full string syntax                   | Abbreviated syntax | Model                                            |
    +==================================+======================================+====================+==================================================+
    | No trend                         | `'irregular'`                        | `'ntrend'`         | .. math:: y_t = \varepsilon_t                    |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed intercept                  | `'fixed intercept'`                  |                    | .. math:: y_t = \mu                              |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic constant           | `'deterministic constant'`           | `'dconstant'`      | .. math:: y_t = \mu + \varepsilon_t              |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local level                      | `'local level'`                      | `'llevel'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk                      | `'random walk'`                      | `'rwalk'`          | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed slope                      | `'fixed slope'`                      |                    | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic trend              | `'deterministic trend'`              | `'dtrend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear deterministic trend | `'local linear deterministic trend'` | `'lldtrend'`       | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk with drift           | `'random walk with drift'`           | `'rwdrift'`        | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear trend               | `'local linear trend'`               | `'lltrend'`        | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\ |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Smooth trend                     | `'smooth trend'`                     | `'strend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random trend                     | `'random trend'`                     | `'rtrend'`         | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+

    Following the fitting of the model, the unobserved level and trend
    component time series are available in the results class in the
    `level` and `trend` attributes, respectively.

    **Seasonal (Time-domain)**

    The seasonal component is modeled as:

    .. math::

        \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \\
        \omega_t \sim N(0, \sigma_\omega^2)

    The periodicity (number of seasons) is s, and the defining character is
    that (without the error term), the seasonal components sum to zero across
    one complete cycle. The inclusion of an error term allows the seasonal
    effects to vary over time (if this is not desired, :math:`\sigma_\omega^2`
    can be set to zero using the `stochastic_seasonal=False` keyword argument).

    This component results in one parameter to be selected via maximum
    likelihood: :math:`\sigma_\omega^2`, and one parameter to be chosen, the
    number of seasons `s`.

    Following the fitting of the model, the unobserved seasonal component
    time series is available in the results class in the `seasonal`
    attribute.

    **Frequency-domain Seasonal**

    Each frequency-domain seasonal component is modeled as:

    .. math::

        \gamma_t & =  \sum_{j=1}^h \gamma_{j, t} \\
        \gamma_{j, t+1} & = \gamma_{j, t}\cos(\lambda_j)
                        + \gamma^{*}_{j, t}\sin(\lambda_j) + \omega_{j,t} \\
        \gamma^{*}_{j, t+1} & = -\gamma^{(1)}_{j, t}\sin(\lambda_j)
                            + \gamma^{*}_{j, t}\cos(\lambda_j)
                            + \omega^{*}_{j, t}, \\
        \omega^{*}_{j, t}, \omega_{j, t} & \sim N(0, \sigma_{\omega^2}) \\
        \lambda_j & = \frac{2 \pi j}{s}

    where j ranges from 1 to h.

    The periodicity (number of "seasons" in a "year") is s and the number of
    harmonics is h.  Note that h is configurable to be less than s/2, but
    s/2 harmonics is sufficient to fully model all seasonal variations of
    periodicity s.  Like the time domain seasonal term (cf. Seasonal section,
    above), the inclusion of the error terms allows for the seasonal effects to
    vary over time.  The argument stochastic_freq_seasonal can be used to set
    one or more of the seasonal components of this type to be non-random,
    meaning they will not vary over time.

    This component results in one parameter to be fitted using maximum
    likelihood: :math:`\sigma_{\omega^2}`, and up to two parameters to be
    chosen, the number of seasons s and optionally the number of harmonics
    h, with :math:`1 \leq h \leq \lfloor s/2 \rfloor`.

    After fitting the model, each unobserved seasonal component modeled in the
    frequency domain is available in the results class in the `freq_seasonal`
    attribute.

    **Cycle**

    The cyclical component is intended to capture cyclical effects at time
    frames much longer than captured by the seasonal component. For example,
    in economics the cyclical term is often intended to capture the business
    cycle, and is then expected to have a period between "1.5 and 12 years"
    (see Durbin and Koopman).

    .. math::

        c_{t+1} & = \rho_c (\tilde c_t \cos \lambda_c t
                + \tilde c_t^* \sin \lambda_c) +
                \tilde \omega_t \\
        c_{t+1}^* & = \rho_c (- \tilde c_t \sin \lambda_c  t +
                \tilde c_t^* \cos \lambda_c) +
                \tilde \omega_t^* \\

    where :math:`\omega_t, \tilde \omega_t iid N(0, \sigma_{\tilde \omega}^2)`

    The parameter :math:`\lambda_c` (the frequency of the cycle) is an
    additional parameter to be estimated by MLE.

    If the cyclical effect is stochastic (`stochastic_cycle=True`), then there
    is another parameter to estimate (the variance of the error term - note
    that both of the error terms here share the same variance, but are assumed
    to have independent draws).

    If the cycle is damped (`damped_cycle=True`), then there is a third
    parameter to estimate, :math:`\rho_c`.

    In order to achieve cycles with the appropriate frequencies, bounds are
    imposed on the parameter :math:`\lambda_c` in estimation. These can be
    controlled via the keyword argument `cycle_period_bounds`, which, if
    specified, must be a tuple of bounds on the **period** `(lower, upper)`.
    The bounds on the frequency are then calculated from those bounds.

    The default bounds, if none are provided, are selected in the following
    way:

    1. If no date / time information is provided, the frequency is
       constrained to be between zero and :math:`\pi`, so the period is
       constrained to be in :math:`[0.5, \infty]`.
    2. If the date / time information is provided, the default bounds
       allow the cyclical component to be between 1.5 and 12 years; depending
       on the frequency of the endogenous variable, this will imply different
       specific bounds.

    Following the fitting of the model, the unobserved cyclical component
    time series is available in the results class in the `cycle`
    attribute.

    **Irregular**

    The irregular components are independent and identically distributed (iid):

    .. math::

        \varepsilon_t \sim N(0, \sigma_\varepsilon^2)

    **Autoregressive Irregular**

    An autoregressive component (often used as a replacement for the white
    noise irregular term) can be specified as:

    .. math::

        \varepsilon_t = \rho(L) \varepsilon_{t-1} + \epsilon_t \\
        \epsilon_t \sim N(0, \sigma_\epsilon^2)

    In this case, the AR order is specified via the `autoregressive` keyword,
    and the autoregressive coefficients are estimated.

    Following the fitting of the model, the unobserved autoregressive component
    time series is available in the results class in the `autoregressive`
    attribute.

    **Regression effects**

    Exogenous regressors can be pass to the `exog` argument. The regression
    coefficients will be estimated by maximum likelihood unless
    `mle_regression=False`, in which case the regression coefficients will be
    included in the state vector where they are essentially estimated via
    recursive OLS.

    If the regression_coefficients are included in the state vector, the
    recursive estimates are available in the results class in the
    `regression_coefficients` attribute.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """  # noqa:E501

    def __init__(self, endog, level=False, trend=False, seasonal=None,
                 freq_seasonal=None, cycle=False, autoregressive=None,
                 exog=None, irregular=False,
                 stochastic_level=False,
                 stochastic_trend=False,
                 stochastic_seasonal=True,
                 stochastic_freq_seasonal=None,
                 stochastic_cycle=False,
                 damped_cycle=False, cycle_period_bounds=None,
                 mle_regression=True, use_exact_diffuse=False,
                 **kwargs):

        # Model options
        self.level = level
        self.trend = trend
        self.seasonal_periods = seasonal if seasonal is not None else 0
        self.seasonal = self.seasonal_periods > 0
        if freq_seasonal:
            self.freq_seasonal_periods = [d['period'] for d in freq_seasonal]
            self.freq_seasonal_harmonics = [d.get(
                'harmonics', int(np.floor(d['period'] / 2))) for
                d in freq_seasonal]
        else:
            self.freq_seasonal_periods = []
            self.freq_seasonal_harmonics = []
        self.freq_seasonal = any(x > 0 for x in self.freq_seasonal_periods)
        self.cycle = cycle
        self.ar_order = autoregressive if autoregressive is not None else 0
        self.autoregressive = self.ar_order > 0
        self.irregular = irregular

        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        if stochastic_freq_seasonal is None:
            self.stochastic_freq_seasonal = [True] * len(
                self.freq_seasonal_periods)
        else:
            if len(stochastic_freq_seasonal) != len(freq_seasonal):
                raise ValueError(
                    "Length of stochastic_freq_seasonal must equal length"
                    " of freq_seasonal: {!r} vs {!r}".format(
                        len(stochastic_freq_seasonal), len(freq_seasonal)))
            self.stochastic_freq_seasonal = stochastic_freq_seasonal
        self.stochastic_cycle = stochastic_cycle

        self.damped_cycle = damped_cycle
        self.mle_regression = mle_regression
        self.use_exact_diffuse = use_exact_diffuse

        # Check for string trend/level specification
        self.trend_specification = None
        if isinstance(self.level, str):
            self.trend_specification = level
            self.level = False

            # Check if any of the trend/level components have been set, and
            # reset everything to False
            trend_attributes = ['irregular', 'level', 'trend',
                                'stochastic_level', 'stochastic_trend']
            for attribute in trend_attributes:
                if not getattr(self, attribute) is False:
                    warn("Value of `%s` may be overridden when the trend"
                         " component is specified using a model string."
                         % attribute, SpecificationWarning)
                    setattr(self, attribute, False)

            # Now set the correct specification
            spec = self.trend_specification
            if spec == 'irregular' or spec == 'ntrend':
                self.irregular = True
                self.trend_specification = 'irregular'
            elif spec == 'fixed intercept':
                self.level = True
            elif spec == 'deterministic constant' or spec == 'dconstant':
                self.irregular = True
                self.level = True
                self.trend_specification = 'deterministic constant'
            elif spec == 'local level' or spec == 'llevel':
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend_specification = 'local level'
            elif spec == 'random walk' or spec == 'rwalk':
                self.level = True
                self.stochastic_level = True
                self.trend_specification = 'random walk'
            elif spec == 'fixed slope':
                self.level = True
                self.trend = True
            elif spec == 'deterministic trend' or spec == 'dtrend':
                self.irregular = True
                self.level = True
                self.trend = True
                self.trend_specification = 'deterministic trend'
            elif (spec == 'local linear deterministic trend' or
                    spec == 'lldtrend'):
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.trend_specification = 'local linear deterministic trend'
            elif spec == 'random walk with drift' or spec == 'rwdrift':
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.trend_specification = 'random walk with drift'
            elif spec == 'local linear trend' or spec == 'lltrend':
                self.irregular = True
                self.level = True
                self.stochastic_level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'local linear trend'
            elif spec == 'smooth trend' or spec == 'strend':
                self.irregular = True
                self.level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'smooth trend'
            elif spec == 'random trend' or spec == 'rtrend':
                self.level = True
                self.trend = True
                self.stochastic_trend = True
                self.trend_specification = 'random trend'
            else:
                raise ValueError("Invalid level/trend specification: '%s'"
                                 % spec)

        # Check for a model that makes sense
        if trend and not level:
            warn("Trend component specified without level component;"
                 " deterministic level component added.", SpecificationWarning)
            self.level = True
            self.stochastic_level = False

        if not (self.irregular or
                (self.level and self.stochastic_level) or
                (self.trend and self.stochastic_trend) or
                (self.seasonal and self.stochastic_seasonal) or
                (self.freq_seasonal and any(
                    self.stochastic_freq_seasonal)) or
                (self.cycle and self.stochastic_cycle) or
                self.autoregressive):
            warn("Specified model does not contain a stochastic element;"
                 " irregular component added.", SpecificationWarning)
            self.irregular = True

        if self.seasonal and self.seasonal_periods < 2:
            raise ValueError('Seasonal component must have a seasonal period'
                             ' of at least 2.')

        if self.freq_seasonal:
            for p in self.freq_seasonal_periods:
                if p < 2:
                    raise ValueError(
                        'Frequency Domain seasonal component must have a '
                        'seasonal period of at least 2.')

        # Create a bitmask holding the level/trend specification
        self.trend_mask = (
            self.irregular * 0x01 |
            self.level * 0x02 |
            self.level * self.stochastic_level * 0x04 |
            self.trend * 0x08 |
            self.trend * self.stochastic_trend * 0x10
        )

        # Create the trend specification, if it was not given
        if self.trend_specification is None:
            # trend specification may be none, e.g. if the model is only
            # a stochastic cycle, etc.
            self.trend_specification = _mask_map.get(self.trend_mask, None)

        # Exogenous component
        (self.k_exog, exog) = prepare_exog(exog)

        self.regression = self.k_exog > 0

        # Model parameters
        self._k_seasonal_states = (self.seasonal_periods - 1) * self.seasonal
        self._k_freq_seas_states = (
            sum(2 * h for h in self.freq_seasonal_harmonics)
            * self.freq_seasonal)
        self._k_cycle_states = self.cycle * 2
        k_states = (
            self.level + self.trend +
            self._k_seasonal_states +
            self._k_freq_seas_states +
            self._k_cycle_states +
            self.ar_order +
            (not self.mle_regression) * self.k_exog
        )
        k_posdef = (
            self.stochastic_level * self.level +
            self.stochastic_trend * self.trend +
            self.stochastic_seasonal * self.seasonal +
            ((sum(2 * h if self.stochastic_freq_seasonal[ix] else 0 for
                  ix, h in enumerate(self.freq_seasonal_harmonics))) *
             self.freq_seasonal) +
            self.stochastic_cycle * (self._k_cycle_states) +
            self.autoregressive
        )

        # Handle non-default loglikelihood burn
        self._loglikelihood_burn = kwargs.get('loglikelihood_burn', None)

        # We can still estimate the model with just the irregular component,
        # just need to have one state that does nothing.
        self._unused_state = False
        if k_states == 0:
            if not self.irregular:
                raise ValueError('Model has no components specified.')
            k_states = 1
            self._unused_state = True
        if k_posdef == 0:
            k_posdef = 1

        # Setup the representation
        super(UnobservedComponents, self).__init__(
            endog, k_states, k_posdef=k_posdef, exog=exog, **kwargs
        )
        self.setup()

        # Set as time-varying model if we have exog
        if self.k_exog > 0:
            self.ssm._time_invariant = False

        # Need to reset the MLE names (since when they were first set, `setup`
        # had not been run (and could not have been at that point))
        self.data.param_names = self.param_names

        # Get bounds for the frequency of the cycle, if we know the frequency
        # of the data.
        if cycle_period_bounds is None:
            freq = self.data.freq[0] if self.data.freq is not None else ''
            if freq == 'A':
                cycle_period_bounds = (1.5, 12)
            elif freq == 'Q':
                cycle_period_bounds = (1.5*4, 12*4)
            elif freq == 'M':
                cycle_period_bounds = (1.5*12, 12*12)
            else:
                # If we have no information on data frequency, require the
                # cycle frequency to be between 0 and pi
                cycle_period_bounds = (2, np.inf)

        self.cycle_frequency_bound = (
            2*np.pi / cycle_period_bounds[1], 2*np.pi / cycle_period_bounds[0]
        )

        # Update _init_keys attached by super
        self._init_keys += ['level', 'trend', 'seasonal', 'freq_seasonal',
                            'cycle', 'autoregressive', 'irregular',
                            'stochastic_level', 'stochastic_trend',
                            'stochastic_seasonal', 'stochastic_freq_seasonal',
                            'stochastic_cycle',
                            'damped_cycle', 'cycle_period_bounds',
                            'mle_regression'] + list(kwargs.keys())

        # Initialize the state
        self.initialize_default()

    def _get_init_kwds(self):
        # Get keywords based on model attributes
        kwds = super(UnobservedComponents, self)._get_init_kwds()

        # Modifications
        if self.trend_specification is not None:
            kwds['level'] = self.trend_specification

            for attr in ['irregular', 'trend', 'stochastic_level',
                         'stochastic_trend']:
                kwds[attr] = False

        kwds['seasonal'] = self.seasonal_periods
        kwds['freq_seasonal'] = [
            {'period': p,
             'harmonics': self.freq_seasonal_harmonics[ix]} for
            ix, p in enumerate(self.freq_seasonal_periods)]
        kwds['autoregressive'] = self.ar_order

        return kwds

    def setup(self):
        """
        Setup the structural time series representation
        """
        # Initialize the ordered sets of parameters
        self.parameters = {}
        self.parameters_obs_intercept = {}
        self.parameters_obs_cov = {}
        self.parameters_transition = {}
        self.parameters_state_cov = {}

        # Initialize the fixed components of the state space matrices,
        i = 0  # state offset
        j = 0  # state covariance offset

        if self.irregular:
            self.parameters_obs_cov['irregular_var'] = 1
        if self.level:
            self.ssm['design', 0, i] = 1.
            self.ssm['transition', i, i] = 1.
            if self.trend:
                self.ssm['transition', i, i+1] = 1.
            if self.stochastic_level:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['level_var'] = 1
                j += 1
            i += 1
        if self.trend:
            self.ssm['transition', i, i] = 1.
            if self.stochastic_trend:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['trend_var'] = 1
                j += 1
            i += 1
        if self.seasonal:
            n = self.seasonal_periods - 1
            self.ssm['design', 0, i] = 1.
            self.ssm['transition', i:i + n, i:i + n] = (
                companion_matrix(np.r_[1, [1] * n]).transpose()
            )
            if self.stochastic_seasonal:
                self.ssm['selection', i, j] = 1.
                self.parameters_state_cov['seasonal_var'] = 1
                j += 1
            i += n
        if self.freq_seasonal:
            for ix, h in enumerate(self.freq_seasonal_harmonics):
                # These are the \gamma_jt and \gamma^*_jt terms in D&K (3.8)
                n = 2 * h
                p = self.freq_seasonal_periods[ix]
                lambda_p = 2 * np.pi / float(p)

                t = 0  # frequency transition matrix offset
                for block in range(1, h + 1):
                    # ibid. eqn (3.7)
                    self.ssm['design', 0, i+t] = 1.

                    # ibid. eqn (3.8)
                    cos_lambda_block = np.cos(lambda_p * block)
                    sin_lambda_block = np.sin(lambda_p * block)
                    trans = np.array([[cos_lambda_block, sin_lambda_block],
                                      [-sin_lambda_block, cos_lambda_block]])
                    trans_s = np.s_[i + t:i + t + 2]
                    self.ssm['transition', trans_s, trans_s] = trans
                    t += 2

                if self.stochastic_freq_seasonal[ix]:
                    self.ssm['selection', i:i + n, j:j + n] = np.eye(n)
                    cov_key = 'freq_seasonal_var_{!r}'.format(ix)
                    self.parameters_state_cov[cov_key] = 1
                    j += n
                i += n
        if self.cycle:
            self.ssm['design', 0, i] = 1.
            self.parameters_transition['cycle_freq'] = 1
            if self.damped_cycle:
                self.parameters_transition['cycle_damp'] = 1
            if self.stochastic_cycle:
                self.ssm['selection', i:i+2, j:j+2] = np.eye(2)
                self.parameters_state_cov['cycle_var'] = 1
                j += 2
            self._idx_cycle_transition = np.s_['transition', i:i+2, i:i+2]
            i += 2
        if self.autoregressive:
            self.ssm['design', 0, i] = 1.
            self.parameters_transition['ar_coeff'] = self.ar_order
            self.parameters_state_cov['ar_var'] = 1
            self.ssm['selection', i, j] = 1
            self.ssm['transition', i:i+self.ar_order, i:i+self.ar_order] = (
                companion_matrix(self.ar_order).T
            )
            self._idx_ar_transition = (
                np.s_['transition', i, i:i+self.ar_order]
            )
            j += 1
            i += self.ar_order
        if self.regression:
            if self.mle_regression:
                self.parameters_obs_intercept['reg_coeff'] = self.k_exog
            else:
                design = np.repeat(self.ssm['design', :, :, 0], self.nobs,
                                   axis=0)
                self.ssm['design'] = design.transpose()[np.newaxis, :, :]
                self.ssm['design', 0, i:i+self.k_exog, :] = (
                    self.exog.transpose())
                self.ssm['transition', i:i+self.k_exog, i:i+self.k_exog] = (
                    np.eye(self.k_exog)
                )

                i += self.k_exog

        # Update to get the actual parameter set
        self.parameters.update(self.parameters_obs_cov)
        self.parameters.update(self.parameters_state_cov)
        self.parameters.update(self.parameters_transition)  # ordered last
        self.parameters.update(self.parameters_obs_intercept)

        self.k_obs_intercept = sum(self.parameters_obs_intercept.values())
        self.k_obs_cov = sum(self.parameters_obs_cov.values())
        self.k_transition = sum(self.parameters_transition.values())
        self.k_state_cov = sum(self.parameters_state_cov.values())
        self.k_params = sum(self.parameters.values())

        # Other indices
        idx = np.diag_indices(self.ssm.k_posdef)
        self._idx_state_cov = ('state_cov', idx[0], idx[1])

        # Some of the variances may be tied together (repeated parameter usage)
        # Use list() for compatibility with python 3.5
        param_keys = list(self.parameters_state_cov.keys())
        self._var_repetitions = np.ones(self.k_state_cov, dtype=int)
        if self.freq_seasonal:
            for ix, is_stochastic in enumerate(self.stochastic_freq_seasonal):
                if is_stochastic:
                    num_harmonics = self.freq_seasonal_harmonics[ix]
                    repeat_times = 2 * num_harmonics
                    cov_key = 'freq_seasonal_var_{!r}'.format(ix)
                    cov_ix = param_keys.index(cov_key)
                    self._var_repetitions[cov_ix] = repeat_times

        if self.stochastic_cycle and self.cycle:
            cov_ix = param_keys.index('cycle_var')
            self._var_repetitions[cov_ix] = 2
        self._repeat_any_var = any(self._var_repetitions > 1)

    def initialize_default(self, approximate_diffuse_variance=None):
        if approximate_diffuse_variance is None:
            approximate_diffuse_variance = self.ssm.initial_variance
        if self.use_exact_diffuse:
            diffuse_type = 'diffuse'
        else:
            diffuse_type = 'approximate_diffuse'

            # Set the loglikelihood burn parameter, if not given in constructor
            if self._loglikelihood_burn is None:
                k_diffuse_states = (
                    self.k_states - int(self._unused_state) - self.ar_order)
                self.loglikelihood_burn = k_diffuse_states

        init = Initialization(
            self.k_states,
            approximate_diffuse_variance=approximate_diffuse_variance)

        if self._unused_state:
            # If this flag is set, it means we have a model with just an
            # irregular component and nothing else. The state is then
            # irrelevant and we can't put it as diffuse, since then the filter
            # will never leave the diffuse state.
            init.set(0, 'known', constant=[0])
        elif self.autoregressive:
            offset = (self.level + self.trend +
                      self._k_seasonal_states +
                      self._k_freq_seas_states +
                      self._k_cycle_states)
            length = self.ar_order
            init.set((0, offset), diffuse_type)
            init.set((offset, offset + length), 'stationary')
            init.set((offset + length, self.k_states), diffuse_type)
        # If we do not have an autoregressive component, then everything has
        # a diffuse initialization
        else:
            init.set(None, diffuse_type)

        self.ssm.initialization = init

    def clone(self, endog, exog=None, **kwargs):
        return self._clone_from_init_kwds(endog, exog=exog, **kwargs)

    @property
    def _res_classes(self):
        return {'fit': (UnobservedComponentsResults,
                        UnobservedComponentsResultsWrapper)}

    @property
    def start_params(self):
        if not hasattr(self, 'parameters'):
            return []

        # Eliminate missing data to estimate starting parameters
        endog = self.endog
        exog = self.exog
        if np.any(np.isnan(endog)):
            mask = ~np.isnan(endog).squeeze()
            endog = endog[mask]
            if exog is not None:
                exog = exog[mask]

        # Level / trend variances
        # (Use the HP filter to get initial estimates of variances)
        _start_params = {}
        if self.level:
            resid, trend1 = hpfilter(endog)

            if self.stochastic_trend:
                cycle2, trend2 = hpfilter(trend1)
                _start_params['trend_var'] = np.std(trend2)**2
                if self.stochastic_level:
                    _start_params['level_var'] = np.std(cycle2)**2
            elif self.stochastic_level:
                _start_params['level_var'] = np.std(trend1)**2
        else:
            resid = self.ssm.endog[0]

        # Regression
        if self.regression and self.mle_regression:
            _start_params['reg_coeff'] = (
                np.linalg.pinv(exog).dot(resid).tolist()
            )
            resid = np.squeeze(
                resid - np.dot(exog, _start_params['reg_coeff'])
            )

        # Autoregressive
        if self.autoregressive:
            Y = resid[self.ar_order:]
            X = lagmat(resid, self.ar_order, trim='both')
            _start_params['ar_coeff'] = np.linalg.pinv(X).dot(Y).tolist()
            resid = np.squeeze(Y - np.dot(X, _start_params['ar_coeff']))
            _start_params['ar_var'] = np.var(resid)

        # The variance of the residual term can be used for all variances,
        # just to get something in the right order of magnitude.
        var_resid = np.var(resid)

        # Seasonal
        if self.stochastic_seasonal:
            _start_params['seasonal_var'] = var_resid

        # Frequency domain seasonal
        for ix, is_stochastic in enumerate(self.stochastic_freq_seasonal):
            cov_key = 'freq_seasonal_var_{!r}'.format(ix)
            _start_params[cov_key] = var_resid

        # Cyclical
        if self.cycle:
            _start_params['cycle_var'] = var_resid
            # Clip this to make sure it is positive and strictly stationary
            # (i.e. do not want negative or 1)
            _start_params['cycle_damp'] = np.clip(
                np.linalg.pinv(resid[:-1, None]).dot(resid[1:])[0], 0, 0.99
            )

            # Set initial period estimate to 3 year, if we know the frequency
            # of the data observations
            freq = self.data.freq[0] if self.data.freq is not None else ''
            if freq == 'A':
                _start_params['cycle_freq'] = 2 * np.pi / 3
            elif freq == 'Q':
                _start_params['cycle_freq'] = 2 * np.pi / 12
            elif freq == 'M':
                _start_params['cycle_freq'] = 2 * np.pi / 36
            else:
                if not np.any(np.isinf(self.cycle_frequency_bound)):
                    _start_params['cycle_freq'] = (
                        np.mean(self.cycle_frequency_bound))
                elif np.isinf(self.cycle_frequency_bound[1]):
                    _start_params['cycle_freq'] = self.cycle_frequency_bound[0]
                else:
                    _start_params['cycle_freq'] = self.cycle_frequency_bound[1]

        # Irregular
        if self.irregular:
            _start_params['irregular_var'] = var_resid

        # Create the starting parameter list
        start_params = []
        for key in self.parameters.keys():
            if np.isscalar(_start_params[key]):
                start_params.append(_start_params[key])
            else:
                start_params += _start_params[key]
        return start_params

    @property
    def param_names(self):
        if not hasattr(self, 'parameters'):
            return []
        param_names = []
        for key in self.parameters.keys():
            if key == 'irregular_var':
                param_names.append('sigma2.irregular')
            elif key == 'level_var':
                param_names.append('sigma2.level')
            elif key == 'trend_var':
                param_names.append('sigma2.trend')
            elif key == 'seasonal_var':
                param_names.append('sigma2.seasonal')
            elif key.startswith('freq_seasonal_var_'):
                # There are potentially multiple frequency domain
                # seasonal terms
                idx_fseas_comp = int(key[-1])
                periodicity = self.freq_seasonal_periods[idx_fseas_comp]
                harmonics = self.freq_seasonal_harmonics[idx_fseas_comp]
                freq_seasonal_name = "{p}({h})".format(
                    p=repr(periodicity),
                    h=repr(harmonics))
                param_names.append(
                    'sigma2.' + 'freq_seasonal_' + freq_seasonal_name)
            elif key == 'cycle_var':
                param_names.append('sigma2.cycle')
            elif key == 'cycle_freq':
                param_names.append('frequency.cycle')
            elif key == 'cycle_damp':
                param_names.append('damping.cycle')
            elif key == 'ar_coeff':
                for i in range(self.ar_order):
                    param_names.append('ar.L%d' % (i+1))
            elif key == 'ar_var':
                param_names.append('sigma2.ar')
            elif key == 'reg_coeff':
                param_names += [
                    'beta.%s' % self.exog_names[i]
                    for i in range(self.k_exog)
                ]
            else:
                param_names.append(key)
        return param_names

    @property
    def state_names(self):
        names = []
        if self.level:
            names.append('level')
        if self.trend:
            names.append('trend')
        if self.seasonal:
            names.append('seasonal')
            names += ['seasonal.L%d' % i
                      for i in range(1, self._k_seasonal_states)]
        if self.freq_seasonal:
            names += ['freq_seasonal.%d' % i
                      for i in range(self._k_freq_seas_states)]
        if self.cycle:
            names += ['cycle', 'cycle.auxilliary']
        if self.ar_order > 0:
            names += ['ar.L%d' % i
                      for i in range(1, self.ar_order + 1)]
        if self.k_exog > 0 and not self.mle_regression:
            names += ['beta.%s' % self.exog_names[i]
                      for i in range(self.k_exog)]
        if self._unused_state:
            names += ['dummy']

        return names

    def transform_params(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation
        """
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = np.zeros(unconstrained.shape, dtype=unconstrained.dtype)

        # Positive parameters: obs_cov, state_cov
        offset = self.k_obs_cov + self.k_state_cov
        constrained[:offset] = unconstrained[:offset]**2

        # Cycle parameters
        if self.cycle:
            # Cycle frequency must be between between our bounds
            low, high = self.cycle_frequency_bound
            constrained[offset] = (
                1 / (1 + np.exp(-unconstrained[offset]))
            ) * (high - low) + low
            offset += 1

            # Cycle damping (if present) must be between 0 and 1
            if self.damped_cycle:
                constrained[offset] = (
                    1 / (1 + np.exp(-unconstrained[offset]))
                )
                offset += 1

        # Autoregressive coefficients must be stationary
        if self.autoregressive:
            constrained[offset:offset + self.ar_order] = (
                constrain_stationary_univariate(
                    unconstrained[offset:offset + self.ar_order]
                )
            )
            offset += self.ar_order

        # Nothing to do with betas
        constrained[offset:offset + self.k_exog] = (
            unconstrained[offset:offset + self.k_exog]
        )

        return constrained

    def untransform_params(self, constrained):
        """
        Reverse the transformation
        """
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.zeros(constrained.shape, dtype=constrained.dtype)

        # Positive parameters: obs_cov, state_cov
        offset = self.k_obs_cov + self.k_state_cov
        unconstrained[:offset] = constrained[:offset]**0.5

        # Cycle parameters
        if self.cycle:
            # Cycle frequency must be between between our bounds
            low, high = self.cycle_frequency_bound
            x = (constrained[offset] - low) / (high - low)
            unconstrained[offset] = np.log(
                x / (1 - x)
            )
            offset += 1

            # Cycle damping (if present) must be between 0 and 1
            if self.damped_cycle:
                unconstrained[offset] = np.log(
                    constrained[offset] / (1 - constrained[offset])
                )
                offset += 1

        # Autoregressive coefficients must be stationary
        if self.autoregressive:
            unconstrained[offset:offset + self.ar_order] = (
                unconstrain_stationary_univariate(
                    constrained[offset:offset + self.ar_order]
                )
            )
            offset += self.ar_order

        # Nothing to do with betas
        unconstrained[offset:offset + self.k_exog] = (
            constrained[offset:offset + self.k_exog]
        )

        return unconstrained

    def _validate_can_fix_params(self, param_names):
        super(UnobservedComponents, self)._validate_can_fix_params(param_names)

        if 'ar_coeff' in self.parameters:
            ar_names = ['ar.L%d' % (i+1) for i in range(self.ar_order)]
            fix_all_ar = param_names.issuperset(ar_names)
            fix_any_ar = len(param_names.intersection(ar_names)) > 0
            if fix_any_ar and not fix_all_ar:
                raise ValueError('Cannot fix individual autoregressive.'
                                 ' parameters. Must either fix all'
                                 ' autoregressive parameters or none.')

    def update(self, params, transformed=True, includes_fixed=False,
               complex_step=False):
        params = self.handle_params(params, transformed=transformed,
                                    includes_fixed=includes_fixed)

        offset = 0

        # Observation covariance
        if self.irregular:
            self.ssm['obs_cov', 0, 0] = params[offset]
            offset += 1

        # State covariance
        if self.k_state_cov > 0:
            variances = params[offset:offset+self.k_state_cov]
            if self._repeat_any_var:
                variances = np.repeat(variances, self._var_repetitions)
            self.ssm[self._idx_state_cov] = variances
            offset += self.k_state_cov

        # Cycle transition
        if self.cycle:
            cos_freq = np.cos(params[offset])
            sin_freq = np.sin(params[offset])
            cycle_transition = np.array(
                [[cos_freq, sin_freq],
                 [-sin_freq, cos_freq]]
            )
            if self.damped_cycle:
                offset += 1
                cycle_transition *= params[offset]
            self.ssm[self._idx_cycle_transition] = cycle_transition
            offset += 1

        # AR transition
        if self.autoregressive:
            self.ssm[self._idx_ar_transition] = (
                params[offset:offset+self.ar_order]
            )
            offset += self.ar_order

        # Beta observation intercept
        if self.regression:
            if self.mle_regression:
                self.ssm['obs_intercept'] = np.dot(
                    self.exog,
                    params[offset:offset+self.k_exog]
                )[None, :]
            offset += self.k_exog


class UnobservedComponentsResults(MLEResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None,
                 **kwargs):
        super(UnobservedComponentsResults, self).__init__(
            model, params, filter_results, cov_type, **kwargs)

        self.df_resid = np.inf  # attribute required for wald tests

        # Save _init_kwds
        self._init_kwds = self.model._get_init_kwds()

        # Save number of states by type
        self._k_states_by_type = {
            'seasonal': self.model._k_seasonal_states,
            'freq_seasonal': self.model._k_freq_seas_states,
            'cycle': self.model._k_cycle_states}

        # Save the model specification
        self.specification = Bunch(**{
            # Model options
            'level': self.model.level,
            'trend': self.model.trend,
            'seasonal_periods': self.model.seasonal_periods,
            'seasonal': self.model.seasonal,
            'freq_seasonal': self.model.freq_seasonal,
            'freq_seasonal_periods': self.model.freq_seasonal_periods,
            'freq_seasonal_harmonics': self.model.freq_seasonal_harmonics,
            'cycle': self.model.cycle,
            'ar_order': self.model.ar_order,
            'autoregressive': self.model.autoregressive,
            'irregular': self.model.irregular,
            'stochastic_level': self.model.stochastic_level,
            'stochastic_trend': self.model.stochastic_trend,
            'stochastic_seasonal': self.model.stochastic_seasonal,
            'stochastic_freq_seasonal': self.model.stochastic_freq_seasonal,
            'stochastic_cycle': self.model.stochastic_cycle,

            'damped_cycle': self.model.damped_cycle,
            'regression': self.model.regression,
            'mle_regression': self.model.mle_regression,
            'k_exog': self.model.k_exog,

            # Check for string trend/level specification
            'trend_specification': self.model.trend_specification
        })

    @property
    def level(self):
        """
        Estimates of unobserved level component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, level is always the first component of the state vector
        out = None
        spec = self.specification
        if spec.level:
            offset = 0
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def trend(self):
        """
        Estimates of of unobserved trend component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, trend is always the second component of the state vector
        # (because level is always present if trend is present)
        out = None
        spec = self.specification
        if spec.trend:
            offset = int(spec.level)
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def seasonal(self):
        """
        Estimates of unobserved seasonal component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, seasonal always follows level/trend (if they are present)
        # Note that we return only the first seasonal state, but there are
        # in fact seasonal_periods-1 seasonal states, however latter states
        # are just lagged versions of the first seasonal state.
        out = None
        spec = self.specification
        if spec.seasonal:
            offset = int(spec.trend + spec.level)
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def freq_seasonal(self):
        """
        Estimates of unobserved frequency domain seasonal component(s)

        Returns
        -------
        out: list of Bunch instances
            Each item has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, freq_seasonal components always follows level/trend
        #  and seasonal.

        # There are 2 * (harmonics) seasonal states per freq_seasonal
        # component.
        # The sum of every other state enters the measurement equation.
        # Additionally, there can be multiple components of this type.
        # These facts make this property messier in implementation than the
        # others.
        # Fortunately, the states are conditionally mutually independent
        # (conditional on previous timestep's states), so that the calculations
        # of the variances are simple summations of individual variances and
        # the calculation of the returned state is likewise a summation.
        out = []
        spec = self.specification
        if spec.freq_seasonal:
            previous_states_offset = int(spec.trend + spec.level
                                         + self._k_states_by_type['seasonal'])
            previous_f_seas_offset = 0
            for ix, h in enumerate(spec.freq_seasonal_harmonics):
                offset = previous_states_offset + previous_f_seas_offset

                period = spec.freq_seasonal_periods[ix]

                # Only the gamma_jt terms enter the measurement equation (cf.
                # D&K 2012 (3.7))
                states_in_sum = np.arange(0, 2 * h, 2)

                filtered_state = np.sum(
                    [self.filtered_state[offset + j] for j in states_in_sum],
                    axis=0)
                filtered_cov = np.sum(
                    [self.filtered_state_cov[offset + j, offset + j] for j in
                     states_in_sum], axis=0)

                item = Bunch(
                    filtered=filtered_state,
                    filtered_cov=filtered_cov,
                    smoothed=None, smoothed_cov=None,
                    offset=offset,
                    pretty_name='seasonal {p}({h})'.format(p=repr(period),
                                                           h=repr(h)))
                if self.smoothed_state is not None:
                    item.smoothed = np.sum(
                        [self.smoothed_state[offset+j] for j in states_in_sum],
                        axis=0)
                if self.smoothed_state_cov is not None:
                    item.smoothed_cov = np.sum(
                        [self.smoothed_state_cov[offset+j, offset+j]
                         for j in states_in_sum], axis=0)
                out.append(item)
                previous_f_seas_offset += 2 * h
        return out

    @property
    def cycle(self):
        """
        Estimates of unobserved cycle component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, cycle always follows level/trend, seasonal, and freq
        #  seasonal.
        # Note that we return only the first cyclical state, but there are
        # in fact 2 cyclical states. The second cyclical state is not simply
        # a lag of the first cyclical state, but the first cyclical state is
        # the one that enters the measurement equation.
        out = None
        spec = self.specification
        if spec.cycle:
            offset = int(spec.trend + spec.level
                         + self._k_states_by_type['seasonal']
                         + self._k_states_by_type['freq_seasonal'])
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def autoregressive(self):
        """
        Estimates of unobserved autoregressive component

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, autoregressive always follows level/trend, seasonal,
        # freq seasonal, and cyclical.
        # If it is an AR(p) model, then there are p associated
        # states, but the second - pth states are just lags of the first state.
        out = None
        spec = self.specification
        if spec.autoregressive:
            offset = int(spec.trend + spec.level
                         + self._k_states_by_type['seasonal']
                         + self._k_states_by_type['freq_seasonal']
                         + self._k_states_by_type['cycle'])
            out = Bunch(filtered=self.filtered_state[offset],
                        filtered_cov=self.filtered_state_cov[offset, offset],
                        smoothed=None, smoothed_cov=None,
                        offset=offset)
            if self.smoothed_state is not None:
                out.smoothed = self.smoothed_state[offset]
            if self.smoothed_state_cov is not None:
                out.smoothed_cov = self.smoothed_state_cov[offset, offset]
        return out

    @property
    def regression_coefficients(self):
        """
        Estimates of unobserved regression coefficients

        Returns
        -------
        out: Bunch
            Has the following attributes:

            - `filtered`: a time series array with the filtered estimate of
                          the component
            - `filtered_cov`: a time series array with the filtered estimate of
                          the variance/covariance of the component
            - `smoothed`: a time series array with the smoothed estimate of
                          the component
            - `smoothed_cov`: a time series array with the smoothed estimate of
                          the variance/covariance of the component
            - `offset`: an integer giving the offset in the state vector where
                        this component begins
        """
        # If present, state-vector regression coefficients always are last
        # (i.e. they follow level/trend, seasonal, freq seasonal, cyclical, and
        # autoregressive states). There is one state associated with each
        # regressor, and all are returned here.
        out = None
        spec = self.specification
        if spec.regression:
            if spec.mle_regression:
                import warnings
                warnings.warn('Regression coefficients estimated via maximum'
                              ' likelihood. Estimated coefficients are'
                              ' available in the parameters list, not as part'
                              ' of the state vector.', OutputWarning)
            else:
                offset = int(spec.trend + spec.level
                             + self._k_states_by_type['seasonal']
                             + self._k_states_by_type['freq_seasonal']
                             + self._k_states_by_type['cycle']
                             + spec.ar_order)
                start = offset
                end = offset + spec.k_exog
                out = Bunch(
                    filtered=self.filtered_state[start:end],
                    filtered_cov=self.filtered_state_cov[start:end, start:end],
                    smoothed=None, smoothed_cov=None,
                    offset=offset
                )
                if self.smoothed_state is not None:
                    out.smoothed = self.smoothed_state[start:end]
                if self.smoothed_state_cov is not None:
                    out.smoothed_cov = (
                        self.smoothed_state_cov[start:end, start:end])
        return out

    def plot_components(self, which=None, alpha=0.05,
                        observed=True, level=True, trend=True,
                        seasonal=True, freq_seasonal=True,
                        cycle=True, autoregressive=True,
                        legend_loc='upper right', fig=None, figsize=None):
        """
        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered', 'smoothed'}, or None, optional
            Type of state estimate to plot. Default is 'smoothed' if smoothed
            results are available otherwise 'filtered'.
        alpha : float, optional
            The confidence intervals for the components are (1 - alpha) %
        observed : bool, optional
            Whether or not to plot the observed series against
            one-step-ahead predictions.
            Default is True.
        level : bool, optional
            Whether or not to plot the level component, if applicable.
            Default is True.
        trend : bool, optional
            Whether or not to plot the trend component, if applicable.
            Default is True.
        seasonal : bool, optional
            Whether or not to plot the seasonal component, if applicable.
            Default is True.
        freq_seasonal : bool, optional
            Whether or not to plot the frequency domain seasonal component(s),
            if applicable. Default is True.
        cycle : bool, optional
            Whether or not to plot the cyclical component, if applicable.
            Default is True.
        autoregressive : bool, optional
            Whether or not to plot the autoregressive state, if applicable.
            Default is True.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        If all options are included in the model and selected, this produces
        a 6x1 plot grid with the following plots (ordered top-to-bottom):

        0. Observed series against predicted series
        1. Level
        2. Trend
        3. Seasonal
        4. Freq Seasonal
        5. Cycle
        6. Autoregressive

        Specific subplots will be removed if the component is not present in
        the estimated model or if the corresponding keyword argument is set to
        False.

        All plots contain (1 - `alpha`) %  confidence intervals.
        """
        from scipy.stats import norm
        from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
        plt = _import_mpl()
        fig = create_mpl_fig(fig, figsize)

        # Determine which results we have
        if which is None:
            which = 'filtered' if self.smoothed_state is None else 'smoothed'

        # Determine which plots we have
        spec = self.specification

        comp = [
            ('level', level and spec.level),
            ('trend', trend and spec.trend),
            ('seasonal', seasonal and spec.seasonal),
        ]

        if freq_seasonal and spec.freq_seasonal:
            for ix, _ in enumerate(spec.freq_seasonal_periods):
                key = 'freq_seasonal_{!r}'.format(ix)
                comp.append((key, True))

        comp.extend(
            [('cycle', cycle and spec.cycle),
             ('autoregressive', autoregressive and spec.autoregressive)])

        components = dict(comp)

        llb = self.filter_results.loglikelihood_burn

        # Number of plots
        k_plots = observed + np.sum(list(components.values()))

        # Get dates, if applicable
        if hasattr(self.data, 'dates') and self.data.dates is not None:
            dates = self.data.dates._mpl_repr()
        else:
            dates = np.arange(len(self.data.endog))

        # Get the critical value for confidence intervals
        critical_value = norm.ppf(1 - alpha / 2.)

        plot_idx = 1

        # Observed, predicted, confidence intervals
        if observed:
            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            # Plot the observed dataset
            ax.plot(dates[llb:], self.model.endog[llb:], color='k',
                    label='Observed')

            # Get the predicted values and confidence intervals
            predict = self.filter_results.forecasts[0]
            std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
            ci_lower = predict - critical_value * std_errors
            ci_upper = predict + critical_value * std_errors

            # Plot
            ax.plot(dates[llb:], predict[llb:],
                    label='One-step-ahead predictions')
            ci_poly = ax.fill_between(
                dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2
            )
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)

            # Proxy artist for fill_between legend entry
            # See e.g. https://matplotlib.org/1.3.1/users/legend_guide.html
            p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            handles.append(p)
            labels.append(ci_label)
            ax.legend(handles, labels, loc=legend_loc)

            ax.set_title('Predicted vs observed')

        # Plot each component
        for component, is_plotted in components.items():
            if not is_plotted:
                continue

            ax = fig.add_subplot(k_plots, 1, plot_idx)
            plot_idx += 1

            try:
                component_bunch = getattr(self, component)
                title = component.title()
            except AttributeError:
                # This might be a freq_seasonal component, of which there are
                #  possibly multiple bagged up in property freq_seasonal
                if component.startswith('freq_seasonal_'):
                    ix = int(component.replace('freq_seasonal_', ''))
                    big_bunch = getattr(self, 'freq_seasonal')
                    component_bunch = big_bunch[ix]
                    title = component_bunch.pretty_name
                else:
                    raise

            # Check for a valid estimation type
            if which not in component_bunch:
                raise ValueError('Invalid type of state estimate.')

            which_cov = '%s_cov' % which

            # Get the predicted values
            value = component_bunch[which]

            # Plot
            state_label = '%s (%s)' % (title, which)
            ax.plot(dates[llb:], value[llb:], label=state_label)

            # Get confidence intervals
            if which_cov in component_bunch:
                std_errors = np.sqrt(component_bunch['%s_cov' % which])
                ci_lower = value - critical_value * std_errors
                ci_upper = value + critical_value * std_errors
                ci_poly = ax.fill_between(
                    dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2
                )
                ci_label = ('$%.3g \\%%$ confidence interval'
                            % ((1 - alpha) * 100))

            # Legend
            ax.legend(loc=legend_loc)

            ax.set_title('%s component' % title)

        # Add a note if first observations excluded
        if llb > 0:
            text = ('Note: The first %d observations are not shown, due to'
                    ' approximate diffuse initialization.')
            fig.text(0.1, 0.01, text % llb, fontsize='large')

        return fig

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=.05, start=None):
        # Create the model name

        model_name = [self.specification.trend_specification]

        if self.specification.seasonal:
            seasonal_name = ('seasonal(%d)'
                             % self.specification.seasonal_periods)
            if self.specification.stochastic_seasonal:
                seasonal_name = 'stochastic ' + seasonal_name
            model_name.append(seasonal_name)

        if self.specification.freq_seasonal:
            for ix, is_stochastic in enumerate(
                    self.specification.stochastic_freq_seasonal):
                periodicity = self.specification.freq_seasonal_periods[ix]
                harmonics = self.specification.freq_seasonal_harmonics[ix]
                freq_seasonal_name = "freq_seasonal({p}({h}))".format(
                    p=repr(periodicity),
                    h=repr(harmonics))
                if is_stochastic:
                    freq_seasonal_name = 'stochastic ' + freq_seasonal_name
                model_name.append(freq_seasonal_name)

        if self.specification.cycle:
            cycle_name = 'cycle'
            if self.specification.stochastic_cycle:
                cycle_name = 'stochastic ' + cycle_name
            if self.specification.damped_cycle:
                cycle_name = 'damped ' + cycle_name
            model_name.append(cycle_name)

        if self.specification.autoregressive:
            autoregressive_name = 'AR(%d)' % self.specification.ar_order
            model_name.append(autoregressive_name)

        return super(UnobservedComponentsResults, self).summary(
            alpha=alpha, start=start, title='Unobserved Components Results',
            model_name=model_name
        )


class UnobservedComponentsResultsWrapper(MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(UnobservedComponentsResultsWrapper,  # noqa:E305
                      UnobservedComponentsResults)
