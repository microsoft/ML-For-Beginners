"""
Hannan-Rissanen procedure for estimating ARMA(p,q) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np

from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams


def hannan_rissanen(endog, ar_order=0, ma_order=0, demean=True,
                    initial_ar_order=None, unbiased=None,
                    fixed_params=None):
    """
    Estimate ARMA parameters using Hannan-Rissanen procedure.

    Parameters
    ----------
    endog : array_like
        Input time series array, assumed to be stationary.
    ar_order : int or list of int
        Autoregressive order
    ma_order : int or list of int
        Moving average order
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the ARMA coefficients. Default is True.
    initial_ar_order : int, optional
        Order of long autoregressive process used for initial computation of
        residuals.
    unbiased : bool, optional
        Whether or not to apply the bias correction step. Default is True if
        the estimated coefficients from the previous step imply a stationary
        and invertible process and False otherwise.
    fixed_params : dict, optional
        Dictionary with names of fixed parameters as keys (e.g. 'ar.L1',
        'ma.L2'), which correspond to SARIMAXSpecification.param_names.
        Dictionary values are the values of the associated fixed parameters.

    Returns
    -------
    parameters : SARIMAXParams object
    other_results : Bunch
        Includes three components: `spec`, containing the
        `SARIMAXSpecification` instance corresponding to the input arguments;
        `initial_ar_order`, containing the autoregressive lag order used in the
        first step; and `resid`, which contains the computed residuals from the
        last step.

    Notes
    -----
    The primary reference is [1]_, section 5.1.4, which describes a three-step
    procedure that we implement here.

    1. Fit a large-order AR model via Yule-Walker to estimate residuals
    2. Compute AR and MA estimates via least squares
    3. (Unless the estimated coefficients from step (2) are non-stationary /
       non-invertible or `unbiased=False`) Perform bias correction

    The order used for the AR model in the first step may be given as an
    argument. If it is not, we compute it as suggested by [2]_.

    The estimate of the variance that we use is computed from the residuals
    of the least-squares regression and not from the innovations algorithm.
    This is because our fast implementation of the innovations algorithm is
    only valid for stationary processes, and the Hannan-Rissanen procedure may
    produce estimates that imply non-stationary processes. To avoid
    inconsistency, we never compute this latter variance here, even if it is
    possible. See test_hannan_rissanen::test_brockwell_davis_example_517 for
    an example of how to compute this variance manually.

    This procedure assumes that the series is stationary, but if this is not
    true, it is still possible that this procedure will return parameters that
    imply a non-stationary / non-invertible process.

    Note that the third stage will only be applied if the parameters from the
    second stage imply a stationary / invertible model. If `unbiased=True` is
    given, then non-stationary / non-invertible parameters in the second stage
    will throw an exception.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    .. [2] Gomez, Victor, and Agustin Maravall. 2001.
       "Automatic Modeling Methods for Univariate Series."
       A Course in Time Series Analysis, 171–201.
    """
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)

    fixed_params = _validate_fixed_params(fixed_params, spec.param_names)

    endog = spec.endog
    if demean:
        endog = endog - endog.mean()

    p = SARIMAXParams(spec=spec)

    nobs = len(endog)
    max_ar_order = spec.max_ar_order
    max_ma_order = spec.max_ma_order

    # Default initial_ar_order is as suggested by Gomez and Maravall (2001)
    if initial_ar_order is None:
        initial_ar_order = max(np.floor(np.log(nobs)**2).astype(int),
                               2 * max(max_ar_order, max_ma_order))
    # Create a spec, just to validate the initial autoregressive order
    _ = SARIMAXSpecification(endog, ar_order=initial_ar_order)

    # Unpack fixed and free ar/ma lags, ix, and params (fixed only)
    params_info = _package_fixed_and_free_params_info(
        fixed_params, spec.ar_lags, spec.ma_lags
    )

    # Compute lagged endog
    lagged_endog = lagmat(endog, max_ar_order, trim='both')

    # If no AR or MA components, this is just a variance computation
    mod = None
    if max_ma_order == 0 and max_ar_order == 0:
        p.sigma2 = np.var(endog, ddof=0)
        resid = endog.copy()
    # If no MA component, this is just CSS
    elif max_ma_order == 0:
        # extract 1) lagged_endog with free params; 2) lagged_endog with fixed
        # params; 3) endog residual after applying fixed params if applicable
        X_with_free_params = lagged_endog[:, params_info.free_ar_ix]
        X_with_fixed_params = lagged_endog[:, params_info.fixed_ar_ix]
        y = endog[max_ar_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(params_info.fixed_ar_params)

        # no free ar params -> variance computation on the endog residual
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        # otherwise OLS with endog residual (after applying fixed params) as y,
        # and lagged_endog with free params as X
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            resid = res.resid
            p.sigma2 = res.scale
            p.ar_params = _stitch_fixed_and_free_params(
                fixed_ar_or_ma_lags=params_info.fixed_ar_lags,
                fixed_ar_or_ma_params=params_info.fixed_ar_params,
                free_ar_or_ma_lags=params_info.free_ar_lags,
                free_ar_or_ma_params=res.params,
                spec_ar_or_ma_lags=spec.ar_lags
            )
    # Otherwise ARMA model
    else:
        # Step 1: Compute long AR model via Yule-Walker, get residuals
        initial_ar_params, _ = yule_walker(
            endog, order=initial_ar_order, method='mle')
        X = lagmat(endog, initial_ar_order, trim='both')
        y = endog[initial_ar_order:]
        resid = y - X.dot(initial_ar_params)

        # Get lagged residuals for `exog` in least-squares regression
        lagged_resid = lagmat(resid, max_ma_order, trim='both')

        # Step 2: estimate ARMA model via least squares
        ix = initial_ar_order + max_ma_order - max_ar_order
        X_with_free_params = np.c_[
            lagged_endog[ix:, params_info.free_ar_ix],
            lagged_resid[:, params_info.free_ma_ix]
        ]
        X_with_fixed_params = np.c_[
            lagged_endog[ix:, params_info.fixed_ar_ix],
            lagged_resid[:, params_info.fixed_ma_ix]
        ]
        y = endog[initial_ar_order + max_ma_order:]
        if X_with_fixed_params.shape[1] != 0:
            y = y - X_with_fixed_params.dot(
                np.r_[params_info.fixed_ar_params, params_info.fixed_ma_params]
            )

        # Step 2.1: no free ar params -> variance computation on the endog
        # residual
        if X_with_free_params.shape[1] == 0:
            p.ar_params = params_info.fixed_ar_params
            p.ma_params = params_info.fixed_ma_params
            p.sigma2 = np.var(y, ddof=0)
            resid = y.copy()
        # Step 2.2: otherwise OLS with endog residual (after applying fixed
        # params) as y, and lagged_endog and lagged_resid with free params as X
        else:
            mod = OLS(y, X_with_free_params)
            res = mod.fit()
            k_free_ar_params = len(params_info.free_ar_lags)
            p.ar_params = _stitch_fixed_and_free_params(
                fixed_ar_or_ma_lags=params_info.fixed_ar_lags,
                fixed_ar_or_ma_params=params_info.fixed_ar_params,
                free_ar_or_ma_lags=params_info.free_ar_lags,
                free_ar_or_ma_params=res.params[:k_free_ar_params],
                spec_ar_or_ma_lags=spec.ar_lags
            )
            p.ma_params = _stitch_fixed_and_free_params(
                fixed_ar_or_ma_lags=params_info.fixed_ma_lags,
                fixed_ar_or_ma_params=params_info.fixed_ma_params,
                free_ar_or_ma_lags=params_info.free_ma_lags,
                free_ar_or_ma_params=res.params[k_free_ar_params:],
                spec_ar_or_ma_lags=spec.ma_lags
            )
            resid = res.resid
            p.sigma2 = res.scale

        # Step 3: bias correction (if requested)

        # Step 3.1: validate `unbiased` argument and handle setting the default
        if unbiased is True:
            if len(fixed_params) != 0:
                raise NotImplementedError(
                    "Third step of Hannan-Rissanen estimation to remove "
                    "parameter bias is not yet implemented for the case "
                    "with fixed parameters."
                )
            elif not (p.is_stationary and p.is_invertible):
                raise ValueError(
                    "Cannot perform third step of Hannan-Rissanen estimation "
                    "to remove parameter bias, because parameters estimated "
                    "from the second step are non-stationary or "
                    "non-invertible."
                )
        elif unbiased is None:
            if len(fixed_params) != 0:
                unbiased = False
            else:
                unbiased = p.is_stationary and p.is_invertible

        # Step 3.2: bias correction
        if unbiased is True:
            if mod is None:
                raise ValueError("Must have free parameters to use unbiased")
            Z = np.zeros_like(endog)

            ar_coef = p.ar_poly.coef
            ma_coef = p.ma_poly.coef

            for t in range(nobs):
                if t >= max(max_ar_order, max_ma_order):
                    # Note: in the case of non-consecutive lag orders, the
                    # polynomials have the appropriate zeros so we don't
                    # need to subset `endog[t - max_ar_order:t]` or
                    # Z[t - max_ma_order:t]
                    tmp_ar = np.dot(
                        -ar_coef[1:], endog[t - max_ar_order:t][::-1])
                    tmp_ma = np.dot(ma_coef[1:],
                                    Z[t - max_ma_order:t][::-1])
                    Z[t] = endog[t] - tmp_ar - tmp_ma

            V = lfilter([1], ar_coef, Z)
            W = lfilter(np.r_[1, -ma_coef[1:]], [1], Z)

            lagged_V = lagmat(V, max_ar_order, trim='both')
            lagged_W = lagmat(W, max_ma_order, trim='both')

            exog = np.c_[
                lagged_V[
                    max(max_ma_order - max_ar_order, 0):,
                    params_info.free_ar_ix
                ],
                lagged_W[
                    max(max_ar_order - max_ma_order, 0):,
                    params_info.free_ma_ix
                ]
            ]

            mod_unbias = OLS(Z[max(max_ar_order, max_ma_order):], exog)
            res_unbias = mod_unbias.fit()

            p.ar_params = (
                p.ar_params + res_unbias.params[:spec.k_ar_params])
            p.ma_params = (
                p.ma_params + res_unbias.params[spec.k_ar_params:])

            # Recompute sigma2
            resid = mod.endog - mod.exog.dot(
                np.r_[p.ar_params, p.ma_params])
            p.sigma2 = np.inner(resid, resid) / len(resid)

    # TODO: Gomez and Maravall (2001) or Gomez (1998)
    # propose one more step here to further improve MA estimates

    # Construct results
    other_results = Bunch({
        'spec': spec,
        'initial_ar_order': initial_ar_order,
        'resid': resid
    })
    return p, other_results


def _validate_fixed_params(fixed_params, spec_param_names):
    """
    Check that keys in fixed_params are a subset of spec.param_names except
    "sigma2"

    Parameters
    ----------
    fixed_params : dict
    spec_param_names : list of string
        SARIMAXSpecification.param_names
    """
    if fixed_params is None:
        fixed_params = {}

    assert isinstance(fixed_params, dict)

    fixed_param_names = set(fixed_params.keys())
    valid_param_names = set(spec_param_names) - {"sigma2"}

    invalid_param_names = fixed_param_names - valid_param_names

    if len(invalid_param_names) > 0:
        raise ValueError(
            f"Invalid fixed parameter(s): {sorted(list(invalid_param_names))}."
            f" Please select among {sorted(list(valid_param_names))}."
        )

    return fixed_params


def _package_fixed_and_free_params_info(fixed_params, spec_ar_lags,
                                        spec_ma_lags):
    """
    Parameters
    ----------
    fixed_params : dict
    spec_ar_lags : list of int
        SARIMAXSpecification.ar_lags
    spec_ma_lags : list of int
        SARIMAXSpecification.ma_lags

    Returns
    -------
    Bunch with
    (lags) fixed_ar_lags, fixed_ma_lags, free_ar_lags, free_ma_lags;
    (ix) fixed_ar_ix, fixed_ma_ix, free_ar_ix, free_ma_ix;
    (params) fixed_ar_params, free_ma_params
    """
    # unpack fixed lags and params
    fixed_ar_lags_and_params = []
    fixed_ma_lags_and_params = []
    for key, val in fixed_params.items():
        lag = int(key.split(".")[-1].lstrip("L"))
        if key.startswith("ar"):
            fixed_ar_lags_and_params.append((lag, val))
        elif key.startswith("ma"):
            fixed_ma_lags_and_params.append((lag, val))

    fixed_ar_lags_and_params.sort()
    fixed_ma_lags_and_params.sort()

    fixed_ar_lags = [lag for lag, _ in fixed_ar_lags_and_params]
    fixed_ar_params = np.array([val for _, val in fixed_ar_lags_and_params])

    fixed_ma_lags = [lag for lag, _ in fixed_ma_lags_and_params]
    fixed_ma_params = np.array([val for _, val in fixed_ma_lags_and_params])

    # unpack free lags
    free_ar_lags = [lag for lag in spec_ar_lags
                    if lag not in set(fixed_ar_lags)]
    free_ma_lags = [lag for lag in spec_ma_lags
                    if lag not in set(fixed_ma_lags)]

    # get ix for indexing purposes: `ar_ix`, and `ma_ix` below, are to account
    # for non-consecutive lags; for indexing purposes, must have dtype int
    free_ar_ix = np.array(free_ar_lags, dtype=int) - 1
    free_ma_ix = np.array(free_ma_lags, dtype=int) - 1
    fixed_ar_ix = np.array(fixed_ar_lags, dtype=int) - 1
    fixed_ma_ix = np.array(fixed_ma_lags, dtype=int) - 1

    return Bunch(
        # lags
        fixed_ar_lags=fixed_ar_lags, fixed_ma_lags=fixed_ma_lags,
        free_ar_lags=free_ar_lags, free_ma_lags=free_ma_lags,
        # ixs
        fixed_ar_ix=fixed_ar_ix, fixed_ma_ix=fixed_ma_ix,
        free_ar_ix=free_ar_ix, free_ma_ix=free_ma_ix,
        # fixed params
        fixed_ar_params=fixed_ar_params, fixed_ma_params=fixed_ma_params,
    )


def _stitch_fixed_and_free_params(fixed_ar_or_ma_lags, fixed_ar_or_ma_params,
                                  free_ar_or_ma_lags, free_ar_or_ma_params,
                                  spec_ar_or_ma_lags):
    """
    Stitch together fixed and free params, by the order of lags, for setting
    SARIMAXParams.ma_params or SARIMAXParams.ar_params

    Parameters
    ----------
    fixed_ar_or_ma_lags : list or np.array
    fixed_ar_or_ma_params : list or np.array
        fixed_ar_or_ma_params corresponds with fixed_ar_or_ma_lags
    free_ar_or_ma_lags : list or np.array
    free_ar_or_ma_params : list or np.array
        free_ar_or_ma_params corresponds with free_ar_or_ma_lags
    spec_ar_or_ma_lags : list
        SARIMAXSpecification.ar_lags or SARIMAXSpecification.ma_lags

    Returns
    -------
    list of fixed and free params by the order of lags
    """
    assert len(fixed_ar_or_ma_lags) == len(fixed_ar_or_ma_params)
    assert len(free_ar_or_ma_lags) == len(free_ar_or_ma_params)

    all_lags = np.r_[fixed_ar_or_ma_lags, free_ar_or_ma_lags]
    all_params = np.r_[fixed_ar_or_ma_params, free_ar_or_ma_params]
    assert set(all_lags) == set(spec_ar_or_ma_lags)

    lag_to_param_map = dict(zip(all_lags, all_params))

    # Sort params by the order of their corresponding lags in
    # spec_ar_or_ma_lags (e.g. SARIMAXSpecification.ar_lags or
    # SARIMAXSpecification.ma_lags)
    all_params_sorted = [lag_to_param_map[lag] for lag in spec_ar_or_ma_lags]
    return all_params_sorted
