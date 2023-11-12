"""
Yule-Walker method for estimating AR(p) model parameters.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import deprecate_kwarg

from statsmodels.regression import linear_model
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification


@deprecate_kwarg("unbiased", "adjusted")
def yule_walker(endog, ar_order=0, demean=True, adjusted=False):
    """
    Estimate AR parameters using Yule-Walker equations.

    Parameters
    ----------
    endog : array_like or SARIMAXSpecification
        Input time series array, assumed to be stationary.
    ar_order : int, optional
        Autoregressive order. Default is 0.
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the autoregressive coefficients. Default is True.
    adjusted : bool, optional
        Whether to use the adjusted autocovariance estimator, which uses
        n - h degrees of freedom rather than n. For some processes this option
        may  result in a non-positive definite autocovariance matrix. Default
        is False.

    Returns
    -------
    parameters : SARIMAXParams object
        Contains the parameter estimates from the final iteration.
    other_results : Bunch
        Includes one component, `spec`, which is the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 5.1.1.

    This procedure assumes that the series is stationary.

    For a description of the effect of the adjusted estimate of the
    autocovariance function, see 2.4.2 of [1]_.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    """
    spec = SARIMAXSpecification(endog, ar_order=ar_order)
    endog = spec.endog
    p = SARIMAXParams(spec=spec)

    if not spec.is_ar_consecutive:
        raise ValueError('Yule-Walker estimation unavailable for models with'
                         ' seasonal or non-consecutive AR orders.')

    # Estimate parameters
    method = 'adjusted' if adjusted else 'mle'
    p.ar_params, sigma = linear_model.yule_walker(
        endog, order=ar_order, demean=demean, method=method)
    p.sigma2 = sigma**2

    # Construct other results
    other_results = Bunch({
        'spec': spec,
    })

    return p, other_results
