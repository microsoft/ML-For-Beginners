"""
Burg's method for estimating AR(p) model parameters.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np

from statsmodels.tools.tools import Bunch
from statsmodels.regression import linear_model

from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams


def burg(endog, ar_order=0, demean=True):
    """
    Estimate AR parameters using Burg technique.

    Parameters
    ----------
    endog : array_like or SARIMAXSpecification
        Input time series array, assumed to be stationary.
    ar_order : int, optional
        Autoregressive order. Default is 0.
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the autoregressive coefficients.

    Returns
    -------
    parameters : SARIMAXParams object
        Contains the parameter estimates from the final iteration.
    other_results : Bunch
        Includes one component, `spec`, which is the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 5.1.2.

    This procedure assumes that the series is stationary.

    This function is a light wrapper around `statsmodels.linear_model.burg`.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    """
    spec = SARIMAXSpecification(endog, ar_order=ar_order)
    endog = spec.endog

    # Workaround for statsmodels.tsa.stattools.pacf_burg which does not work
    # on integer input
    # TODO: remove when possible
    if np.issubdtype(endog.dtype, np.dtype(int)):
        endog = endog * 1.0

    if not spec.is_ar_consecutive:
        raise ValueError('Burg estimation unavailable for models with'
                         ' seasonal or otherwise non-consecutive AR orders.')

    p = SARIMAXParams(spec=spec)

    if ar_order == 0:
        p.sigma2 = np.var(endog)
    else:
        p.ar_params, p.sigma2 = linear_model.burg(endog, order=ar_order,
                                                  demean=demean)

        # Construct other results
    other_results = Bunch({
        'spec': spec,
    })

    return p, other_results
