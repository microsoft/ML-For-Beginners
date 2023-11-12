"""
Durbin-Levinson recursions for estimating AR(p) model parameters.

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import deprecate_kwarg

import numpy as np

from statsmodels.tools.tools import Bunch
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.stattools import acovf


@deprecate_kwarg("unbiased", "adjusted")
def durbin_levinson(endog, ar_order=0, demean=True, adjusted=False):
    """
    Estimate AR parameters at multiple orders using Durbin-Levinson recursions.

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
        Whether to use the "adjusted" autocovariance estimator, which uses
        n - h degrees of freedom rather than n. This option can result in
        a non-positive definite autocovariance matrix. Default is False.

    Returns
    -------
    parameters : list of SARIMAXParams objects
        List elements correspond to estimates at different `ar_order`. For
        example, parameters[0] is an `SARIMAXParams` instance corresponding to
        `ar_order=0`.
    other_results : Bunch
        Includes one component, `spec`, containing the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 2.5.1.

    This procedure assumes that the series is stationary.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    """
    spec = max_spec = SARIMAXSpecification(endog, ar_order=ar_order)
    endog = max_spec.endog

    # Make sure we have a consecutive process
    if not max_spec.is_ar_consecutive:
        raise ValueError('Durbin-Levinson estimation unavailable for models'
                         ' with seasonal or otherwise non-consecutive AR'
                         ' orders.')

    gamma = acovf(endog, adjusted=adjusted, fft=True, demean=demean,
                  nlag=max_spec.ar_order)

    # If no AR component, just a variance computation
    if max_spec.ar_order == 0:
        ar_params = [None]
        sigma2 = [gamma[0]]
    # Otherwise, AR model
    else:
        Phi = np.zeros((max_spec.ar_order, max_spec.ar_order))
        v = np.zeros(max_spec.ar_order + 1)

        Phi[0, 0] = gamma[1] / gamma[0]
        v[0] = gamma[0]
        v[1] = v[0] * (1 - Phi[0, 0]**2)

        for i in range(1, max_spec.ar_order):
            tmp = Phi[i-1, :i]
            Phi[i, i] = (gamma[i + 1] - np.dot(tmp, gamma[i:0:-1])) / v[i]
            Phi[i, :i] = (tmp - Phi[i, i] * tmp[::-1])
            v[i + 1] = v[i] * (1 - Phi[i, i]**2)

        ar_params = [None] + [Phi[i, :i + 1] for i in range(max_spec.ar_order)]
        sigma2 = v

    # Compute output
    out = []
    for i in range(max_spec.ar_order + 1):
        spec = SARIMAXSpecification(ar_order=i)
        p = SARIMAXParams(spec=spec)
        if i == 0:
            p.params = sigma2[i]
        else:
            p.params = np.r_[ar_params[i], sigma2[i]]
        out.append(p)

        # Construct other results
    other_results = Bunch({
        'spec': spec,
    })

    return out, other_results
