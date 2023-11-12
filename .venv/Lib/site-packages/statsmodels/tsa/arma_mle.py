"""
Created on Sun Oct 10 14:57:50 2010

Author: josef-pktd, Skipper Seabold
License: BSD

TODO: check everywhere initialization of signal.lfilter

"""


class Arma:
    """
    Removed in 0.14. Use SARIMAX, ARIMA or AutoReg.

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
    statsmodels.tsa.arima.model.ARIMA
    statsmodels.tsa.ar_model.AutoReg
    """

    def __init__(self, endog, exog=None):
        raise NotImplementedError(
            "ARMA has been removed. Use SARIMAX, ARIMA or AutoReg"
        )
