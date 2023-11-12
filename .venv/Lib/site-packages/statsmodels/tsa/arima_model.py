"""
See statsmodels.tsa.arima.model.ARIMA and statsmodels.tsa.SARIMAX.
"""

ARIMA_DEPRECATION_ERROR = """
statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
been removed in favor of statsmodels.tsa.arima.model.ARIMA (note the .
between arima and model) and statsmodels.tsa.SARIMAX.

statsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and
is both well tested and maintained. It also offers alternative specialized
parameter estimators.
"""


class ARMA:
    """
    ARMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(ARIMA_DEPRECATION_ERROR)


class ARIMA(ARMA):
    """
    ARIMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ARMAResults:
    """
    ARMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(ARIMA_DEPRECATION_ERROR)


class ARIMAResults(ARMAResults):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
