import numpy as np
from statsmodels.tools.tools import Bunch


class _MinimalWLS:
    """
    Minimal implementation of WLS optimized for performance.

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable. The dependent variable.
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`.
    weights : array_like, optional
        1d array of weights.  If you supply 1/W then the variables are pre-
        multiplied by 1/sqrt(W).  If no weights are supplied the default value
        is 1 and WLS reults are the same as OLS.
    check_endog : bool, optional
        Flag indicating whether to check for inf/nan in endog.
        If True and any are found, ValueError is raised.
    check_weights : bool, optional
        Flag indicating whether to check for inf/nan in weights.
        If True and any are found, ValueError is raised.

    Notes
    -----
    Provides only resid, scale, fittedvalues, model.weights which are used by
    methods that iteratively apply WLS.

    Does not perform any checks on the input data for type or shape
    compatibility
    """

    msg = 'NaN, inf or invalid value detected in {0}, estimation infeasible.'

    def __init__(self, endog, exog, weights=1.0, check_endog=False,
                 check_weights=False):
        self.endog = endog
        self.exog = exog
        self.weights = weights
        w_half = np.sqrt(weights)
        if check_weights:
            if not np.all(np.isfinite(w_half)):
                raise ValueError(self.msg.format('weights'))

        if check_endog:
            if not np.all(np.isfinite(endog)):
                raise ValueError(self.msg.format('endog'))

        self.wendog = w_half * endog
        if np.isscalar(weights):
            self.wexog = w_half * exog
        else:
            self.wexog = np.asarray(w_half)[:, None] * exog

    def fit(self, method='pinv'):
        """
        Minimal implementation of WLS optimized for performance.

        Parameters
        ----------
        method : str, optional
            Method to use to estimate parameters.  "pinv", "qr" or "lstsq"

              * "pinv" uses the Moore-Penrose pseudoinverse
                 to solve the least squares problem.
              * "qr" uses the QR factorization.
              * "lstsq" uses the least squares implementation in numpy.linalg

        Returns
        -------
        results : namedtuple
            Named tuple containing the fewest terms needed to implement
            iterative estimation in models. Currently

              * params : Estimated parameters
              * fittedvalues : Fit values using original data
              * resid : Residuals using original data
              * model : namedtuple with one field, weights
              * scale : scale computed using weighted residuals

        Notes
        -----
        Does not perform and checks on the input data

        See Also
        --------
        statsmodels.regression.linear_model.WLS
        """
        if method == 'pinv':
            pinv_wexog = np.linalg.pinv(self.wexog)
            params = pinv_wexog.dot(self.wendog)
        elif method == 'qr':
            Q, R = np.linalg.qr(self.wexog)
            params = np.linalg.solve(R, np.dot(Q.T, self.wendog))
        else:
            params, _, _, _ = np.linalg.lstsq(self.wexog, self.wendog,
                                              rcond=-1)
        return self.results(params)

    def results(self, params):
        """
        Construct results

        params : ndarray
            Model parameters

        Notes
        -----
        Allows results to be constructed from either existing parameters or
        when estimated using using ``fit``
        """
        fitted_values = self.exog.dot(params)
        resid = self.endog - fitted_values
        wresid = self.wendog - self.wexog.dot(params)
        df_resid = self.wexog.shape[0] - self.wexog.shape[1]
        scale = np.dot(wresid, wresid) / df_resid

        return Bunch(params=params, fittedvalues=fitted_values, resid=resid,
                     model=self, scale=scale)
