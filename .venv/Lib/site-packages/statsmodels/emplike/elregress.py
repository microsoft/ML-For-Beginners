"""
Empirical Likelihood Linear Regression Inference

The script contains the function that is optimized over nuisance parameters to
 conduct inference on linear regression parameters.  It is called by eltest
in OLSResults.


General References
-----------------

Owen, A.B.(2001). Empirical Likelihood. Chapman and Hall

"""
import numpy as np
from statsmodels.emplike.descriptive import _OptFuncts



class _ELRegOpts(_OptFuncts):
    """

    A class that holds functions to be optimized over when conducting
    hypothesis tests and calculating confidence intervals.

    Parameters
    ----------

    OLSResults : Results instance
        A fitted OLS result.
    """
    def __init__(self):
        pass

    def _opt_nuis_regress(self, nuisance_params, param_nums=None,
                          endog=None, exog=None,
                          nobs=None, nvar=None, params=None, b0_vals=None,
                          stochastic_exog=None):
        """
        A function that is optimized over nuisance parameters to conduct a
        hypothesis test for the parameters of interest.

        Parameters
        ----------
        nuisance_params: 1darray
            Parameters to be optimized over.

        Returns
        -------
        llr : float
            -2 x the log-likelihood of the nuisance parameters and the
            hypothesized value of the parameter(s) of interest.
        """
        params[param_nums] = b0_vals
        nuis_param_index = np.int_(np.delete(np.arange(nvar),
                                             param_nums))
        params[nuis_param_index] = nuisance_params
        new_params = params.reshape(nvar, 1)
        self.new_params = new_params
        est_vect = exog * \
          (endog - np.squeeze(np.dot(exog, new_params))).reshape(int(nobs), 1)
        if not stochastic_exog:
            exog_means = np.mean(exog, axis=0)[1:]
            exog_mom2 = (np.sum(exog * exog, axis=0))[1:]\
                          / nobs
            mean_est_vect = exog[:, 1:] - exog_means
            mom2_est_vect = (exog * exog)[:, 1:] - exog_mom2
            regressor_est_vect = np.concatenate((mean_est_vect, mom2_est_vect),
                                                axis=1)
            est_vect = np.concatenate((est_vect, regressor_est_vect),
                                           axis=1)

        wts = np.ones(int(nobs)) * (1. / nobs)
        x0 = np.zeros(est_vect.shape[1]).reshape(-1, 1)
        try:
            eta_star = self._modif_newton(x0, est_vect, wts)
            denom = 1. + np.dot(eta_star, est_vect.T)
            self.new_weights = 1. / nobs * 1. / denom
            # the following commented out code is to verify weights
            # see open issue #1845
            #self.new_weights /= self.new_weights.sum()
            #if not np.allclose(self.new_weights.sum(), 1., rtol=0, atol=1e-10):
            #    raise RuntimeError('weights do not sum to 1')
            llr = np.sum(np.log(nobs * self.new_weights))
            return -2 * llr
        except np.linalg.linalg.LinAlgError:
            return np.inf
