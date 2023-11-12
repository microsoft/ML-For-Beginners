from statsmodels.regression.linear_model import OLS
import numpy as np


def _calc_nodewise_row(exog, idx, alpha):
    """calculates the nodewise_row values for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    idx : scalar
        Index of the current variable.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    An array-like object of length p-1

    Notes
    -----

    nodewise_row_i = arg min 1/(2n) ||exog_i - exog_-i gamma||_2^2
                             + alpha ||gamma||_1
    """

    p = exog.shape[1]
    ind = list(range(p))
    ind.pop(idx)

    # handle array alphas
    if not np.isscalar(alpha):
        alpha = alpha[ind]

    tmod = OLS(exog[:, idx], exog[:, ind])

    nodewise_row = tmod.fit_regularized(alpha=alpha).params

    return nodewise_row


def _calc_nodewise_weight(exog, nodewise_row, idx, alpha):
    """calculates the nodewise_weightvalue for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    nodewise_row : array_like
        The nodewise_row values for the current variable.
    idx : scalar
        Index of the current variable
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    A scalar

    Notes
    -----

    nodewise_weight_i = sqrt(1/n ||exog,i - exog_-i nodewise_row||_2^2
                             + alpha ||nodewise_row||_1)
    """

    n, p = exog.shape
    ind = list(range(p))
    ind.pop(idx)

    # handle array alphas
    if not np.isscalar(alpha):
        alpha = alpha[ind]

    d = np.linalg.norm(exog[:, idx] - exog[:, ind].dot(nodewise_row))**2
    d = np.sqrt(d / n + alpha * np.linalg.norm(nodewise_row, 1))
    return d


def _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l):
    """calculates the approximate inverse covariance matrix

    Parameters
    ----------
    nodewise_row_l : list
        A list of array-like object where each object corresponds to
        the nodewise_row values for the corresponding variable, should
        be length p.
    nodewise_weight_l : list
        A list of scalars where each scalar corresponds to the nodewise_weight
        value for the corresponding variable, should be length p.

    Returns
    ------
    An array-like object, p x p matrix

    Notes
    -----

    nwr = nodewise_row
    nww = nodewise_weight

    approx_inv_cov_j = - 1 / nww_j [nwr_j,1,...,1,...nwr_j,p]
    """

    p = len(nodewise_weight_l)

    approx_inv_cov = -np.eye(p)
    for idx in range(p):
        ind = list(range(p))
        ind.pop(idx)
        approx_inv_cov[idx, ind] = nodewise_row_l[idx]
    approx_inv_cov *= -1 / nodewise_weight_l[:, None]**2

    return approx_inv_cov


class RegularizedInvCovariance:
    """
    Class for estimating regularized inverse covariance with
    nodewise regression

    Parameters
    ----------
    exog : array_like
        A weighted design matrix for covariance

    Attributes
    ----------
    exog : array_like
        A weighted design matrix for covariance
    alpha : scalar
        Regularizing constant
    """

    def __init__(self, exog):

        self.exog = exog

    def fit(self, alpha=0):
        """estimates the regularized inverse covariance using nodewise
        regression

        Parameters
        ----------
        alpha : scalar
            Regularizing constant
        """

        n, p = self.exog.shape

        nodewise_row_l = []
        nodewise_weight_l = []

        for idx in range(p):
            nodewise_row = _calc_nodewise_row(self.exog, idx, alpha)
            nodewise_row_l.append(nodewise_row)

            nodewise_weight = _calc_nodewise_weight(self.exog, nodewise_row,
                                                    idx, alpha)
            nodewise_weight_l.append(nodewise_weight)

        nodewise_row_l = np.array(nodewise_row_l)
        nodewise_weight_l = np.array(nodewise_weight_l)

        approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l,
                                              nodewise_weight_l)

        self._approx_inv_cov = approx_inv_cov

    def approx_inv_cov(self):
        return self._approx_inv_cov
