"""
The RegressionFDR class implements the 'Knockoff' approach for
controlling false discovery rates (FDR) in regression analysis.

The knockoff approach does not require standard errors.  Thus one
application is to provide inference for parameter estimates that are
not smooth functions of the data.  For example, the knockoff approach
can be used to do inference for parameter estimates obtained from the
LASSO, of from stepwise variable selection.

The knockoff approach controls FDR for parameter estimates that may be
dependent, such as coefficient estimates in a multiple regression
model.

The knockoff approach is applicable whenever the test statistic can be
computed entirely from x'y and x'x, where x is the design matrix and y
is the vector of responses.

Reference
---------
Rina Foygel Barber, Emmanuel Candes (2015).  Controlling the False
Discovery Rate via Knockoffs.  Annals of Statistics 43:5.
https://candes.su.domains/publications/downloads/FDR_regression.pdf
"""

import numpy as np
import pandas as pd
from statsmodels.iolib import summary2


class RegressionFDR:
    """
    Control FDR in a regression procedure.

    Parameters
    ----------
    endog : array_like
        The dependent variable of the regression
    exog : array_like
        The independent variables of the regression
    regeffects : RegressionEffects instance
        An instance of a RegressionEffects class that can compute
        effect sizes for the regression coefficients.
    method : str
        The approach used to assess and control FDR, currently
        must be 'knockoff'.

    Returns
    -------
    Returns an instance of the RegressionFDR class.  The `fdr` attribute
    holds the estimated false discovery rates.

    Notes
    -----
    This class Implements the knockoff method of Barber and Candes.
    This is an approach for controlling the FDR of a variety of
    regression estimation procedures, including correlation
    coefficients, OLS regression, OLS with forward selection, and
    LASSO regression.

    For other approaches to FDR control in regression, see the
    statsmodels.stats.multitest module.  Methods provided in that
    module use Z-scores or p-values, and therefore require standard
    errors for the coefficient estimates to be available.

    The default method for constructing the augmented design matrix is
    the 'equivariant' approach, set `design_method='sdp'` to use an
    alternative approach involving semidefinite programming.  See
    Barber and Candes for more information about both approaches.  The
    sdp approach requires that the cvxopt package be installed.
    """

    def __init__(self, endog, exog, regeffects, method="knockoff",
                 **kwargs):

        if hasattr(exog, "columns"):
            self.xnames = exog.columns
        else:
            self.xnames = ["x%d" % j for j in range(exog.shape[1])]

        exog = np.asarray(exog)
        endog = np.asarray(endog)

        if "design_method" not in kwargs:
            kwargs["design_method"] = "equi"

        nobs, nvar = exog.shape

        if kwargs["design_method"] == "equi":
            exog1, exog2, _ = _design_knockoff_equi(exog)
        elif kwargs["design_method"] == "sdp":
            exog1, exog2, _ = _design_knockoff_sdp(exog)
        endog = endog - np.mean(endog)

        self.endog = endog
        self.exog = np.concatenate((exog1, exog2), axis=1)
        self.exog1 = exog1
        self.exog2 = exog2

        self.stats = regeffects.stats(self)

        unq, inv, cnt = np.unique(self.stats, return_inverse=True,
                                  return_counts=True)

        # The denominator of the FDR
        cc = np.cumsum(cnt)
        denom = len(self.stats) - cc + cnt
        denom[denom < 1] = 1

        # The numerator of the FDR
        ii = np.searchsorted(unq, -unq, side='right') - 1
        numer = cc[ii]
        numer[ii < 0] = 0

        # The knockoff+ estimated FDR
        fdrp = (1 + numer) / denom

        # The knockoff estimated FDR
        fdr = numer / denom

        self.fdr = fdr[inv]
        self.fdrp = fdrp[inv]
        self._ufdr = fdr
        self._unq = unq

        df = pd.DataFrame(index=self.xnames)
        df["Stat"] = self.stats
        df["FDR+"] = self.fdrp
        df["FDR"] = self.fdr
        self.fdr_df = df

    def threshold(self, tfdr):
        """
        Returns the threshold statistic for a given target FDR.
        """

        if np.min(self._ufdr) <= tfdr:
            return self._unq[self._ufdr <= tfdr][0]
        else:
            return np.inf

    def summary(self):

        summ = summary2.Summary()
        summ.add_title("Regression FDR results")
        summ.add_df(self.fdr_df)

        return summ


def _design_knockoff_sdp(exog):
    """
    Use semidefinite programming to construct a knockoff design
    matrix.

    Requires cvxopt to be installed.
    """

    try:
        from cvxopt import solvers, matrix
    except ImportError:
        raise ValueError("SDP knockoff designs require installation of cvxopt")

    nobs, nvar = exog.shape

    # Standardize exog
    xnm = np.sum(exog**2, 0)
    xnm = np.sqrt(xnm)
    exog = exog / xnm

    Sigma = np.dot(exog.T, exog)

    c = matrix(-np.ones(nvar))

    h0 = np.concatenate((np.zeros(nvar), np.ones(nvar)))
    h0 = matrix(h0)
    G0 = np.concatenate((-np.eye(nvar), np.eye(nvar)), axis=0)
    G0 = matrix(G0)

    h1 = 2 * Sigma
    h1 = matrix(h1)
    i, j = np.diag_indices(nvar)
    G1 = np.zeros((nvar*nvar, nvar))
    G1[i*nvar + j, i] = 1
    G1 = matrix(G1)

    solvers.options['show_progress'] = False
    sol = solvers.sdp(c, G0, h0, [G1], [h1])
    sl = np.asarray(sol['x']).ravel()

    xcov = np.dot(exog.T, exog)
    exogn = _get_knmat(exog, xcov, sl)

    return exog, exogn, sl


def _design_knockoff_equi(exog):
    """
    Construct an equivariant design matrix for knockoff analysis.

    Follows the 'equi-correlated knockoff approach of equation 2.4 in
    Barber and Candes.

    Constructs a pair of design matrices exogs, exogn such that exogs
    is a scaled/centered version of the input matrix exog, exogn is
    another matrix of the same shape with cov(exogn) = cov(exogs), and
    the covariances between corresponding columns of exogn and exogs
    are as small as possible.
    """

    nobs, nvar = exog.shape

    if nobs < 2*nvar:
        msg = "The equivariant knockoff can ony be used when n >= 2*p"
        raise ValueError(msg)

    # Standardize exog
    xnm = np.sum(exog**2, 0)
    xnm = np.sqrt(xnm)
    exog = exog / xnm

    xcov = np.dot(exog.T, exog)
    ev, _ = np.linalg.eig(xcov)
    evmin = np.min(ev)

    sl = min(2*evmin, 1)
    sl = sl * np.ones(nvar)

    exogn = _get_knmat(exog, xcov, sl)

    return exog, exogn, sl


def _get_knmat(exog, xcov, sl):
    # Utility function, see equation 2.2 of Barber & Candes.

    nobs, nvar = exog.shape

    ash = np.linalg.inv(xcov)
    ash *= -np.outer(sl, sl)
    i, j = np.diag_indices(nvar)
    ash[i, j] += 2 * sl

    umat = np.random.normal(size=(nobs, nvar))
    u, _ = np.linalg.qr(exog)
    umat -= np.dot(u, np.dot(u.T, umat))
    umat, _ = np.linalg.qr(umat)

    ashr, xc, _ = np.linalg.svd(ash, 0)
    ashr *= np.sqrt(xc)
    ashr = ashr.T

    ex = (sl[:, None] * np.linalg.solve(xcov, exog.T)).T
    exogn = exog - ex + np.dot(umat, ashr)

    return exogn
