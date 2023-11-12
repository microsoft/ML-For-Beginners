"""Restricted least squares

from pandas
License: Simplified BSD
"""
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults


class RLS(GLS):
    """
    Restricted general least squares model that handles linear constraints

    Parameters
    ----------
    endog : array_like
        n length array containing the dependent variable
    exog : array_like
        n-by-p array of independent variables
    constr : array_like
        k-by-p array of linear constraints
    param : array_like or scalar
        p-by-1 array (or scalar) of constraint parameters
    sigma (None): scalar or array_like
        The weighting matrix of the covariance. No scaling by default (OLS).
        If sigma is a scalar, then it is converted into an n-by-n diagonal
        matrix with sigma as each diagonal element.
        If sigma is an n-length array, then it is assumed to be a diagonal
        matrix with the given sigma on the diagonal (WLS).

    Notes
    -----
    endog = exog * beta + epsilon
    weights' * constr * beta = param

    See Greene and Seaks, "The Restricted Least Squares Estimator:
    A Pedagogical Note", The Review of Economics and Statistics, 1991.
    """

    def __init__(self, endog, exog, constr, param=0., sigma=None):
        N, Q = exog.shape
        constr = np.asarray(constr)
        if constr.ndim == 1:
            K, P = 1, constr.shape[0]
        else:
            K, P = constr.shape
        if Q != P:
            raise Exception('Constraints and design do not align')
        self.ncoeffs = Q
        self.nconstraint = K
        self.constraint = constr
        if np.isscalar(param) and K > 1:
            param = np.ones((K,)) * param
        self.param = param
        if sigma is None:
            sigma = 1.
        if np.isscalar(sigma):
            sigma = np.ones(N) * sigma
        sigma = np.squeeze(sigma)
        if sigma.ndim == 1:
            self.sigma = np.diag(sigma)
            self.cholsigmainv = np.diag(np.sqrt(sigma))
        else:
            self.sigma = sigma
            self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        super(GLS, self).__init__(endog, exog)

    _rwexog = None
    @property
    def rwexog(self):
        """Whitened exogenous variables augmented with restrictions"""
        if self._rwexog is None:
            P = self.ncoeffs
            K = self.nconstraint
            design = np.zeros((P + K, P + K))
            design[:P, :P] = np.dot(self.wexog.T, self.wexog) #top left
            constr = np.reshape(self.constraint, (K, P))
            design[:P, P:] = constr.T #top right partition
            design[P:, :P] = constr #bottom left partition
            design[P:, P:] = np.zeros((K, K)) #bottom right partition
            self._rwexog = design
        return self._rwexog

    _inv_rwexog = None
    @property
    def inv_rwexog(self):
        """Inverse of self.rwexog"""
        if self._inv_rwexog is None:
            self._inv_rwexog = np.linalg.inv(self.rwexog)
        return self._inv_rwexog

    _rwendog = None
    @property
    def rwendog(self):
        """Whitened endogenous variable augmented with restriction parameters"""
        if self._rwendog is None:
            P = self.ncoeffs
            K = self.nconstraint
            response = np.zeros((P + K,))
            response[:P] = np.dot(self.wexog.T, self.wendog)
            response[P:] = self.param
            self._rwendog = response
        return self._rwendog

    _ncp = None
    @property
    def rnorm_cov_params(self):
        """Parameter covariance under restrictions"""
        if self._ncp is None:
            P = self.ncoeffs
            self._ncp = self.inv_rwexog[:P, :P]
        return self._ncp

    _wncp = None
    @property
    def wrnorm_cov_params(self):
        """
        Heteroskedasticity-consistent parameter covariance
        Used to calculate White standard errors.
        """
        if self._wncp is None:
            df = self.df_resid
            pred = np.dot(self.wexog, self.coeffs)
            eps = np.diag((self.wendog - pred) ** 2)
            sigmaSq = np.sum(eps)
            pinvX = np.dot(self.rnorm_cov_params, self.wexog.T)
            self._wncp = np.dot(np.dot(pinvX, eps), pinvX.T) * df / sigmaSq
        return self._wncp

    _coeffs = None
    @property
    def coeffs(self):
        """Estimated parameters"""
        if self._coeffs is None:
            betaLambda = np.dot(self.inv_rwexog, self.rwendog)
            self._coeffs = betaLambda[:self.ncoeffs]
        return self._coeffs

    def fit(self):
        rncp = self.wrnorm_cov_params
        lfit = RegressionResults(self, self.coeffs, normalized_cov_params=rncp)
        return lfit

if __name__=="__main__":
    import statsmodels.api as sm
    dta = np.genfromtxt('./rlsdata.txt', names=True)
    design = np.column_stack((dta['Y'],dta['Y']**2,dta[['NE','NC','W','S']].view(float).reshape(dta.shape[0],-1)))
    design = sm.add_constant(design, prepend=True)
    rls_mod = RLS(dta['G'],design, constr=[0,0,0,1,1,1,1])
    rls_fit = rls_mod.fit()
    print(rls_fit.params)
