"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
    Princeton University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
    and Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
    with Categorical and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)
[5] Liu, R., Yang, L. "Kernel estimation of multivariate
    cumulative distribution function."
    Journal of Nonparametric Statistics (2008)
[6] Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
    with Categorical and Continuous Data." Working Paper
[7] Li, Q., Racine, J. "Cross-validated local linear nonparametric
    regression" Statistica Sinica 14(2004), pp. 485-512
[8] Racine, J.: "Consistent Significance Testing for Nonparametric
        Regression" Journal of Business & Economics Statistics
[9] Racine, J., Hart, J., Li, Q., "Testing the Significance of
        Categorical Predictor Variables in Nonparametric Regression
        Models", 2006, Econometric Reviews 25, 523-544

"""

# TODO: make default behavior efficient=True above a certain n_obs
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles

from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
    gpke, LeaveOneOut, _get_type_pos, _adjust_shape


__all__ = ['SingleIndexModel', 'SemiLinear', 'TestFForm']


class TestFForm:
    """
    Nonparametric test for functional form.

    Parameters
    ----------
    endog : list
        Dependent variable (training set)
    exog : list of array_like objects
        The independent (right-hand-side) variables
    bw : array_like, str
        Bandwidths for exog or specify method for bandwidth selection
    fform : function
        The functional form ``y = g(b, x)`` to be tested. Takes as inputs
        the RHS variables `exog` and the coefficients ``b`` (betas)
        and returns a fitted ``y_hat``.
    var_type : str
        The type of the independent `exog` variables:

            - c: continuous
            - o: ordered
            - u: unordered

    estimator : function
        Must return the estimated coefficients b (betas). Takes as inputs
        ``(endog, exog)``.  E.g. least square estimator::

            lambda (x,y): np.dot(np.pinv(np.dot(x.T, x)), np.dot(x.T, y))

    References
    ----------
    See Racine, J.: "Consistent Significance Testing for Nonparametric
    Regression" Journal of Business & Economics Statistics.

    See chapter 12 in [1]  pp. 355-357.
    """
    def __init__(self, endog, exog, bw, var_type, fform, estimator, nboot=100):
        self.endog = endog
        self.exog = exog
        self.var_type = var_type
        self.fform = fform
        self.estimator = estimator
        self.nboot = nboot
        self.bw = KDEMultivariate(exog, bw=bw, var_type=var_type).bw
        self.sig = self._compute_sig()

    def _compute_sig(self):
        Y = self.endog
        X = self.exog
        b = self.estimator(Y, X)
        m = self.fform(X, b)
        n = np.shape(X)[0]
        resid = Y - m
        resid = resid - np.mean(resid)  # center residuals
        self.test_stat = self._compute_test_stat(resid)
        sqrt5 = np.sqrt(5.)
        fct1 = (1 - sqrt5) / 2.
        fct2 = (1 + sqrt5) / 2.
        u1 = fct1 * resid
        u2 = fct2 * resid
        r = fct2 / sqrt5
        I_dist = np.empty((self.nboot,1))
        for j in range(self.nboot):
            u_boot = u2.copy()

            prob = np.random.uniform(0,1, size = (n,))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            b_hat = self.estimator(Y_boot, X)
            m_hat = self.fform(X, b_hat)
            u_boot_hat = Y_boot - m_hat
            I_dist[j] = self._compute_test_stat(u_boot_hat)

        self.boots_results = I_dist
        sig = "Not Significant"
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = "*"
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = "**"
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = "***"
        return sig

    def _compute_test_stat(self, u):
        n = np.shape(u)[0]
        XLOO = LeaveOneOut(self.exog)
        uLOO = LeaveOneOut(u[:,None]).__iter__()
        ival = 0
        S2 = 0
        for i, X_not_i in enumerate(XLOO):
            u_j = next(uLOO)
            u_j = np.squeeze(u_j)
            # See Bootstrapping procedure on p. 357 in [1]
            K = gpke(self.bw, data=-X_not_i, data_predict=-self.exog[i, :],
                     var_type=self.var_type, tosum=False)
            f_i = (u[i] * u_j * K)
            assert u_j.shape == K.shape
            ival += f_i.sum()  # See eq. 12.7 on p. 355 in [1]
            S2 += (f_i**2).sum()  # See Theorem 12.1 on p.356 in [1]
            assert np.size(ival) == 1
            assert np.size(S2) == 1

        ival *= 1. / (n * (n - 1))
        ix_cont = _get_type_pos(self.var_type)[0]
        hp = self.bw[ix_cont].prod()
        S2 *= 2 * hp / (n * (n - 1))
        T = n * ival * np.sqrt(hp / S2)
        return T


class SingleIndexModel(KernelReg):
    """
    Single index semiparametric model ``y = g(X * b) + e``.

    Parameters
    ----------
    endog : array_like
        The dependent variable
    exog : array_like
        The independent variable(s)
    var_type : str
        The type of variables in X:

            - c: continuous
            - o: ordered
            - u: unordered

    Attributes
    ----------
    b : array_like
        The linear coefficients b (betas)
    bw : array_like
        Bandwidths

    Methods
    -------
    fit(): Computes the fitted values ``E[Y|X] = g(X * b)``
           and the marginal effects ``dY/dX``.

    References
    ----------
    See chapter on semiparametric models in [1]

    Notes
    -----
    This model resembles the binary choice models. The user knows
    that X and b interact linearly, but ``g(X * b)`` is unknown.
    In the parametric binary choice models the user usually assumes
    some distribution of g() such as normal or logistic.
    """
    def __init__(self, endog, exog, var_type):
        self.var_type = var_type
        self.K = len(var_type)
        self.var_type = self.var_type[0]
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.K)
        self.nobs = np.shape(self.exog)[0]
        self.data_type = self.var_type
        self.ckertype = 'gaussian'
        self.okertype = 'wangryzin'
        self.ukertype = 'aitchisonaitken'
        self.func = self._est_loc_linear

        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        params0 = np.random.uniform(size=(self.K + 1, ))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0:self.K]
        bw = b_bw[self.K:]
        bw = self._set_bw_bounds(bw)
        return b, bw

    def cv_loo(self, params):
        # See p. 254 in Textbook
        params = np.asarray(params)
        b = params[0 : self.K]
        bw = params[self.K:]
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        L = 0
        for i, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            #print b.shape, np.dot(self.exog[i:i+1, :], b).shape, bw,
            G = self.func(bw, endog=Y, exog=-np.dot(X_not_i, b)[:,None],
                          #data_predict=-b*self.exog[i, :])[0]
                          data_predict=-np.dot(self.exog[i:i+1, :], b))[0]
            #print G.shape
            L += (self.endog[i] - G) ** 2

        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.nobs

    def fit(self, data_predict=None):
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = _adjust_shape(data_predict, self.K)

        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        for i in range(N_data_predict):
            mean_mfx = self.func(self.bw, self.endog,
                                 np.dot(self.exog, self.b)[:,None],
                                 data_predict=np.dot(data_predict[i:i+1, :],self.b))
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Single Index Model \n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   nobs = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: cv_ls" + "\n"
        repr += "Estimator type: local constant" + "\n"
        return repr


class SemiLinear(KernelReg):
    """
    Semiparametric partially linear model, ``Y = Xb + g(Z) + e``.

    Parameters
    ----------
    endog : array_like
        The dependent variable
    exog : array_like
        The linear component in the regression
    exog_nonparametric : array_like
        The nonparametric component in the regression
    var_type : str
        The type of the variables in the nonparametric component;

            - c: continuous
            - o: ordered
            - u: unordered

    k_linear : int
        The number of variables that comprise the linear component.

    Attributes
    ----------
    bw : array_like
        Bandwidths for the nonparametric component exog_nonparametric
    b : array_like
        Coefficients in the linear component
    nobs : int
        The number of observations.
    k_linear : int
        The number of variables that comprise the linear component.

    Methods
    -------
    fit
        Returns the fitted mean and marginal effects dy/dz

    Notes
    -----
    This model uses only the local constant regression estimator

    References
    ----------
    See chapter on Semiparametric Models in [1]
    """

    def __init__(self, endog, exog, exog_nonparametric, var_type, k_linear):
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, k_linear)
        self.K = len(var_type)
        self.exog_nonparametric = _adjust_shape(exog_nonparametric, self.K)
        self.k_linear = k_linear
        self.nobs = np.shape(self.exog)[0]
        self.var_type = var_type
        self.data_type = self.var_type
        self.ckertype = 'gaussian'
        self.okertype = 'wangryzin'
        self.ukertype = 'aitchisonaitken'
        self.func = self._est_loc_linear

        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        """
        Computes the (beta) coefficients and the bandwidths.

        Minimizes ``cv_loo`` with respect to ``b`` and ``bw``.
        """
        params0 = np.random.uniform(size=(self.k_linear + self.K, ))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0 : self.k_linear]
        bw = b_bw[self.k_linear:]
        #bw = self._set_bw_bounds(np.asarray(bw))
        return b, bw

    def cv_loo(self, params):
        """
        Similar to the cross validation leave-one-out estimator.

        Modified to reflect the linear components.

        Parameters
        ----------
        params : array_like
            Vector consisting of the coefficients (b) and the bandwidths (bw).
            The first ``k_linear`` elements are the coefficients.

        Returns
        -------
        L : float
            The value of the objective function

        References
        ----------
        See p.254 in [1]
        """
        params = np.asarray(params)
        b = params[0 : self.k_linear]
        bw = params[self.k_linear:]
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        LOO_Z = LeaveOneOut(self.exog_nonparametric).__iter__()
        Xb = np.dot(self.exog, b)[:,None]
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            Z = next(LOO_Z)
            Xb_j = np.dot(X_not_i, b)[:,None]
            Yx = Y - Xb_j
            G = self.func(bw, endog=Yx, exog=-Z,
                          data_predict=-self.exog_nonparametric[ii, :])[0]
            lt = Xb[ii, :] #.sum()  # linear term
            L += (self.endog[ii] - lt - G) ** 2

        return L

    def fit(self, exog_predict=None, exog_nonparametric_predict=None):
        """Computes fitted values and marginal effects"""

        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = _adjust_shape(exog_predict, self.k_linear)

        if exog_nonparametric_predict is None:
            exog_nonparametric_predict = self.exog_nonparametric
        else:
            exog_nonparametric_predict = _adjust_shape(exog_nonparametric_predict, self.K)

        N_data_predict = np.shape(exog_nonparametric_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        Y = self.endog - np.dot(exog_predict, self.b)[:,None]
        for i in range(N_data_predict):
            mean_mfx = self.func(self.bw, Y, self.exog_nonparametric,
                                 data_predict=exog_nonparametric_predict[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Semiparamatric Partially Linear Model \n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: cv_ls" + "\n"
        repr += "Estimator type: local constant" + "\n"
        return repr
