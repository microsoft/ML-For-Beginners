import numpy as np

# TODO: add plots to weighting functions for online docs.


def _cabs(x):
    """absolute value function that changes complex sign based on real sign

    This could be useful for complex step derivatives of functions that
    need abs. Not yet used.
    """
    sign = (x.real >= 0) * 2 - 1
    return sign * x


class RobustNorm:
    """
    The parent class for the norms used for robust regression.

    Lays out the methods expected of the robust norms to be used
    by statsmodels.RLM.

    See Also
    --------
    statsmodels.rlm

    Notes
    -----
    Currently only M-estimators are available.

    References
    ----------
    PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

    DC Montgomery, EA Peck. 'Introduction to Linear Regression Analysis',
        John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. 'Modern Applied Statistics in S'
        Springer, New York, 2002.
    """

    def rho(self, z):
        """
        The robust criterion estimator function.

        Abstract method:

        -2 loglike used in M-estimator
        """
        raise NotImplementedError

    def psi(self, z):
        """
        Derivative of rho.  Sometimes referred to as the influence function.

        Abstract method:

        psi = rho'
        """
        raise NotImplementedError

    def weights(self, z):
        """
        Returns the value of psi(z) / z

        Abstract method:

        psi(z) / z
        """
        raise NotImplementedError

    def psi_deriv(self, z):
        """
        Derivative of psi.  Used to obtain robust covariance matrix.

        See statsmodels.rlm for more information.

        Abstract method:

        psi_derive = psi'
        """
        raise NotImplementedError

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)


class LeastSquares(RobustNorm):
    """
    Least squares rho for M-estimation and its derived functions.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def rho(self, z):
        """
        The least squares estimator rho function

        Parameters
        ----------
        z : ndarray
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2
        """

        return z**2 * 0.5

    def psi(self, z):
        """
        The psi function for the least squares estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z
        """

        return np.asarray(z)

    def weights(self, z):
        """
        The least squares estimator weighting function for the IRLS algorithm.

        The psi function scaled by the input z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = np.ones(z.shape)
        """

        z = np.asarray(z)
        return np.ones(z.shape, np.float64)

    def psi_deriv(self, z):
        """
        The derivative of the least squares psi function.

        Returns
        -------
        psi_deriv : ndarray
            ones(z.shape)

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.ones(z.shape, np.float64)


class HuberT(RobustNorm):
    """
    Huber's T for M estimation.

    Parameters
    ----------
    t : float, optional
        The tuning constant for Huber's t function. The default value is
        1.345.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, t=1.345):
        self.t = t

    def _subset(self, z):
        """
        Huber's T is defined piecewise over the range for z
        """
        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.t)

    def rho(self, z):
        r"""
        The robust criterion function for Huber's t.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = .5*z**2            for \|z\| <= t

            rho(z) = \|z\|*t - .5*t**2    for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return (test * 0.5 * z**2 +
                (1 - test) * (np.abs(z) * self.t - 0.5 * self.t**2))

    def psi(self, z):
        r"""
        The psi function for Huber's t estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z      for \|z\| <= t

            psi(z) = sign(z)*t for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z + (1 - test) * self.t * np.sign(z)

    def weights(self, z):
        r"""
        Huber's t weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1          for \|z\| <= t

            weights(z) = t/\|z\|      for \|z\| > t
        """
        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)

        test = self._subset(z)
        absz = np.abs(z)
        absz[test] = 1.0
        v = test + (1 - test) * self.t / absz

        if z_isscalar:
            v = v[0]
        return v

    def psi_deriv(self, z):
        """
        The derivative of Huber's t psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.less_equal(np.abs(z), self.t).astype(float)


# TODO: untested, but looks right.  RamsayE not available in R or SAS?
class RamsayE(RobustNorm):
    """
    Ramsay's Ea for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        0.3.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=.3):
        self.a = a

    def rho(self, z):
        r"""
        The robust criterion function for Ramsay's Ea.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = a**-2 * (1 - exp(-a*\|z\|)*(1 + a*\|z\|))
        """
        z = np.asarray(z)
        return (1 - np.exp(-self.a * np.abs(z)) *
                (1 + self.a * np.abs(z))) / self.a**2

    def psi(self, z):
        r"""
        The psi function for Ramsay's Ea estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*exp(-a*\|z\|)
        """
        z = np.asarray(z)
        return z * np.exp(-self.a * np.abs(z))

    def weights(self, z):
        r"""
        Ramsay's Ea weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = exp(-a*\|z\|)
        """

        z = np.asarray(z)
        return np.exp(-self.a * np.abs(z))

    def psi_deriv(self, z):
        """
        The derivative of Ramsay's Ea psi function.

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        a = self.a
        x = np.exp(-a * np.abs(z))
        dx = -a * x * np.sign(z)
        y = z
        dy = 1
        return x * dy + y * dx


class AndrewWave(RobustNorm):
    """
    Andrew's wave for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Andrew's Wave function.  The default value is
        1.339.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """
    def __init__(self, a=1.339):
        self.a = a

    def _subset(self, z):
        """
        Andrew's wave is defined piecewise over the range of z.
        """
        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.a * np.pi)

    def rho(self, z):
        r"""
        The robust criterion function for Andrew's wave.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            The elements of rho are defined as:

            .. math::

                rho(z) & = a^2 *(1-cos(z/a)), |z| \leq a\pi \\
                rho(z) & = 2a, |z|>q\pi
        """

        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        return (test * a**2 * (1 - np.cos(z / a)) +
                (1 - test) * a**2 * 2)

    def psi(self, z):
        r"""
        The psi function for Andrew's wave

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = a * sin(z/a)   for \|z\| <= a*pi

            psi(z) = 0              for \|z\| > a*pi
        """

        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        return test * a * np.sin(z / a)

    def weights(self, z):
        r"""
        Andrew's wave weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = sin(z/a) / (z/a)     for \|z\| <= a*pi

            weights(z) = 0                    for \|z\| > a*pi
        """
        a = self.a
        z = np.asarray(z)
        test = self._subset(z)
        ratio = z / a
        small = np.abs(ratio) < np.finfo(np.double).eps
        if np.any(small):
            weights = np.ones_like(ratio)
            large = ~small
            ratio = ratio[large]
            weights[large] = test[large] * np.sin(ratio) / ratio
        else:
            weights = test * np.sin(ratio) / ratio
        return weights

    def psi_deriv(self, z):
        """
        The derivative of Andrew's wave psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """

        test = self._subset(z)
        return test * np.cos(z / self.a)


# TODO: this is untested
class TrimmedMean(RobustNorm):
    """
    Trimmed mean function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        2.0.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, c=2.):
        self.c = c

    def _subset(self, z):
        """
        Least trimmed mean is defined piecewise over the range of z.
        """

        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.c)

    def rho(self, z):
        r"""
        The robust criterion function for least trimmed mean.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2    for \|z\| <= c

            rho(z) = (1/2.)*c**2              for \|z\| > c
        """

        z = np.asarray(z)
        test = self._subset(z)
        return test * z**2 * 0.5 + (1 - test) * self.c**2 * 0.5

    def psi(self, z):
        r"""
        The psi function for least trimmed mean

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z              for \|z\| <= c

            psi(z) = 0              for \|z\| > c
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z

    def weights(self, z):
        r"""
        Least trimmed mean weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1             for \|z\| <= c

            weights(z) = 0             for \|z\| > c
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test

    def psi_deriv(self, z):
        """
        The derivative of least trimmed mean psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        test = self._subset(z)
        return test


class Hampel(RobustNorm):
    """

    Hampel function for M-estimation.

    Parameters
    ----------
    a : float, optional
    b : float, optional
    c : float, optional
        The tuning constants for Hampel's function.  The default values are
        a,b,c = 2, 4, 8.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=2., b=4., c=8.):
        self.a = a
        self.b = b
        self.c = c

    def _subset(self, z):
        """
        Hampel's function is defined piecewise over the range of z
        """
        z = np.abs(np.asarray(z))
        t1 = np.less_equal(z, self.a)
        t2 = np.less_equal(z, self.b) * np.greater(z, self.a)
        t3 = np.less_equal(z, self.c) * np.greater(z, self.b)
        return t1, t2, t3

    def rho(self, z):
        r"""
        The robust criterion function for Hampel's estimator

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = z**2 / 2                     for \|z\| <= a

            rho(z) = a*\|z\| - 1/2.*a**2               for a < \|z\| <= b

            rho(z) = a*(c - \|z\|)**2 / (c - b) / 2    for b < \|z\| <= c

            rho(z) = a*(b + c - a) / 2                 for \|z\| > c
        """
        a, b, c = self.a, self.b, self.c

        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)

        t1, t2, t3 = self._subset(z)
        t34 = ~(t1 | t2)
        dt = np.promote_types(z.dtype, "float")
        v = np.zeros(z.shape, dtype=dt)
        z = np.abs(z)
        v[t1] = z[t1]**2 * 0.5
        # v[t2] = (a * (z[t2] - a) + a**2 * 0.5)
        v[t2] = (a * z[t2] - a**2 * 0.5)
        v[t3] = a * (c - z[t3])**2 / (c - b) * (-0.5)
        v[t34] += a * (b + c - a) * 0.5

        if z_isscalar:
            v = v[0]

        return v

    def psi(self, z):
        r"""
        The psi function for Hampel's estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z                            for \|z\| <= a

            psi(z) = a*sign(z)                    for a < \|z\| <= b

            psi(z) = a*sign(z)*(c - \|z\|)/(c-b)    for b < \|z\| <= c

            psi(z) = 0                            for \|z\| > c
        """
        a, b, c = self.a, self.b, self.c

        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)

        t1, t2, t3 = self._subset(z)
        dt = np.promote_types(z.dtype, "float")
        v = np.zeros(z.shape, dtype=dt)
        s = np.sign(z)
        za = np.abs(z)

        v[t1] = z[t1]
        v[t2] = a * s[t2]
        v[t3] = a * s[t3] * (c - za[t3]) / (c - b)

        if z_isscalar:
            v = v[0]
        return v

    def weights(self, z):
        r"""
        Hampel weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1                                for \|z\| <= a

            weights(z) = a/\|z\|                          for a < \|z\| <= b

            weights(z) = a*(c - \|z\|)/(\|z\|*(c-b))      for b < \|z\| <= c

            weights(z) = 0                                for \|z\| > c
        """
        a, b, c = self.a, self.b, self.c

        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)

        t1, t2, t3 = self._subset(z)

        dt = np.promote_types(z.dtype, "float")
        v = np.zeros(z.shape, dtype=dt)
        v[t1] = 1.0
        abs_z = np.abs(z)
        v[t2] = a / abs_z[t2]
        abs_zt3 = abs_z[t3]
        v[t3] = a * (c - abs_zt3) / (abs_zt3 * (c - b))

        if z_isscalar:
            v = v[0]
        return v

    def psi_deriv(self, z):
        """Derivative of psi function, second derivative of rho function.
        """
        a, b, c = self.a, self.b, self.c

        z_isscalar = np.isscalar(z)
        z = np.atleast_1d(z)

        t1, _, t3 = self._subset(z)

        dt = np.promote_types(z.dtype, "float")
        d = np.zeros(z.shape, dtype=dt)
        d[t1] = 1.0
        zt3 = z[t3]
        d[t3] = -(a * np.sign(zt3) * zt3) / (np.abs(zt3) * (c - b))

        if z_isscalar:
            d = d[0]
        return d


class TukeyBiweight(RobustNorm):
    """

    Tukey's biweight function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Tukey's Biweight.  The default value is
        c = 4.685.

    Notes
    -----
    Tukey's biweight is sometime's called bisquare.
    """

    def __init__(self, c=4.685):
        self.c = c

    def _subset(self, z):
        """
        Tukey's biweight is defined piecewise over the range of z
        """
        z = np.abs(np.asarray(z))
        return np.less_equal(z, self.c)

    def rho(self, z):
        r"""
        The robust criterion function for Tukey's biweight estimator

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = -(1 - (z/c)**2)**3 * c**2/6.   for \|z\| <= R

            rho(z) = 0                              for \|z\| > R
        """
        subset = self._subset(z)
        factor = self.c**2 / 6.
        return -(1 - (z / self.c)**2)**3 * subset * factor + factor

    def psi(self, z):
        r"""
        The psi function for Tukey's biweight estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*(1 - (z/c)**2)**2        for \|z\| <= R

            psi(z) = 0                           for \|z\| > R
        """

        z = np.asarray(z)
        subset = self._subset(z)
        return z * (1 - (z / self.c)**2)**2 * subset

    def weights(self, z):
        r"""
        Tukey's biweight weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            psi(z) = (1 - (z/c)**2)**2          for \|z\| <= R

            psi(z) = 0                          for \|z\| > R
        """

        subset = self._subset(z)
        return (1 - (z / self.c)**2)**2 * subset

    def psi_deriv(self, z):
        """
        The derivative of Tukey's biweight psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        subset = self._subset(z)
        return subset * ((1 - (z/self.c)**2)**2
                         - (4*z**2/self.c**2) * (1-(z/self.c)**2))


class MQuantileNorm(RobustNorm):
    """M-quantiles objective function based on a base norm

    This norm has the same asymmetric structure as the objective function
    in QuantileRegression but replaces the L1 absolute value by a chosen
    base norm.

        rho_q(u) = abs(q - I(q < 0)) * rho_base(u)

    or, equivalently,

        rho_q(u) = q * rho_base(u)  if u >= 0
        rho_q(u) = (1 - q) * rho_base(u)  if u < 0


    Parameters
    ----------
    q : float
        M-quantile, must be between 0 and 1
    base_norm : RobustNorm instance
        basic norm that is transformed into an asymmetric M-quantile norm

    Notes
    -----
    This is mainly for base norms that are not redescending, like HuberT or
    LeastSquares. (See Jones for the relationship of M-quantiles to quantiles
    in the case of non-redescending Norms.)

    Expectiles are M-quantiles with the LeastSquares as base norm.

    References
    ----------

    .. [*] Bianchi, Annamaria, and Nicola Salvati. 2015. “Asymptotic Properties
       and Variance Estimators of the M-Quantile Regression Coefficients
       Estimators.” Communications in Statistics - Theory and Methods 44 (11):
       2416–29. doi:10.1080/03610926.2013.791375.

    .. [*] Breckling, Jens, and Ray Chambers. 1988. “M-Quantiles.”
       Biometrika 75 (4): 761–71. doi:10.2307/2336317.

    .. [*] Jones, M. C. 1994. “Expectiles and M-Quantiles Are Quantiles.”
       Statistics & Probability Letters 20 (2): 149–53.
       doi:10.1016/0167-7152(94)90031-0.

    .. [*] Newey, Whitney K., and James L. Powell. 1987. “Asymmetric Least
       Squares Estimation and Testing.” Econometrica 55 (4): 819–47.
       doi:10.2307/1911031.
    """

    def __init__(self, q, base_norm):
        self.q = q
        self.base_norm = base_norm

    def _get_q(self, z):

        nobs = len(z)
        mask_neg = (z < 0)  # if self.q < 0.5 else (z <= 0)  # maybe symmetric
        qq = np.empty(nobs)
        qq[mask_neg] = 1 - self.q
        qq[~mask_neg] = self.q
        return qq

    def rho(self, z):
        """
        The robust criterion function for MQuantileNorm.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.rho(z)

    def psi(self, z):
        """
        The psi function for MQuantileNorm estimator.

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.psi(z)

    def weights(self, z):
        """
        MQuantileNorm weighting function for the IRLS algorithm

        The psi function scaled by z, psi(z) / z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
        """
        qq = self._get_q(z)
        return qq * self.base_norm.weights(z)

    def psi_deriv(self, z):
        '''
        The derivative of MQuantileNorm function

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi_deriv : ndarray

        Notes
        -----
        Used to estimate the robust covariance matrix.
        '''
        qq = self._get_q(z)
        return qq * self.base_norm.psi_deriv(z)

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)


def estimate_location(a, scale, norm=None, axis=0, initial=None,
                      maxiter=30, tol=1.0e-06):
    """
    M-estimator of location using self.norm and a current
    estimator of scale.

    This iteratively finds a solution to

    norm.psi((a-mu)/scale).sum() == 0

    Parameters
    ----------
    a : ndarray
        Array over which the location parameter is to be estimated
    scale : ndarray
        Scale parameter to be used in M-estimator
    norm : RobustNorm, optional
        Robust norm used in the M-estimator.  The default is HuberT().
    axis : int, optional
        Axis along which to estimate the location parameter.  The default is 0.
    initial : ndarray, optional
        Initial condition for the location parameter.  Default is None, which
        uses the median of a.
    niter : int, optional
        Maximum number of iterations.  The default is 30.
    tol : float, optional
        Toleration for convergence.  The default is 1e-06.

    Returns
    -------
    mu : ndarray
        Estimate of location
    """
    if norm is None:
        norm = HuberT()

    if initial is None:
        mu = np.median(a, axis)
    else:
        mu = initial

    for _ in range(maxiter):
        W = norm.weights((a-mu)/scale)
        nmu = np.sum(W*a, axis) / np.sum(W, axis)
        if np.all(np.less(np.abs(mu - nmu), scale * tol)):
            return nmu
        else:
            mu = nmu
    raise ValueError("location estimator failed to converge in %d iterations"
                     % maxiter)
