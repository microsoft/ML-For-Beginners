"""
Support and standalone functions for Robust Linear Models

References
----------
PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

R Venables, B Ripley. 'Modern Applied Statistics in S'
    Springer, New York, 2002.

C Croux, PJ Rousseeuw, 'Time-efficient algorithms for two highly robust
estimators of scale' Computational statistics. Physica, Heidelberg, 1992.
"""
import numpy as np
from scipy.stats import norm as Gaussian

from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like

from . import norms
from ._qn import _qn


def mad(a, c=Gaussian.ppf(3 / 4.0), axis=0, center=np.median):
    """
    The Median Absolute Deviation along given axis of an array

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately 0.6745.
    axis : int, optional
        The default is 0. Can also be None.
    center : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called center(a). The axis argument will be applied
        via np.apply_over_axes. Otherwise, provide a float.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    a = array_like(a, "a", ndim=None)
    c = float_like(c, "c")
    if not a.size:
        center_val = 0.0
    elif callable(center):
        if axis is not None:
            center_val = np.apply_over_axes(center, a, axis)
        else:
            center_val = center(a.ravel())
    else:
        center_val = float_like(center, "center")
    err = (np.abs(a - center_val)) / c
    if not err.size:
        if axis is None or err.ndim == 1:
            return np.nan
        else:
            shape = list(err.shape)
            shape.pop(axis)
            return np.empty(shape)
    return np.median(err, axis=axis)


def iqr(a, c=Gaussian.ppf(3 / 4) - Gaussian.ppf(1 / 4), axis=0):
    """
    The normalized interquartile range along given axis of an array

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        scipy.stats.norm.ppf(3/4.) - scipy.stats.norm.ppf(1/4.), which is
        approximately 1.349.
    axis : int, optional
        The default is 0. Can also be None.

    Returns
    -------
    The normalized interquartile range
    """
    a = array_like(a, "a", ndim=None)
    c = float_like(c, "c")

    if a.ndim == 0:
        raise ValueError("a should have at least one dimension")
    elif a.size == 0:
        return np.nan
    else:
        quantiles = np.quantile(a, [0.25, 0.75], axis=axis)
        return np.squeeze(np.diff(quantiles, axis=0) / c)


def qn_scale(a, c=1 / (np.sqrt(2) * Gaussian.ppf(5 / 8)), axis=0):
    """
    Computes the Qn robust estimator of scale

    The Qn scale estimator is a more efficient alternative to the MAD.
    The Qn scale estimator of an array a of length n is defined as
    c * {abs(a[i] - a[j]): i<j}_(k), for k equal to [n/2] + 1 choose 2. Thus,
    the Qn estimator is the k-th order statistic of the absolute differences
    of the array. The optional constant is used to normalize the estimate
    as explained below. The implementation follows the algorithm described
    in Croux and Rousseeuw (1992).

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant. The default value is used to get consistent
        estimates of the standard deviation at the normal distribution.
    axis : int, optional
        The default is 0.

    Returns
    -------
    {float, ndarray}
        The Qn robust estimator of scale
    """
    a = array_like(
        a, "a", ndim=None, dtype=np.float64, contiguous=True, order="C"
    )
    c = float_like(c, "c")
    if a.ndim == 0:
        raise ValueError("a should have at least one dimension")
    elif a.size == 0:
        return np.nan
    else:
        out = np.apply_along_axis(_qn, axis=axis, arr=a, c=c)
        if out.ndim == 0:
            return float(out)
        return out


def _qn_naive(a, c=1 / (np.sqrt(2) * Gaussian.ppf(5 / 8))):
    """
    A naive implementation of the Qn robust estimator of scale, used solely
    to test the faster, more involved one

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant, used to get consistent estimates of the
        standard deviation at the normal distribution.  Defined as
        1/(np.sqrt(2) * scipy.stats.norm.ppf(5/8)), which is 2.219144.

    Returns
    -------
    The Qn robust estimator of scale
    """
    a = np.squeeze(a)
    n = a.shape[0]
    if a.size == 0:
        return np.nan
    else:
        h = int(n // 2 + 1)
        k = int(h * (h - 1) / 2)
        idx = np.triu_indices(n, k=1)
        diffs = np.abs(a[idx[0]] - a[idx[1]])
        output = np.partition(diffs, kth=k - 1)[k - 1]
        output = c * output
        return output


class Huber:
    """
    Huber's proposal 2 for estimating location and scale jointly.

    Parameters
    ----------
    c : float, optional
        Threshold used in threshold for chi=psi**2.  Default value is 1.5.
    tol : float, optional
        Tolerance for convergence.  Default value is 1e-08.
    maxiter : int, optional0
        Maximum number of iterations.  Default value is 30.
    norm : statsmodels.robust.norms.RobustNorm, optional
        A robust norm used in M estimator of location. If None,
        the location estimator defaults to a one-step
        fixed point version of the M-estimator using Huber's T.

    call
        Return joint estimates of Huber's scale and location.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> chem_data = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
    ...        3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7,
    ...        3.77, 5.28, 28.95])
    >>> sm.robust.scale.huber(chem_data)
    (array(3.2054980819923693), array(0.67365260010478967))
    """

    def __init__(self, c=1.5, tol=1.0e-08, maxiter=30, norm=None):
        self.c = c
        self.maxiter = maxiter
        self.tol = tol
        self.norm = norm
        tmp = 2 * Gaussian.cdf(c) - 1
        self.gamma = tmp + c ** 2 * (1 - tmp) - 2 * c * Gaussian.pdf(c)

    def __call__(self, a, mu=None, initscale=None, axis=0):
        """
        Compute Huber's proposal 2 estimate of scale, using an optional
        initial value of scale and an optional estimate of mu. If mu
        is supplied, it is not reestimated.

        Parameters
        ----------
        a : ndarray
            1d array
        mu : float or None, optional
            If the location mu is supplied then it is not reestimated.
            Default is None, which means that it is estimated.
        initscale : float or None, optional
            A first guess on scale.  If initscale is None then the standardized
            median absolute deviation of a is used.

        Notes
        -----
        `Huber` minimizes the function

        sum(psi((a[i]-mu)/scale)**2)

        as a function of (mu, scale), where

        psi(x) = np.clip(x, -self.c, self.c)
        """
        a = np.asarray(a)
        if mu is None:
            n = a.shape[0] - 1
            mu = np.median(a, axis=axis)
            est_mu = True
        else:
            n = a.shape[0]
            mu = mu
            est_mu = False

        if initscale is None:
            scale = mad(a, axis=axis)
        else:
            scale = initscale
        scale = tools.unsqueeze(scale, axis, a.shape)
        mu = tools.unsqueeze(mu, axis, a.shape)
        return self._estimate_both(a, scale, mu, axis, est_mu, n)

    def _estimate_both(self, a, scale, mu, axis, est_mu, n):
        """
        Estimate scale and location simultaneously with the following
        pseudo_loop:

        while not_converged:
            mu, scale = estimate_location(a, scale, mu), estimate_scale(a, scale, mu)

        where estimate_location is an M-estimator and estimate_scale implements
        the check used in Section 5.5 of Venables & Ripley
        """  # noqa:E501
        for _ in range(self.maxiter):
            # Estimate the mean along a given axis
            if est_mu:
                if self.norm is None:
                    # This is a one-step fixed-point estimator
                    # if self.norm == norms.HuberT
                    # It should be faster than using norms.HuberT
                    nmu = (
                        np.clip(
                            a, mu - self.c * scale, mu + self.c * scale
                        ).sum(axis)
                        / a.shape[axis]
                    )
                else:
                    nmu = norms.estimate_location(
                        a, scale, self.norm, axis, mu, self.maxiter, self.tol
                    )
            else:
                # Effectively, do nothing
                nmu = mu.squeeze()
            nmu = tools.unsqueeze(nmu, axis, a.shape)

            subset = np.less_equal(np.abs((a - mu) / scale), self.c)
            card = subset.sum(axis)

            scale_num = np.sum(subset * (a - nmu) ** 2, axis)
            scale_denom = n * self.gamma - (a.shape[axis] - card) * self.c ** 2
            nscale = np.sqrt(scale_num / scale_denom)
            nscale = tools.unsqueeze(nscale, axis, a.shape)

            test1 = np.all(
                np.less_equal(np.abs(scale - nscale), nscale * self.tol)
            )
            test2 = np.all(
                np.less_equal(np.abs(mu - nmu), nscale * self.tol)
            )
            if not (test1 and test2):
                mu = nmu
                scale = nscale
            else:
                return nmu.squeeze(), nscale.squeeze()
        raise ValueError(
            "joint estimation of location and scale failed "
            "to converge in %d iterations" % self.maxiter
        )


huber = Huber()


class HuberScale:
    r"""
    Huber's scaling for fitting robust linear models.

    Huber's scale is intended to be used as the scale estimate in the
    IRLS algorithm and is slightly different than the `Huber` class.

    Parameters
    ----------
    d : float, optional
        d is the tuning constant for Huber's scale.  Default is 2.5
    tol : float, optional
        The convergence tolerance
    maxiter : int, optiona
        The maximum number of iterations.  The default is 30.

    Methods
    -------
    call
        Return's Huber's scale computed as below

    Notes
    -----
    Huber's scale is the iterative solution to

    scale_(i+1)**2 = 1/(n*h)*sum(chi(r/sigma_i)*sigma_i**2

    where the Huber function is

    chi(x) = (x**2)/2       for \|x\| < d
    chi(x) = (d**2)/2       for \|x\| >= d

    and the Huber constant h = (n-p)/n*(d**2 + (1-d**2)*
    scipy.stats.norm.cdf(d) - .5 - d*sqrt(2*pi)*exp(-0.5*d**2)
    """

    def __init__(self, d=2.5, tol=1e-08, maxiter=30):
        self.d = d
        self.tol = tol
        self.maxiter = maxiter

    def __call__(self, df_resid, nobs, resid):
        h = (
            df_resid
            / nobs
            * (
                self.d ** 2
                + (1 - self.d ** 2) * Gaussian.cdf(self.d)
                - 0.5
                - self.d / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.d ** 2)
            )
        )
        s = mad(resid)

        def subset(x):
            return np.less(np.abs(resid / x), self.d)

        def chi(s):
            return subset(s) * (resid / s) ** 2 / 2 + (1 - subset(s)) * (
                self.d ** 2 / 2
            )

        scalehist = [np.inf, s]
        niter = 1
        while (
            np.abs(scalehist[niter - 1] - scalehist[niter]) > self.tol
            and niter < self.maxiter
        ):
            nscale = np.sqrt(
                1
                / (nobs * h)
                * np.sum(chi(scalehist[-1]))
                * scalehist[-1] ** 2
            )
            scalehist.append(nscale)
            niter += 1
            # TODO: raise on convergence failure?
        return scalehist[-1]


hubers_scale = HuberScale()
