"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions have not been formally tested.
"""

from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning


def durbin_watson(resids, axis=0):
    r"""
    Calculates the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.
    """
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw


def omni_normtest(resids, axis=0):
    """
    Omnibus test for normality

    Parameters
    ----------
    resid : array_like
    axis : int, optional
        Default is 0

    Returns
    -------
    Chi^2 score, two-tail probability
    """
    # TODO: change to exception in summary branch and catch in summary()
    #   behavior changed between scipy 0.9 and 0.10
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        from warnings import warn
        warn("omni_normtest is not valid with less than 8 observations; %i "
             "samples were given." % int(n), ValueWarning)
        return np.nan, np.nan

    return stats.normaltest(resids, axis=axis)


def jarque_bera(resids, axis=0):
    r"""
    The Jarque-Bera test of normality.

    Parameters
    ----------
    resids : array_like
        Data to test for normality. Usually regression model residuals that
        are mean 0.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    JB : {float, ndarray}
        The Jarque-Bera test statistic.
    JBpv : {float, ndarray}
        The pvalue of the test statistic.
    skew : {float, ndarray}
        Estimated skewness of the data.
    kurtosis : {float, ndarray}
        Estimated kurtosis of the data.

    Notes
    -----
    Each output returned has 1 dimension fewer than data

    The Jarque-Bera test statistic tests the null that the data is normally
    distributed against an alternative that the data follow some other
    distribution. The test statistic is based on two moments of the data,
    the skewness, and the kurtosis, and has an asymptotic :math:`\chi^2_2`
    distribution.

    The test statistic is defined

    .. math:: JB = n(S^2/6+(K-3)^2/24)

    where n is the number of data points, S is the sample skewness, and K is
    the sample kurtosis of the data.
    """
    resids = np.atleast_1d(np.asarray(resids, dtype=float))
    if resids.size < 2:
        raise ValueError("resids must contain at least 2 elements")
    # Calculate residual skewness and kurtosis
    skew = stats.skew(resids, axis=axis)
    kurtosis = 3 + stats.kurtosis(resids, axis=axis)

    # Calculate the Jarque-Bera test for normality
    n = resids.shape[axis]
    jb = (n / 6.) * (skew ** 2 + (1 / 4.) * (kurtosis - 3) ** 2)
    jb_pv = stats.chi2.sf(jb, 2)

    return jb, jb_pv, skew, kurtosis


def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the skewness measures are computed.  If `None`, the
        entire array is used.

    Returns
    -------
    sk1 : ndarray
          The standard skewness estimator.
    sk2 : ndarray
          Skewness estimator based on quartiles.
    sk3 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          absolute deviation.
    sk4 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          standard deviation.

    Notes
    -----
    The robust skewness measures are defined

    .. math::

        SK_{2}=\\frac{\\left(q_{.75}-q_{.5}\\right)
        -\\left(q_{.5}-q_{.25}\\right)}{q_{.75}-q_{.25}}

    .. math::

        SK_{3}=\\frac{\\mu-\\hat{q}_{0.5}}
        {\\hat{E}\\left[\\left|y-\\hat{\\mu}\\right|\\right]}

    .. math::

        SK_{4}=\\frac{\\mu-\\hat{q}_{0.5}}{\\hat{\\sigma}}

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """

    if axis is None:
        y = y.ravel()
        axis = 0

    y = np.sort(y, axis)

    q1, q2, q3 = np.percentile(y, [25.0, 50.0, 75.0], axis=axis)

    mu = y.mean(axis)
    shape = (y.size,)
    if axis is not None:
        shape = list(mu.shape)
        shape.insert(axis, 1)
        shape = tuple(shape)

    mu_b = np.reshape(mu, shape)
    q2_b = np.reshape(q2, shape)

    sigma = np.sqrt(np.mean(((y - mu_b)**2), axis))

    sk1 = stats.skew(y, axis=axis)
    sk2 = (q1 + q3 - 2.0 * q2) / (q3 - q1)
    sk3 = (mu - q2) / np.mean(abs(y - q2_b), axis=axis)
    sk4 = (mu - q2) / sigma

    return sk1, sk2, sk3, sk4


def _kr3(y, alpha=5.0, beta=50.0):
    """
    KR3 estimator from Kim & White

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.
    alpha : float, optional
        Lower cut-off for measuring expectation in tail.
    beta :  float, optional
        Lower cut-off for measuring expectation in center.

    Returns
    -------
    kr3 : float
        Robust kurtosis estimator based on standardized lower- and upper-tail
        expected values

    Notes
    -----
    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    perc = (alpha, 100.0 - alpha, beta, 100.0 - beta)
    lower_alpha, upper_alpha, lower_beta, upper_beta = np.percentile(y, perc)
    l_alpha = np.mean(y[y < lower_alpha])
    u_alpha = np.mean(y[y > upper_alpha])

    l_beta = np.mean(y[y < lower_beta])
    u_beta = np.mean(y[y > upper_beta])

    return (u_alpha - l_alpha) / (u_beta - l_beta)


def expected_robust_kurtosis(ab=(5.0, 50.0), dg=(2.5, 25.0)):
    """
    Calculates the expected value of the robust kurtosis measures in Kim and
    White assuming the data are normally distributed.

    Parameters
    ----------
    ab : iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure

    Returns
    -------
    ekr : ndarray, 4-element
        Contains the expected values of the 4 robust kurtosis measures

    Notes
    -----
    See `robust_kurtosis` for definitions of the robust kurtosis measures
    """

    alpha, beta = ab
    delta, gamma = dg
    expected_value = np.zeros(4)
    ppf = stats.norm.ppf
    pdf = stats.norm.pdf
    q1, q2, q3, q5, q6, q7 = ppf(np.array((1.0, 2.0, 3.0, 5.0, 6.0, 7.0)) / 8)
    expected_value[0] = 3

    expected_value[1] = ((q7 - q5) + (q3 - q1)) / (q6 - q2)

    q_alpha, q_beta = ppf(np.array((alpha / 100.0, beta / 100.0)))
    expected_value[2] = (2 * pdf(q_alpha) / alpha) / (2 * pdf(q_beta) / beta)

    q_delta, q_gamma = ppf(np.array((delta / 100.0, gamma / 100.0)))
    expected_value[3] = (-2.0 * q_delta) / (-2.0 * q_gamma)

    return expected_value


def robust_kurtosis(y, axis=0, ab=(5.0, 50.0), dg=(2.5, 25.0), excess=True):
    """
    Calculates the four kurtosis measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the kurtosis are computed.  If `None`, the
        entire array is used.
    a iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure
    excess : bool, optional
        If true (default), computed values are excess of those for a standard
        normal distribution.

    Returns
    -------
    kr1 : ndarray
          The standard kurtosis estimator.
    kr2 : ndarray
          Kurtosis estimator based on octiles.
    kr3 : ndarray
          Kurtosis estimators based on exceedance expectations.
    kr4 : ndarray
          Kurtosis measure based on the spread between high and low quantiles.

    Notes
    -----
    The robust kurtosis measures are defined

    .. math::

        KR_{2}=\\frac{\\left(\\hat{q}_{.875}-\\hat{q}_{.625}\\right)
        +\\left(\\hat{q}_{.375}-\\hat{q}_{.125}\\right)}
        {\\hat{q}_{.75}-\\hat{q}_{.25}}

    .. math::

        KR_{3}=\\frac{\\hat{E}\\left(y|y>\\hat{q}_{1-\\alpha}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\alpha}\\right)}
        {\\hat{E}\\left(y|y>\\hat{q}_{1-\\beta}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\beta}\\right)}

    .. math::

        KR_{4}=\\frac{\\hat{q}_{1-\\delta}-\\hat{q}_{\\delta}}
        {\\hat{q}_{1-\\gamma}-\\hat{q}_{\\gamma}}

    where :math:`\\hat{q}_{p}` is the estimated quantile at :math:`p`.

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    if (axis is None or
            (y.squeeze().ndim == 1 and y.ndim != 1)):
        y = y.ravel()
        axis = 0

    alpha, beta = ab
    delta, gamma = dg

    perc = (12.5, 25.0, 37.5, 62.5, 75.0, 87.5,
            delta, 100.0 - delta, gamma, 100.0 - gamma)
    e1, e2, e3, e5, e6, e7, fd, f1md, fg, f1mg = np.percentile(y, perc,
                                                               axis=axis)

    expected_value = (expected_robust_kurtosis(ab, dg)
                      if excess else np.zeros(4))

    kr1 = stats.kurtosis(y, axis, False) - expected_value[0]
    kr2 = ((e7 - e5) + (e3 - e1)) / (e6 - e2) - expected_value[1]
    if y.ndim == 1:
        kr3 = _kr3(y, alpha, beta)
    else:
        kr3 = np.apply_along_axis(_kr3, axis, y, alpha, beta)
    kr3 -= expected_value[2]
    kr4 = (f1md - fd) / (f1mg - fg) - expected_value[3]
    return kr1, kr2, kr3, kr4


def _medcouple_1d(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """

    # Parameter changes the algorithm to the slower for large n

    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    # GH5395
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
        # and 1 below the anti-diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)
        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements

    return np.median(h)


def medcouple(y, axis=0):
    """
    Calculate the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : {int, None}
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.

    Returns
    -------
    mc : ndarray
        The medcouple statistic with the same shape as `y`, with the specified
        axis removed.

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    y = np.asarray(y, dtype=np.double)  # GH 4243
    if axis is None:
        return _medcouple_1d(y.ravel())

    return np.apply_along_axis(_medcouple_1d, axis, y)
