# -*- coding: utf-8 -*-
""" Distance dependence measure and the dCov test.

Implementation of Székely et al. (2007) calculation of distance
dependence statistics, including the Distance covariance (dCov) test
for independence of random vectors of arbitrary length.

Author: Ron Itzikovitch

References
----------
.. Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
   "Measuring and testing dependence by correlation of distances".
   Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

"""
from collections import namedtuple
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

from statsmodels.tools.sm_exceptions import HypothesisTestWarning

DistDependStat = namedtuple(
    "DistDependStat",
    ["test_statistic", "distance_correlation",
     "distance_covariance", "dvar_x", "dvar_y", "S"],
)


def distance_covariance_test(x, y, B=None, method="auto"):
    r"""The Distance Covariance (dCov) test

    Apply the Distance Covariance (dCov) test of independence to `x` and `y`.
    This test was introduced in [1]_, and is based on the distance covariance
    statistic. The test is applicable to random vectors of arbitrary length
    (see the notes section for more details).

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int, optional, default=`None`
        The number of iterations to perform when evaluating the null
        distribution of the test statistic when the `emp` method is
        applied (see below). if `B` is `None` than as in [1]_ we set
        `B` to be ``B = 200 + 5000/n``, where `n` is the number of
        observations.
    method : {'auto', 'emp', 'asym'}, optional, default=auto
        The method by which to obtain the p-value for the test.

        - `auto` : Default method. The number of observations will be used to
          determine the method.
        - `emp` : Empirical evaluation of the p-value using permutations of
          the rows of `y` to obtain the null distribution.
        - `asym` : An asymptotic approximation of the distribution of the test
          statistic is used to find the p-value.

    Returns
    -------
    test_statistic : float
        The value of the test statistic used in the test.
    pval : float
        The p-value.
    chosen_method : str
        The method that was used to obtain the p-value. Mostly relevant when
        the function is called with `method='auto'`.

    Notes
    -----
    The test applies to random vectors of arbitrary dimensions, i.e., `x`
    can be a 1-D vector of observations for a single random variable while
    `y` can be a `k` by `n` 2-D array (where `k > 1`). In other words, it
    is also possible for `x` and `y` to both be 2-D arrays and have the
    same number of rows (observations) while differing in the number of
    columns.

    As noted in [1]_ the statistics are sensitive to all types of departures
    from independence, including nonlinear or nonmonotone dependence
    structure.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------
    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_covariance_test
    >>> data = np.random.rand(1000, 10)
    >>> x, y = data[:, :3], data[:, 3:]
    >>> x.shape
    (1000, 3)
    >>> y.shape
    (1000, 7)
    >>> distance_covariance_test(x, y)
    (1.0426404792714983, 0.2971148340813543, 'asym')
    # (test_statistic, pval, chosen_method)

    """
    x, y = _validate_and_tranform_x_and_y(x, y)

    n = x.shape[0]
    stats = distance_statistics(x, y)

    if method == "auto" and n <= 500 or method == "emp":
        chosen_method = "emp"
        test_statistic, pval = _empirical_pvalue(x, y, B, n, stats)

    elif method == "auto" and n > 500 or method == "asym":
        chosen_method = "asym"
        test_statistic, pval = _asymptotic_pvalue(stats)

    else:
        raise ValueError("Unknown 'method' parameter: {}".format(method))

    # In case we got an extreme p-value (0 or 1) when using the empirical
    # distribution of the test statistic under the null, we fall back
    # to the asymptotic approximation.
    if chosen_method == "emp" and pval in [0, 1]:
        msg = (
            f"p-value was {pval} when using the empirical method. "
            "The asymptotic approximation will be used instead"
        )
        warnings.warn(msg, HypothesisTestWarning)
        _, pval = _asymptotic_pvalue(stats)

    return test_statistic, pval, chosen_method


def _validate_and_tranform_x_and_y(x, y):
    r"""Ensure `x` and `y` have proper shape and transform/reshape them if
    required.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.

    Returns
    -------
    x : array_like, 1-D or 2-D
    y : array_like, 1-D or 2-D

    Raises
    ------
    ValueError
        If `x` and `y` have a different number of observations.

    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y must have the same number of observations (rows)."
        )

    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))

    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))

    return x, y


def _empirical_pvalue(x, y, B, n, stats):
    r"""Calculate the empirical p-value based on permutations of `y`'s rows

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int
        The number of iterations when evaluating the null distribution.
    n : Number of observations found in each of `x` and `y`.
    stats: namedtuple
        The result obtained from calling ``distance_statistics(x, y)``.

    Returns
    -------
    test_statistic : float
        The empirical test statistic.
    pval : float
        The empirical p-value.

    """
    B = int(B) if B else int(np.floor(200 + 5000 / n))
    empirical_dist = _get_test_statistic_distribution(x, y, B)
    pval = 1 - np.searchsorted(
        sorted(empirical_dist), stats.test_statistic
    ) / len(empirical_dist)
    test_statistic = stats.test_statistic

    return test_statistic, pval


def _asymptotic_pvalue(stats):
    r"""Calculate the p-value based on an approximation of the distribution of
    the test statistic under the null.

    Parameters
    ----------
    stats: namedtuple
        The result obtained from calling ``distance_statistics(x, y)``.

    Returns
    -------
    test_statistic : float
        The test statistic.
    pval : float
        The asymptotic p-value.

    """
    test_statistic = np.sqrt(stats.test_statistic / stats.S)
    pval = (1 - norm.cdf(test_statistic)) * 2

    return test_statistic, pval


def _get_test_statistic_distribution(x, y, B):
    r"""
    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int
        The number of iterations to perform when evaluating the null
        distribution.

    Returns
    -------
    emp_dist : array_like
        The empirical distribution of the test statistic.

    """
    y = y.copy()
    emp_dist = np.zeros(B)
    x_dist = squareform(pdist(x, "euclidean"))

    for i in range(B):
        np.random.shuffle(y)
        emp_dist[i] = distance_statistics(x, y, x_dist=x_dist).test_statistic

    return emp_dist


def distance_statistics(x, y, x_dist=None, y_dist=None):
    r"""Calculate various distance dependence statistics.

    Calculate several distance dependence statistics as described in [1]_.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    x_dist : array_like, 2-D, optional
        A square 2-D array_like object whose values are the euclidean
        distances between `x`'s rows.
    y_dist : array_like, 2-D, optional
        A square 2-D array_like object whose values are the euclidean
        distances between `y`'s rows.

    Returns
    -------
    namedtuple
        A named tuple of distance dependence statistics (DistDependStat) with
        the following values:

        - test_statistic : float - The "basic" test statistic (i.e., the one
          used when the `emp` method is chosen when calling
          ``distance_covariance_test()``
        - distance_correlation : float - The distance correlation
          between `x` and `y`.
        - distance_covariance : float - The distance covariance of
          `x` and `y`.
        - dvar_x : float - The distance variance of `x`.
        - dvar_y : float - The distance variance of `y`.
        - S : float - The mean of the euclidean distances in `x` multiplied
          by those of `y`. Mostly used internally.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_statistics
    >>> distance_statistics(np.random.random(1000), np.random.random(1000))
    DistDependStat(test_statistic=0.07948284320205831,
    distance_correlation=0.04269511890990793,
    distance_covariance=0.008915315092696293,
    dvar_x=0.20719027438266704, dvar_y=0.21044934264957588,
    S=0.10892061635588891)

    """
    x, y = _validate_and_tranform_x_and_y(x, y)

    n = x.shape[0]

    a = x_dist if x_dist is not None else squareform(pdist(x, "euclidean"))
    b = y_dist if y_dist is not None else squareform(pdist(y, "euclidean"))

    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()

    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean

    S = a_mean * b_mean
    dcov = np.sqrt(np.multiply(A, B).mean())
    dvar_x = np.sqrt(np.multiply(A, A).mean())
    dvar_y = np.sqrt(np.multiply(B, B).mean())
    dcor = dcov / np.sqrt(dvar_x * dvar_y)

    test_statistic = n * dcov ** 2

    return DistDependStat(
        test_statistic=test_statistic,
        distance_correlation=dcor,
        distance_covariance=dcov,
        dvar_x=dvar_x,
        dvar_y=dvar_y,
        S=S,
    )


def distance_covariance(x, y):
    r"""Distance covariance.

    Calculate the empirical distance covariance as described in [1]_.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.

    Returns
    -------
    float
        The empirical distance covariance between `x` and `y`.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_covariance
    >>> distance_covariance(np.random.random(1000), np.random.random(1000))
    0.007575063951951362

    """
    return distance_statistics(x, y).distance_covariance


def distance_variance(x):
    r"""Distance variance.

    Calculate the empirical distance variance as described in [1]_.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.

    Returns
    -------
    float
        The empirical distance variance of `x`.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_variance
    >>> distance_variance(np.random.random(1000))
    0.21732609190659702

    """
    return distance_covariance(x, x)


def distance_correlation(x, y):
    r"""Distance correlation.

    Calculate the empirical distance correlation as described in [1]_.
    This statistic is analogous to product-moment correlation and describes
    the dependence between `x` and `y`, which are random vectors of
    arbitrary length. The statistics' values range between 0 (implies
    independence) and 1 (implies complete dependence).

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.

    Returns
    -------
    float
        The empirical distance correlation between `x` and `y`.

    References
    ----------
    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
       "Measuring and testing dependence by correlation of distances".
       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

    Examples
    --------

    >>> from statsmodels.stats.dist_dependence_measures import
    ... distance_correlation
    >>> distance_correlation(np.random.random(1000), np.random.random(1000))
    0.04060497840149489

    """
    return distance_statistics(x, y).distance_correlation
