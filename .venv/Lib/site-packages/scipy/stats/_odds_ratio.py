import numpy as np

from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval


def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    # table must be a 2x2 numpy array.
    if table[1, 0] > 0 and table[0, 1] > 0:
        oddsratio = table[0, 0] * table[1, 1] / (table[1, 0] * table[0, 1])
    elif table[0, 0] == 0 or table[1, 1] == 0:
        oddsratio = np.nan
    else:
        oddsratio = np.inf
    return oddsratio


def _solve(func):
    """
    Solve func(nc) = 0.  func must be an increasing function.
    """
    # We could just as well call the variable `x` instead of `nc`, but we
    # always call this function with functions for which nc (the noncentrality
    # parameter) is the variable for which we are solving.
    nc = 1.0
    value = func(nc)
    if value == 0:
        return nc

    # Multiplicative factor by which to increase or decrease nc when
    # searching for a bracketing interval.
    factor = 2.0
    # Find a bracketing interval.
    if value > 0:
        nc /= factor
        while func(nc) > 0:
            nc /= factor
        lo = nc
        hi = factor*nc
    else:
        nc *= factor
        while func(nc) < 0:
            nc *= factor
        lo = nc/factor
        hi = nc

    # lo and hi bracket the solution for nc.
    nc = brentq(func, lo, hi, xtol=1e-13)
    return nc


def _nc_hypergeom_mean_inverse(x, M, n, N):
    """
    For the given noncentral hypergeometric parameters x, M, n,and N
    (table[0,0], total, row 0 sum and column 0 sum, resp., of a 2x2
    contingency table), find the noncentrality parameter of Fisher's
    noncentral hypergeometric distribution whose mean is x.
    """
    nc = _solve(lambda nc: nchypergeom_fisher.mean(M, n, N, nc) - x)
    return nc


def _hypergeom_params_from_table(table):
    # The notation M, n and N is consistent with stats.hypergeom and
    # stats.nchypergeom_fisher.
    x = table[0, 0]
    M = table.sum()
    n = table[0].sum()
    N = table[:, 0].sum()
    return x, M, n, N


def _ci_upper(table, alpha):
    """
    Compute the upper end of the confidence interval.
    """
    if _sample_odds_ratio(table) == np.inf:
        return np.inf

    x, M, n, N = _hypergeom_params_from_table(table)

    # nchypergeom_fisher.cdf is a decreasing function of nc, so we negate
    # it in the lambda expression.
    nc = _solve(lambda nc: -nchypergeom_fisher.cdf(x, M, n, N, nc) + alpha)
    return nc


def _ci_lower(table, alpha):
    """
    Compute the lower end of the confidence interval.
    """
    if _sample_odds_ratio(table) == 0:
        return 0

    x, M, n, N = _hypergeom_params_from_table(table)

    nc = _solve(lambda nc: nchypergeom_fisher.sf(x - 1, M, n, N, nc) - alpha)
    return nc


def _conditional_oddsratio(table):
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    x, M, n, N = _hypergeom_params_from_table(table)
    # Get the bounds of the support.  The support of the noncentral
    # hypergeometric distribution with parameters M, n, and N is the same
    # for all values of the noncentrality parameter, so we can use 1 here.
    lo, hi = nchypergeom_fisher.support(M, n, N, 1)

    # Check if x is at one of the extremes of the support.  If so, we know
    # the odds ratio is either 0 or inf.
    if x == lo:
        # x is at the low end of the support.
        return 0
    if x == hi:
        # x is at the high end of the support.
        return np.inf

    nc = _nc_hypergeom_mean_inverse(x, M, n, N)
    return nc


def _conditional_oddsratio_ci(table, confidence_level=0.95,
                              alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.
    """
    if alternative == 'two-sided':
        alpha = 0.5*(1 - confidence_level)
        lower = _ci_lower(table, alpha)
        upper = _ci_upper(table, alpha)
    elif alternative == 'less':
        lower = 0.0
        upper = _ci_upper(table, 1 - confidence_level)
    else:
        # alternative == 'greater'
        lower = _ci_lower(table, 1 - confidence_level)
        upper = np.inf

    return lower, upper


def _sample_odds_ratio_ci(table, confidence_level=0.95,
                          alternative='two-sided'):
    oddsratio = _sample_odds_ratio(table)
    log_or = np.log(oddsratio)
    se = np.sqrt((1/table).sum())
    if alternative == 'less':
        z = ndtri(confidence_level)
        loglow = -np.inf
        loghigh = log_or + z*se
    elif alternative == 'greater':
        z = ndtri(confidence_level)
        loglow = log_or - z*se
        loghigh = np.inf
    else:
        # alternative is 'two-sided'
        z = ndtri(0.5*confidence_level + 0.5)
        loglow = log_or - z*se
        loghigh = log_or + z*se

    return np.exp(loglow), np.exp(loghigh)


class OddsRatioResult:
    """
    Result of `scipy.stats.contingency.odds_ratio`.  See the
    docstring for `odds_ratio` for more details.

    Attributes
    ----------
    statistic : float
        The computed odds ratio.

        * If `kind` is ``'sample'``, this is sample (or unconditional)
          estimate, given by
          ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
        * If `kind` is ``'conditional'``, this is the conditional
          maximum likelihood estimate for the odds ratio. It is
          the noncentrality parameter of Fisher's noncentral
          hypergeometric distribution with the same hypergeometric
          parameters as `table` and whose mean is ``table[0, 0]``.

    Methods
    -------
    confidence_interval :
        Confidence interval for the odds ratio.
    """

    def __init__(self, _table, _kind, statistic):
        # for now, no need to make _table and _kind public, since this sort of
        # information is returned in very few `scipy.stats` results
        self._table = _table
        self._kind = _kind
        self.statistic = statistic

    def __repr__(self):
        return f"OddsRatioResult(statistic={self.statistic})"

    def confidence_interval(self, confidence_level=0.95,
                            alternative='two-sided'):
        """
        Confidence interval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.
            Default is 0.95 (meaning 95%).

        alternative : {'two-sided', 'less', 'greater'}, optional
            The alternative hypothesis of the hypothesis test to which the
            confidence interval corresponds. That is, suppose the null
            hypothesis is that the true odds ratio equals ``OR`` and the
            confidence interval is ``(low, high)``. Then the following options
            for `alternative` are available (default is 'two-sided'):

            * 'two-sided': the true odds ratio is not equal to ``OR``. There
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``high < OR`` or ``low > OR``.
            * 'less': the true odds ratio is less than ``OR``. The ``low`` end
              of the confidence interval is 0, and there is evidence against
              the null hypothesis at  the chosen `confidence_level` if
              ``high < OR``.
            * 'greater': the true odds ratio is greater than ``OR``.  The
              ``high`` end of the confidence interval is ``np.inf``, and there
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``low > OR``.

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When `kind` is ``'conditional'``, the limits of the confidence
        interval are the conditional "exact confidence limits" as described
        by Fisher [1]_. The conditional odds ratio and confidence interval are
        also discussed in Section 4.1.2 of the text by Sahai and Khurshid [2]_.

        When `kind` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [2]_, section 3.1.3.2,
        or [3]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [3] Alan Agresti, An Introduction to Categorical Data Analysis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
        if alternative not in ['two-sided', 'less', 'greater']:
            raise ValueError("`alternative` must be 'two-sided', 'less' or "
                             "'greater'.")

        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        if self._kind == 'conditional':
            ci = self._conditional_odds_ratio_ci(confidence_level, alternative)
        else:
            ci = self._sample_odds_ratio_ci(confidence_level, alternative)
        return ci

    def _conditional_odds_ratio_ci(self, confidence_level=0.95,
                                   alternative='two-sided'):
        """
        Confidence interval for the conditional odds ratio.
        """

        table = self._table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _conditional_oddsratio_ci(table,
                                           confidence_level=confidence_level,
                                           alternative=alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])

    def _sample_odds_ratio_ci(self, confidence_level=0.95,
                              alternative='two-sided'):
        """
        Confidence interval for the sample odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        table = self._table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _sample_odds_ratio_ci(table,
                                       confidence_level=confidence_level,
                                       alternative=alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])


def odds_ratio(table, *, kind='conditional'):
    r"""
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.

    Returns
    -------
    result : `~scipy.stats._result_classes.OddsRatioResult` instance
        The returned object has two computed attributes:

        statistic : float
            * If `kind` is ``'sample'``, this is sample (or unconditional)
              estimate, given by
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.

        The object has the method `confidence_interval` that computes
        the confidence interval of the odds ratio.

    See Also
    --------
    scipy.stats.fisher_exact
    relative_risk

    Notes
    -----
    The conditional odds ratio was discussed by Fisher (see "Example 1"
    of [1]_).  Texts that cover the odds ratio include [2]_ and [3]_.

    .. versionadded:: 1.10.0

    References
    ----------
    .. [1] R. A. Fisher (1935), The logic of inductive inference,
           Journal of the Royal Statistical Society, Vol. 98, No. 1,
           pp. 39-82.
    .. [2] Breslow NE, Day NE (1980). Statistical methods in cancer research.
           Volume I - The analysis of case-control studies. IARC Sci Publ.
           (32):5-338. PMID: 7216345. (See section 4.2.)
    .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.
    .. [4] Berger, Jeffrey S. et al. "Aspirin for the Primary Prevention of
           Cardiovascular Events in Women and Men: A Sex-Specific
           Meta-analysis of Randomized Controlled Trials."
           JAMA, 295(3):306-313, :doi:`10.1001/jama.295.3.306`, 2006.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classified as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of a case occurring in the exposed group,
    and ``b/d`` as the odds of a case occurring in the unexposed group.
    The sample odds ratio is the ratio of these odds.  If the odds ratio
    is greater than 1, it suggests that there is a positive association
    between being exposed and being a case.

    Interchanging the rows or columns of the contingency table inverts
    the odds ratio, so it is import to understand the meaning of labels
    given to the rows and columns of the table when interpreting the
    odds ratio.

    In [4]_, the use of aspirin to prevent cardiovascular events in women
    and men was investigated. The study notably concluded:

        ...aspirin therapy reduced the risk of a composite of
        cardiovascular events due to its effect on reducing the risk of
        ischemic stroke in women [...]

    The article lists studies of various cardiovascular events. Let's
    focus on the ischemic stoke in women.

    The following table summarizes the results of the experiment in which
    participants took aspirin or a placebo on a regular basis for several
    years. Cases of ischemic stroke were recorded::

                          Aspirin   Control/Placebo
        Ischemic stroke     176           230
        No stroke         21035         21018

    The question we ask is "Is there evidence that the aspirin reduces the
    risk of ischemic stroke?"

    Compute the odds ratio:

    >>> from scipy.stats.contingency import odds_ratio
    >>> res = odds_ratio([[176, 230], [21035, 21018]])
    >>> res.statistic
    0.7646037659999126

    For this sample, the odds of getting an ischemic stroke for those who have
    been taking aspirin are 0.76 times that of those
    who have received the placebo.

    To make statistical inferences about the population under study,
    we can compute the 95% confidence interval for the odds ratio:

    >>> res.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=0.6241234078749812, high=0.9354102892100372)

    The 95% confidence interval for the conditional odds ratio is
    approximately (0.62, 0.94).

    The fact that the entire 95% confidence interval falls below 1 supports
    the authors' conclusion that the aspirin was associated with a
    statistically significant reduction in ischemic stroke.
    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("`kind` must be 'conditional' or 'sample'.")

    c = np.asarray(table)

    if c.shape != (2, 2):
        raise ValueError(f"Invalid shape {c.shape}. The input `table` must be "
                         "of shape (2, 2).")

    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("`table` must be an array of integers, but got "
                         f"type {c.dtype}")
    c = c.astype(np.int64)

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is NaN and
        # the odds ratio is NaN.
        result = OddsRatioResult(_table=c, _kind=kind, statistic=np.nan)
        return result

    if kind == 'sample':
        oddsratio = _sample_odds_ratio(c)
    else:  # kind is 'conditional'
        oddsratio = _conditional_oddsratio(c)

    result = OddsRatioResult(_table=c, _kind=kind, statistic=oddsratio)
    return result
