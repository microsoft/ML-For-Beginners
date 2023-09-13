from __future__ import annotations

import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect

from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning

__all__ = ['bootstrap', 'monte_carlo_test', 'permutation_test']


def _vectorize_statistic(statistic):
    """Vectorize an n-sample statistic"""
    # This is a little cleaner than np.nditer at the expense of some data
    # copying: concatenate samples together, then use np.apply_along_axis
    def stat_nd(*data, axis=0):
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)

        # move working axis to position 0 so that new dimensions in the output
        # of `statistic` are _prepended_. ("This axis is removed, and replaced
        # with new dimensions...")
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            data = np.split(z, split_indices)
            return statistic(*data)

        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd


def _jackknife_resample(sample, batch=None):
    """Jackknife resample the sample. Only one-sample stats for now."""
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end are the observations to remove
        batch_actual = min(batch_nominal, n-k)

        # jackknife - each row leaves out one observation
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k+batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n-1))

        resamples = sample[..., i]
        yield resamples


def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """Bootstrap resample the sample."""
    n = sample.shape[-1]

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(random_state, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples


def _percentile_of_score(a, score, axis):
    """Vectorized, simplified `scipy.stats.percentileofscore`.
    Uses logic of the 'mean' value of percentileofscore's kind parameter.

    Unlike `stats.percentileofscore`, the percentile returned is a fraction
    in [0, 1].
    """
    B = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * B)


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(DegenerateDataWarning(msg))
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""
    # closely follows [1] 14.3 and 15.4 (Eq. 15.36)

    # calculate z0_hat
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    for j, sample in enumerate(data):
        # _jackknife_resample will add an axis prior to the last axis that
        # corresponds with the different jackknife resamples. Do the same for
        # each sample of the data to ensure broadcastability. We need to
        # create a copy of the list containing the samples anyway, so do this
        # in the loop to simplify the code. This is not the bottleneck...
        samples = [np.expand_dims(sample, -2) for sample in data]
        theta_hat_i = []
        for jackknife_sample in _jackknife_resample(sample, batch):
            samples[j] = jackknife_sample
            broadcasted = _broadcast_arrays(samples, axis=-1)
            theta_hat_i.append(statistic(*broadcasted, axis=-1))
        theta_hat_ji.append(theta_hat_i)

    theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1)
                    for theta_hat_i in theta_hat_ji]

    n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]

    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                       for theta_hat_i in theta_hat_ji]

    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
            for theta_hat_dot, theta_hat_i, n
            in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

    nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2, a_hat  # return a_hat for testing


def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level,
                  alternative, n_resamples, batch, method, bootstrap_result,
                  random_state):
    """Input validation and standardization for `bootstrap`."""

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError("`data` must be a sequence of samples.")

    if n_samples == 0:
        raise ValueError("`data` must contain at least one sample.")

    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis_int] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    if paired not in {True, False}:
        raise ValueError("`paired` must be `True` or `False`.")

    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = ("When `paired is True`, all samples must have the "
                           "same length along `axis`")
                raise ValueError(message)

        # to generate the bootstrap distribution for paired-sample statistics,
        # resample the indices of the observations
        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)

        data_iv = [np.arange(n)]

    confidence_level_float = float(confidence_level)

    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be one of {alternatives}")

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f"`method` must be in {methods}")

    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    if (bootstrap_result is not None
            and not hasattr(bootstrap_result, "bootstrap_distribution")):
        raise ValueError(message)

    message = ("Either `bootstrap_result.bootstrap_distribution.size` or "
               "`n_resamples` must be positive.")
    if ((not bootstrap_result or
         not bootstrap_result.bootstrap_distribution.size)
            and n_resamples_int == 0):
        raise ValueError(message)

    random_state = check_random_state(random_state)

    return (data_iv, statistic, vectorized, paired, axis_int,
            confidence_level_float, alternative, n_resamples_int, batch_iv,
            method, bootstrap_result, random_state)


@dataclass
class BootstrapResult:
    """Result object returned by `scipy.stats.bootstrap`.

    Attributes
    ----------
    confidence_interval : ConfidenceInterval
        The bootstrap confidence interval as an instance of
        `collections.namedtuple` with attributes `low` and `high`.
    bootstrap_distribution : ndarray
        The bootstrap distribution, that is, the value of `statistic` for
        each resample. The last dimension corresponds with the resamples
        (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
    standard_error : float or ndarray
        The bootstrap standard error, that is, the sample standard
        deviation of the bootstrap distribution.

    """
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray


def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              alternative='two-sided', method='BCa', bootstrap_result=None,
              random_state=None):
    r"""
    Compute a two-sided bootstrap confidence interval of a statistic.

    When `method` is ``'percentile'`` and `alternative` is ``'two-sided'``,
    a bootstrap confidence interval is computed according to the following
    procedure.

    1. Resample the data: for each sample in `data` and for each of
       `n_resamples`, take a random sample of the original sample
       (with replacement) of the same size as the original sample.

    2. Compute the bootstrap distribution of the statistic: for each set of
       resamples, compute the test statistic.

    3. Determine the confidence interval: find the interval of the bootstrap
       distribution that is

       - symmetric about the median and
       - contains `confidence_level` of the resampled statistic values.

    While the ``'percentile'`` method is the most intuitive, it is rarely
    used in practice. Two more common methods are available, ``'basic'``
    ('reverse percentile') and ``'BCa'`` ('bias-corrected and accelerated');
    they differ in how step 3 is performed.

    If the samples in `data` are  taken at random from their respective
    distributions :math:`n` times, the confidence interval returned by
    `bootstrap` will contain the true value of the statistic for those
    distributions approximately `confidence_level`:math:`\, \times \, n` times.

    Parameters
    ----------
    data : sequence of array-like
         Each element of data is a sample from an underlying distribution.
    statistic : callable
        Statistic for which the confidence interval is to be calculated.
        `statistic` must be a callable that accepts ``len(data)`` samples
        as separate arguments and returns the resulting statistic.
        If `vectorized` is set ``True``,
        `statistic` must also accept a keyword argument `axis` and be
        vectorized to compute the statistic along the provided `axis`.
    n_resamples : int, default: ``9999``
        The number of resamples performed to form the bootstrap distribution
        of the statistic.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        `statistic`. Memory usage is O(`batch`*``n``), where ``n`` is the
        sample size. Default is ``None``, in which case ``batch = n_resamples``
        (or ``batch = max(n_resamples, n)`` for ``method='BCa'``).
    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed an ND sample array. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of
        a vectorized statistic typically reduces computation time.
    paired : bool, default: ``False``
        Whether the statistic treats corresponding elements of the samples
        in `data` as paired.
    axis : int, default: ``0``
        The axis of the samples in `data` along which the `statistic` is
        calculated.
    confidence_level : float, default: ``0.95``
        The confidence level of the confidence interval.
    alternative : {'two-sided', 'less', 'greater'}, default: ``'two-sided'``
        Choose ``'two-sided'`` (default) for a two-sided confidence interval,
        ``'less'`` for a one-sided confidence interval with the lower bound
        at ``-np.inf``, and ``'greater'`` for a one-sided confidence interval
        with the upper bound at ``np.inf``. The other bound of the one-sided
        confidence intervals is the same as that of a two-sided confidence
        interval with `confidence_level` twice as far from 1.0; e.g. the upper
        bound of a 95% ``'less'``  confidence interval is the same as the upper
        bound of a 90% ``'two-sided'`` confidence interval.
    method : {'percentile', 'basic', 'bca'}, default: ``'BCa'``
        Whether to return the 'percentile' bootstrap confidence interval
        (``'percentile'``), the 'basic' (AKA 'reverse') bootstrap confidence
        interval (``'basic'``), or the bias-corrected and accelerated bootstrap
        confidence interval (``'BCa'``).
    bootstrap_result : BootstrapResult, optional
        Provide the result object returned by a previous call to `bootstrap`
        to include the previous bootstrap distribution in the new bootstrap
        distribution. This can be used, for example, to change
        `confidence_level`, change `method`, or see the effect of performing
        additional resampling without repeating computations.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Returns
    -------
    res : BootstrapResult
        An object with attributes:

        confidence_interval : ConfidenceInterval
            The bootstrap confidence interval as an instance of
            `collections.namedtuple` with attributes `low` and `high`.
        bootstrap_distribution : ndarray
            The bootstrap distribution, that is, the value of `statistic` for
            each resample. The last dimension corresponds with the resamples
            (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
        standard_error : float or ndarray
            The bootstrap standard error, that is, the sample standard
            deviation of the bootstrap distribution.

    Warns
    -----
    `~scipy.stats.DegenerateDataWarning`
        Generated when ``method='BCa'`` and the bootstrap distribution is
        degenerate (e.g. all elements are identical).

    Notes
    -----
    Elements of the confidence interval may be NaN for ``method='BCa'`` if
    the bootstrap distribution is degenerate (e.g. all elements are identical).
    In this case, consider using another `method` or inspecting `data` for
    indications that other analysis may be more appropriate (e.g. all
    observations are identical).

    References
    ----------
    .. [1] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap,
       Chapman & Hall/CRC, Boca Raton, FL, USA (1993)
    .. [2] Nathaniel E. Helwig, "Bootstrap Confidence Intervals",
       http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
    .. [3] Bootstrapping (statistics), Wikipedia,
       https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    Examples
    --------
    Suppose we have sampled data from an unknown distribution.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> from scipy.stats import norm
    >>> dist = norm(loc=2, scale=4)  # our "unknown" distribution
    >>> data = dist.rvs(size=100, random_state=rng)

    We are interested in the standard deviation of the distribution.

    >>> std_true = dist.std()      # the true value of the statistic
    >>> print(std_true)
    4.0
    >>> std_sample = np.std(data)  # the sample statistic
    >>> print(std_sample)
    3.9460644295563863

    The bootstrap is used to approximate the variability we would expect if we
    were to repeatedly sample from the unknown distribution and calculate the
    statistic of the sample each time. It does this by repeatedly resampling
    values *from the original sample* with replacement and calculating the
    statistic of each resample. This results in a "bootstrap distribution" of
    the statistic.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import bootstrap
    >>> data = (data,)  # samples must be in a sequence
    >>> res = bootstrap(data, np.std, confidence_level=0.9,
    ...                 random_state=rng)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25)
    >>> ax.set_title('Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('frequency')
    >>> plt.show()

    The standard error quantifies this variability. It is calculated as the
    standard deviation of the bootstrap distribution.

    >>> res.standard_error
    0.24427002125829136
    >>> res.standard_error == np.std(res.bootstrap_distribution, ddof=1)
    True

    The bootstrap distribution of the statistic is often approximately normal
    with scale equal to the standard error.

    >>> x = np.linspace(3, 5)
    >>> pdf = norm.pdf(x, loc=std_sample, scale=res.standard_error)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25, density=True)
    >>> ax.plot(x, pdf)
    >>> ax.set_title('Normal Approximation of the Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('pdf')
    >>> plt.show()

    This suggests that we could construct a 90% confidence interval on the
    statistic based on quantiles of this normal distribution.

    >>> norm.interval(0.9, loc=std_sample, scale=res.standard_error)
    (3.5442759991341726, 4.3478528599786)

    Due to central limit theorem, this normal approximation is accurate for a
    variety of statistics and distributions underlying the samples; however,
    the approximation is not reliable in all cases. Because `bootstrap` is
    designed to work with arbitrary underlying distributions and statistics,
    it uses more advanced techniques to generate an accurate confidence
    interval.

    >>> print(res.confidence_interval)
    ConfidenceInterval(low=3.57655333533867, high=4.382043696342881)

    If we sample from the original distribution 1000 times and form a bootstrap
    confidence interval for each sample, the confidence interval
    contains the true value of the statistic approximately 90% of the time.

    >>> n_trials = 1000
    >>> ci_contains_true_std = 0
    >>> for i in range(n_trials):
    ...    data = (dist.rvs(size=100, random_state=rng),)
    ...    ci = bootstrap(data, np.std, confidence_level=0.9, n_resamples=1000,
    ...                   random_state=rng).confidence_interval
    ...    if ci[0] < std_true < ci[1]:
    ...        ci_contains_true_std += 1
    >>> print(ci_contains_true_std)
    875

    Rather than writing a loop, we can also determine the confidence intervals
    for all 1000 samples at once.

    >>> data = (dist.rvs(size=(n_trials, 100), random_state=rng),)
    >>> res = bootstrap(data, np.std, axis=-1, confidence_level=0.9,
    ...                 n_resamples=1000, random_state=rng)
    >>> ci_l, ci_u = res.confidence_interval

    Here, `ci_l` and `ci_u` contain the confidence interval for each of the
    ``n_trials = 1000`` samples.

    >>> print(ci_l[995:])
    [3.77729695 3.75090233 3.45829131 3.34078217 3.48072829]
    >>> print(ci_u[995:])
    [4.88316666 4.86924034 4.32032996 4.2822427  4.59360598]

    And again, approximately 90% contain the true value, ``std_true = 4``.

    >>> print(np.sum((ci_l < std_true) & (std_true < ci_u)))
    900

    `bootstrap` can also be used to estimate confidence intervals of
    multi-sample statistics, including those calculated by hypothesis
    tests. `scipy.stats.mood` perform's Mood's test for equal scale parameters,
    and it returns two outputs: a statistic, and a p-value. To get a
    confidence interval for the test statistic, we first wrap
    `scipy.stats.mood` in a function that accepts two sample arguments,
    accepts an `axis` keyword argument, and returns only the statistic.

    >>> from scipy.stats import mood
    >>> def my_statistic(sample1, sample2, axis):
    ...     statistic, _ = mood(sample1, sample2, axis=-1)
    ...     return statistic

    Here, we use the 'percentile' method with the default 95% confidence level.

    >>> sample1 = norm.rvs(scale=1, size=100, random_state=rng)
    >>> sample2 = norm.rvs(scale=2, size=100, random_state=rng)
    >>> data = (sample1, sample2)
    >>> res = bootstrap(data, my_statistic, method='basic', random_state=rng)
    >>> print(mood(sample1, sample2)[0])  # element 0 is the statistic
    -5.521109549096542
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=-7.255994487314675, high=-4.016202624747605)

    The bootstrap estimate of the standard error is also available.

    >>> print(res.standard_error)
    0.8344963846318795

    Paired-sample statistics work, too. For example, consider the Pearson
    correlation coefficient.

    >>> from scipy.stats import pearsonr
    >>> n = 100
    >>> x = np.linspace(0, 10, n)
    >>> y = x + rng.uniform(size=n)
    >>> print(pearsonr(x, y)[0])  # element 0 is the statistic
    0.9962357936065914

    We wrap `pearsonr` so that it returns only the statistic.

    >>> def my_statistic(x, y):
    ...     return pearsonr(x, y)[0]

    We call `bootstrap` using ``paired=True``.
    Also, since ``my_statistic`` isn't vectorized to calculate the statistic
    along a given axis, we pass in ``vectorized=False``.

    >>> res = bootstrap((x, y), my_statistic, vectorized=False, paired=True,
    ...                 random_state=rng)
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=0.9950085825848624, high=0.9971212407917498)

    The result object can be passed back into `bootstrap` to perform additional
    resampling:

    >>> len(res.bootstrap_distribution)
    9999
    >>> res = bootstrap((x, y), my_statistic, vectorized=False, paired=True,
    ...                 n_resamples=1001, random_state=rng,
    ...                 bootstrap_result=res)
    >>> len(res.bootstrap_distribution)
    11000

    or to change the confidence interval options:

    >>> res2 = bootstrap((x, y), my_statistic, vectorized=False, paired=True,
    ...                  n_resamples=0, random_state=rng, bootstrap_result=res,
    ...                  method='percentile', confidence_level=0.9)
    >>> np.testing.assert_equal(res2.bootstrap_distribution,
    ...                         res.bootstrap_distribution)
    >>> res.confidence_interval
    ConfidenceInterval(low=0.9950035351407804, high=0.9971170323404578)

    without repeating computation of the original bootstrap distribution.

    """
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, alternative, n_resamples, batch,
                         method, bootstrap_result, random_state)
    (data, statistic, vectorized, paired, axis, confidence_level,
     alternative, n_resamples, batch, method, bootstrap_result,
     random_state) = args

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1

    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual,
                                           random_state=random_state)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))


def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples,
                         batch, alternative, axis):
    """Input validation for `monte_carlo_test`."""

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` must be callable or sequence of callables.")

    if not len(rvs) == len(data):
        message = "If `rvs` is a sequence, `len(rvs)` must equal `len(data)`."
        raise ValueError(message)

    if not callable(statistic):
        raise TypeError("`statistic` must be callable.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic_vectorized = _vectorize_statistic(statistic)
    else:
        statistic_vectorized = statistic

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int)


@dataclass
class MonteCarloTestResult:
    """Result object returned by `scipy.stats.monte_carlo_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the sample.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


@_rename_parameter('sample', 'data')
def monte_carlo_test(data, rvs, statistic, *, vectorized=None,
                     n_resamples=9999, batch=None, alternative="two-sided",
                     axis=0):
    r"""Perform a Monte Carlo hypothesis test.

    `data` contains a sample or a sequence of one or more samples. `rvs`
    specifies the distribution(s) of the sample(s) in `data` under the null
    hypothesis. The value of `statistic` for the given `data` is compared
    against a Monte Carlo null distribution: the value of the statistic for
    each of `n_resamples` sets of samples generated using `rvs`. This gives
    the p-value, the probability of observing such an extreme value of the
    test statistic under the null hypothesis.

    Parameters
    ----------
    data : array-like or sequence of array-like
        An array or sequence of arrays of observations.
    rvs : callable or tuple of callables
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples in
        `data`, i.e. ``len(rvs) == len(data)``. If `rvs` is a single callable,
        `data` is treated as a single sample.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts a sample
        (e.g. ``statistic(sample)``) or ``len(rvs)`` separate samples (e.g.
        ``statistic(samples1, sample2)`` if `rvs` contains two callables and
        `data` contains two samples) and returns the resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the samples in `data`.
    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed ND sample arrays. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of
        a vectorized statistic typically reduces computation time.
    n_resamples : int, default: 9999
        Number of samples drawn from each of the callables of `rvs`.
        Equivalently, the number statistic values under the null hypothesis
        used as the Monte Carlo null distribution.
    batch : int, optional
        The number of Monte Carlo samples to process in each call to
        `statistic`. Memory usage is O(`batch`*``sample.size[axis]``). Default
        is ``None``, in which case `batch` equals `n_resamples`.
    alternative : {'two-sided', 'less', 'greater'}
        The alternative hypothesis for which the p-value is calculated.
        For each alternative, the p-value is defined as follows.

        - ``'greater'`` : the percentage of the null distribution that is
          greater than or equal to the observed value of the test statistic.
        - ``'less'`` : the percentage of the null distribution that is
          less than or equal to the observed value of the test statistic.
        - ``'two-sided'`` : twice the smaller of the p-values above.

    axis : int, default: 0
        The axis of `data` (or each sample within `data`) over which to
        calculate the statistic.

    Returns
    -------
    res : MonteCarloTestResult
        An object with attributes:

        statistic : float or ndarray
            The test statistic of the observed `data`.
        pvalue : float or ndarray
            The p-value for the given alternative.
        null_distribution : ndarray
            The values of the test statistic generated under the null
            hypothesis.

    References
    ----------

    .. [1] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).

    Examples
    --------

    Suppose we wish to test whether a small sample has been drawn from a normal
    distribution. We decide that we will use the skew of the sample as a
    test statistic, and we will consider a p-value of 0.05 to be statistically
    significant.

    >>> import numpy as np
    >>> from scipy import stats
    >>> def statistic(x, axis):
    ...     return stats.skew(x, axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> rng = np.random.default_rng()
    >>> x = stats.skewnorm.rvs(a=1, size=50, random_state=rng)
    >>> statistic(x, axis=0)
    0.12457412450240658

    To determine the probability of observing such an extreme value of the
    skewness by chance if the sample were drawn from the normal distribution,
    we can perform a Monte Carlo hypothesis test. The test will draw many
    samples at random from their normal distribution, calculate the skewness
    of each sample, and compare our original skewness against this
    distribution to determine an approximate p-value.

    >>> from scipy.stats import monte_carlo_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)
    >>> print(res.statistic)
    0.12457412450240658
    >>> print(res.pvalue)
    0.7012

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is ~70%. This is greater than
    our chosen threshold of 5%, so we cannot consider this to be significant
    evidence against the null hypothesis.

    Note that this p-value essentially matches that of
    `scipy.stats.skewtest`, which relies on an asymptotic distribution of a
    test statistic based on the sample skewness.

    >>> stats.skewtest(x).pvalue
    0.6892046027110614

    This asymptotic approximation is not valid for small sample sizes, but
    `monte_carlo_test` can be used with samples of any size.

    >>> x = stats.skewnorm.rvs(a=1, size=7, random_state=rng)
    >>> # stats.skewtest(x) would produce an error due to small sample
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)

    The Monte Carlo distribution of the test statistic is provided for
    further investigation.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.null_distribution, bins=50)
    >>> ax.set_title("Monte Carlo distribution of test statistic")
    >>> ax.set_xlabel("Value of Statistic")
    >>> ax.set_ylabel("Frequency")
    >>> plt.show()

    """
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized,
                                n_resamples, batch, alternative, axis)
    (data, rvs, statistic, vectorized,
     n_resamples, batch, alternative, axis) = args

    # Some statistics return plain floats; ensure they're at least np.float64
    observed = np.asarray(statistic(*data, axis=-1))[()]

    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resamples = [rvs_i(size=(batch_actual, n_observations_i))
                     for rvs_i, n_observations_i in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    null_distribution = np.concatenate(null_distribution)
    null_distribution = null_distribution.reshape([-1] + [1]*observed.ndim)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)  # see [1]
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)  # see [1]
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return MonteCarloTestResult(observed, pvalues, null_distribution)


@dataclass
class PermutationTestResult:
    """Result object returned by `scipy.stats.permutation_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the data.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


def _all_partitions_concatenated(ns):
    """
    Generate all partitions of indices of groups of given sizes, concatenated

    `ns` is an iterable of ints.
    """
    def all_partitions(z, n):
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition)
                            for partition in partitioning]).astype(int)
        yield x


def _batch_generator(iterable, batch):
    """A generator that yields batches of elements from an iterable"""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]
    while z:  # we don't want StopIteration without yielding an empty list
        yield z
        z = [item for i, item in zip(range(batch), iterator)]


def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch,
                               random_state):
    # Returns a generator that yields arrays of size
    # `(batch, n_samples, n_obs_sample)`.
    # Each row is an independent permutation of indices 0 to `n_obs_sample`.
    batch = min(batch, n_permutations)

    if hasattr(random_state, 'permuted'):
        def batched_perm_generator():
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                # Don't permute in place, otherwise results depend on `batch`
                permuted_indices = random_state.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:  # RandomState and early Generators don't have `permuted`
        def batched_perm_generator():
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = random_state.random(size=size)
                yield np.argsort(x, axis=-1)[:batch_actual]

    return batched_perm_generator()


def _calculate_null_both(data, statistic, n_permutations, batch,
                         random_state=None):
    """
    Calculate null distribution for independent sample tests.
    """
    n_samples = len(data)

    # compute number of permutations
    # (distinct partitions of data into samples of these sizes)
    n_obs_i = [sample.shape[-1] for sample in data]  # observations per sample
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]  # total number of observations
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i-1])
                     for i in range(n_samples-1, 0, -1)])

    # perm_generator is an iterator that produces permutations of indices
    # from 0 to n_obs. We'll concatenate the samples, use these indices to
    # permute the data, then split the samples apart again.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        # Neither RandomState.permutation nor Generator.permutation
        # can permute axis-slices independently. If this feature is
        # added in the future, batches of the desired size should be
        # generated in a single call.
        perm_generator = (random_state.permutation(n_obs)
                          for i in range(n_permutations))

    batch = batch or int(n_permutations)
    null_distribution = []

    # First, concatenate all the samples. In batches, permute samples with
    # indices produced by the `perm_generator`, split them into new samples of
    # the original sizes, compute the statistic for each batch, and add these
    # statistic values to the null distribution.
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)

        # `indices` is 2D: each row is a permutation of the indices.
        # We use it to index `data` along its last axis, which corresponds
        # with observations.
        # After indexing, the second to last axis of `data_batch` corresponds
        # with permutations, and the last axis corresponds with observations.
        data_batch = data[..., indices]

        # Move the permutation axis to the front: we'll concatenate a list
        # of batched statistic values along this zeroth axis to form the
        # null distribution.
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_pairings(data, statistic, n_permutations, batch,
                             random_state=None):
    """
    Calculate null distribution for association tests.
    """
    n_samples = len(data)

    # compute number of permutations (factorial(n) permutations of each sample)
    n_obs_sample = data[0].shape[-1]  # observations per sample; same for each
    n_max = factorial(n_obs_sample)**n_samples

    # `perm_generator` is an iterator that produces a list of permutations of
    # indices from 0 to n_obs_sample, one for each sample.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        # cartesian product of the sets of all permutations of indices
        perm_generator = product(*(permutations(range(n_obs_sample))
                                   for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        # Separate random permutations of indices for each sample.
        # Again, it would be nice if RandomState/Generator.permutation
        # could permute each axis-slice separately.
        args = n_permutations, n_samples, n_obs_sample, batch, random_state
        batched_perm_generator = _pairings_permutations_gen(*args)

    null_distribution = []

    for indices in batched_perm_generator:
        indices = np.array(indices)

        # `indices` is 3D: the zeroth axis is for permutations, the next is
        # for samples, and the last is for observations. Swap the first two
        # to make the zeroth axis correspond with samples, as it does for
        # `data`.
        indices = np.swapaxes(indices, 0, 1)

        # When we're done, `data_batch` will be a list of length `n_samples`.
        # Each element will be a batch of random permutations of one sample.
        # The zeroth axis of each batch will correspond with permutations,
        # and the last will correspond with observations. (This makes it
        # easy to pass into `statistic`.)
        data_batch = [None]*n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)

        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_samples(data, statistic, n_permutations, batch,
                            random_state=None):
    """
    Calculate null distribution for paired-sample tests.
    """
    n_samples = len(data)

    # By convention, the meaning of the "samples" permutations type for
    # data with only one sample is to flip the sign of the observations.
    # Achieve this by adding a second sample - the negative of the original.
    if n_samples == 1:
        data = [data[0], -data[0]]

    # The "samples" permutation strategy is the same as the "pairings"
    # strategy except the roles of samples and observations are flipped.
    # So swap these axes, then we'll use the function for the "pairings"
    # strategy to do all the work!
    data = np.swapaxes(data, 0, -1)

    # (Of course, the user's statistic doesn't know what we've done here,
    # so we need to pass it what it's expecting.)
    def statistic_wrapped(*data, axis):
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)

    return _calculate_null_pairings(data, statistic_wrapped, n_permutations,
                                    batch, random_state)


def _permutation_test_iv(data, statistic, permutation_type, vectorized,
                         n_resamples, batch, alternative, axis, random_state):
    """Input validation for `permutation_test`."""

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = (int(n_resamples) if not np.isinf(n_resamples)
                       else np.inf)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    random_state = check_random_state(random_state)

    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, random_state)


def permutation_test(data, statistic, *, permutation_type='independent',
                     vectorized=None, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, random_state=None):
    r"""
    Performs a permutation test of a given statistic on provided data.

    For independent sample statistics, the null hypothesis is that the data are
    randomly sampled from the same distribution.
    For paired sample statistics, two null hypothesis can be tested:
    that the data are paired at random or that the data are assigned to samples
    at random.

    Parameters
    ----------
    data : iterable of array-like
        Contains the samples, each of which is an array of observations.
        Dimensions of sample arrays must be compatible for broadcasting except
        along `axis`.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts samples
        as separate arguments (e.g. ``statistic(*data)``) and returns the
        resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the sample arrays.
    permutation_type : {'independent', 'samples', 'pairings'}, optional
        The type of permutations to be performed, in accordance with the
        null hypothesis. The first two permutation types are for paired sample
        statistics, in which all samples contain the same number of
        observations and observations with corresponding indices along `axis`
        are considered to be paired; the third is for independent sample
        statistics.

        - ``'samples'`` : observations are assigned to different samples
          but remain paired with the same observations from other samples.
          This permutation type is appropriate for paired sample hypothesis
          tests such as the Wilcoxon signed-rank test and the paired t-test.
        - ``'pairings'`` : observations are paired with different observations,
          but they remain within the same sample. This permutation type is
          appropriate for association/correlation tests with statistics such
          as Spearman's :math:`\rho`, Kendall's :math:`\tau`, and Pearson's
          :math:`r`.
        - ``'independent'`` (default) : observations are assigned to different
          samples. Samples may contain different numbers of observations. This
          permutation type is appropriate for independent sample hypothesis
          tests such as the Mann-Whitney :math:`U` test and the independent
          sample t-test.

          Please see the Notes section below for more detailed descriptions
          of the permutation types.

    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed an ND sample array. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use
        of a vectorized statistic typically reduces computation time.
    n_resamples : int or np.inf, default: 9999
        Number of random permutations (resamples) used to approximate the null
        distribution. If greater than or equal to the number of distinct
        permutations, the exact null distribution will be computed.
        Note that the number of distinct permutations grows very rapidly with
        the sizes of samples, so exact tests are feasible only for very small
        data sets.
    batch : int, optional
        The number of permutations to process in each call to `statistic`.
        Memory usage is O(`batch`*``n``), where ``n`` is the total size
        of all samples, regardless of the value of `vectorized`. Default is
        ``None``, in which case ``batch`` is the number of permutations.
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis for which the p-value is calculated.
        For each alternative, the p-value is defined for exact tests as
        follows.

        - ``'greater'`` : the percentage of the null distribution that is
          greater than or equal to the observed value of the test statistic.
        - ``'less'`` : the percentage of the null distribution that is
          less than or equal to the observed value of the test statistic.
        - ``'two-sided'`` (default) : twice the smaller of the p-values above.

        Note that p-values for randomized tests are calculated according to the
        conservative (over-estimated) approximation suggested in [2]_ and [3]_
        rather than the unbiased estimator suggested in [4]_. That is, when
        calculating the proportion of the randomized null distribution that is
        as extreme as the observed value of the test statistic, the values in
        the numerator and denominator are both increased by one. An
        interpretation of this adjustment is that the observed value of the
        test statistic is always included as an element of the randomized
        null distribution.
        The convention used for two-sided p-values is not universal;
        the observed test statistic and null distribution are returned in
        case a different definition is preferred.

    axis : int, default: 0
        The axis of the (broadcasted) samples over which to calculate the
        statistic. If samples have a different number of dimensions,
        singleton dimensions are prepended to samples with fewer dimensions
        before `axis` is considered.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate permutations.

        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Returns
    -------
    res : PermutationTestResult
        An object with attributes:

        statistic : float or ndarray
            The observed test statistic of the data.
        pvalue : float or ndarray
            The p-value for the given alternative.
        null_distribution : ndarray
            The values of the test statistic generated under the null
            hypothesis.

    Notes
    -----

    The three types of permutation tests supported by this function are
    described below.

    **Unpaired statistics** (``permutation_type='independent'``):

    The null hypothesis associated with this permutation type is that all
    observations are sampled from the same underlying distribution and that
    they have been assigned to one of the samples at random.

    Suppose ``data`` contains two samples; e.g. ``a, b = data``.
    When ``1 < n_resamples < binom(n, k)``, where

    * ``k`` is the number of observations in ``a``,
    * ``n`` is the total number of observations in ``a`` and ``b``, and
    * ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),

    the data are pooled (concatenated), randomly assigned to either the first
    or second sample, and the statistic is calculated. This process is
    performed repeatedly, `permutation` times, generating a distribution of the
    statistic under the null hypothesis. The statistic of the original
    data is compared to this distribution to determine the p-value.

    When ``n_resamples >= binom(n, k)``, an exact test is performed: the data
    are *partitioned* between the samples in each distinct way exactly once,
    and the exact null distribution is formed.
    Note that for a given partitioning of the data between the samples,
    only one ordering/permutation of the data *within* each sample is
    considered. For statistics that do not depend on the order of the data
    within samples, this dramatically reduces computational cost without
    affecting the shape of the null distribution (because the frequency/count
    of each value is affected by the same factor).

    For ``a = [a1, a2, a3, a4]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``x = [b3, a1, a2, b2]`` and ``y = [a4, b1, a3]``.
    Because only one ordering/permutation of the data *within* each sample
    is considered in an exact test, a resampling like ``x = [b3, a1, b2, a2]``
    and ``y = [a4, a3, b1]`` would *not* be considered distinct from the
    example above.

    ``permutation_type='independent'`` does not support one-sample statistics,
    but it can be applied to statistics with more than two samples. In this
    case, if ``n`` is an array of the number of observations within each
    sample, the number of distinct partitions is::

        np.prod([binom(sum(n[i:]), sum(n[i+1:])) for i in range(len(n)-1)])

    **Paired statistics, permute pairings** (``permutation_type='pairings'``):

    The null hypothesis associated with this permutation type is that
    observations within each sample are drawn from the same underlying
    distribution and that pairings with elements of other samples are
    assigned at random.

    Suppose ``data`` contains only one sample; e.g. ``a, = data``, and we
    wish to consider all possible pairings of elements of ``a`` with elements
    of a second sample, ``b``. Let ``n`` be the number of observations in
    ``a``, which must also equal the number of observations in ``b``.

    When ``1 < n_resamples < factorial(n)``, the elements of ``a`` are
    randomly permuted. The user-supplied statistic accepts one data argument,
    say ``a_perm``, and calculates the statistic considering ``a_perm`` and
    ``b``. This process is performed repeatedly, `permutation` times,
    generating a distribution of the statistic under the null hypothesis.
    The statistic of the original data is compared to this distribution to
    determine the p-value.

    When ``n_resamples >= factorial(n)``, an exact test is performed:
    ``a`` is permuted in each distinct way exactly once. Therefore, the
    `statistic` is computed for each unique pairing of samples between ``a``
    and ``b`` exactly once.

    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``a_perm = [a3, a1, a2]`` while ``b`` is left
    in its original order.

    ``permutation_type='pairings'`` supports ``data`` containing any number
    of samples, each of which must contain the same number of observations.
    All samples provided in ``data`` are permuted *independently*. Therefore,
    if ``m`` is the number of samples and ``n`` is the number of observations
    within each sample, then the number of permutations in an exact test is::

        factorial(n)**m

    Note that if a two-sample statistic, for example, does not inherently
    depend on the order in which observations are provided - only on the
    *pairings* of observations - then only one of the two samples should be
    provided in ``data``. This dramatically reduces computational cost without
    affecting the shape of the null distribution (because the frequency/count
    of each value is affected by the same factor).

    **Paired statistics, permute samples** (``permutation_type='samples'``):

    The null hypothesis associated with this permutation type is that
    observations within each pair are drawn from the same underlying
    distribution and that the sample to which they are assigned is random.

    Suppose ``data`` contains two samples; e.g. ``a, b = data``.
    Let ``n`` be the number of observations in ``a``, which must also equal
    the number of observations in ``b``.

    When ``1 < n_resamples < 2**n``, the elements of ``a`` are ``b`` are
    randomly swapped between samples (maintaining their pairings) and the
    statistic is calculated. This process is performed repeatedly,
    `permutation` times,  generating a distribution of the statistic under the
    null hypothesis. The statistic of the original data is compared to this
    distribution to determine the p-value.

    When ``n_resamples >= 2**n``, an exact test is performed: the observations
    are assigned to the two samples in each distinct way (while maintaining
    pairings) exactly once.

    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``x = [b1, a2, b3]`` and ``y = [a1, b2, a3]``.

    ``permutation_type='samples'`` supports ``data`` containing any number
    of samples, each of which must contain the same number of observations.
    If ``data`` contains more than one sample, paired observations within
    ``data`` are exchanged between samples *independently*. Therefore, if ``m``
    is the number of samples and ``n`` is the number of observations within
    each sample, then the number of permutations in an exact test is::

        factorial(m)**n

    Several paired-sample statistical tests, such as the Wilcoxon signed rank
    test and paired-sample t-test, can be performed considering only the
    *difference* between two paired elements. Accordingly, if ``data`` contains
    only one sample, then the null distribution is formed by independently
    changing the *sign* of each observation.

    .. warning::
        The p-value is calculated by counting the elements of the null
        distribution that are as extreme or more extreme than the observed
        value of the statistic. Due to the use of finite precision arithmetic,
        some statistic functions return numerically distinct values when the
        theoretical values would be exactly equal. In some cases, this could
        lead to a large error in the calculated p-value. `permutation_test`
        guards against this by considering elements in the null distribution
        that are "close" (within a factor of ``1+1e-14``) to the observed
        value of the test statistic as equal to the observed value of the
        test statistic. However, the user is advised to inspect the null
        distribution to assess whether this method of comparison is
        appropriate, and if not, calculate the p-value manually. See example
        below.

    References
    ----------

    .. [1] R. A. Fisher. The Design of Experiments, 6th Ed (1951).
    .. [2] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).
    .. [3] M. D. Ernst. "Permutation Methods: A Basis for Exact Inference".
       Statistical Science (2004).
    .. [4] B. Efron and R. J. Tibshirani. An Introduction to the Bootstrap
       (1993).

    Examples
    --------

    Suppose we wish to test whether two samples are drawn from the same
    distribution. Assume that the underlying distributions are unknown to us,
    and that before observing the data, we hypothesized that the mean of the
    first sample would be less than that of the second sample. We decide that
    we will use the difference between the sample means as a test statistic,
    and we will consider a p-value of 0.05 to be statistically significant.

    For efficiency, we write the function defining the test statistic in a
    vectorized fashion: the samples ``x`` and ``y`` can be ND arrays, and the
    statistic will be calculated for each axis-slice along `axis`.

    >>> import numpy as np
    >>> def statistic(x, y, axis):
    ...     return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng()
    >>> x = norm.rvs(size=5, random_state=rng)
    >>> y = norm.rvs(size=6, loc = 3, random_state=rng)
    >>> statistic(x, y, 0)
    -3.5411688580987266

    Indeed, the test statistic is negative, suggesting that the true mean of
    the distribution underlying ``x`` is less than that of the distribution
    underlying ``y``. To determine the probability of this occuring by chance
    if the two samples were drawn from the same distribution, we perform
    a permutation test.

    >>> from scipy.stats import permutation_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> # `n_resamples=np.inf` indicates that an exact test is to be performed
    >>> res = permutation_test((x, y), statistic, vectorized=True,
    ...                        n_resamples=np.inf, alternative='less')
    >>> print(res.statistic)
    -3.5411688580987266
    >>> print(res.pvalue)
    0.004329004329004329

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is 0.4329%. This is less than our
    chosen threshold of 5%, so we consider this to be significant evidence
    against the null hypothesis in favor of the alternative.

    Because the size of the samples above was small, `permutation_test` could
    perform an exact test. For larger samples, we resort to a randomized
    permutation test.

    >>> x = norm.rvs(size=100, random_state=rng)
    >>> y = norm.rvs(size=120, loc=0.3, random_state=rng)
    >>> res = permutation_test((x, y), statistic, n_resamples=100000,
    ...                        vectorized=True, alternative='less',
    ...                        random_state=rng)
    >>> print(res.statistic)
    -0.5230459671240913
    >>> print(res.pvalue)
    0.00016999830001699983

    The approximate probability of obtaining a test statistic less than or
    equal to the observed value under the null hypothesis is 0.0225%. This is
    again less than our chosen threshold of 5%, so again we have significant
    evidence to reject the null hypothesis in favor of the alternative.

    For large samples and number of permutations, the result is comparable to
    that of the corresponding asymptotic test, the independent sample t-test.

    >>> from scipy.stats import ttest_ind
    >>> res_asymptotic = ttest_ind(x, y, alternative='less')
    >>> print(res_asymptotic.pvalue)
    0.00012688101537979522

    The permutation distribution of the test statistic is provided for
    further investigation.

    >>> import matplotlib.pyplot as plt
    >>> plt.hist(res.null_distribution, bins=50)
    >>> plt.title("Permutation distribution of test statistic")
    >>> plt.xlabel("Value of Statistic")
    >>> plt.ylabel("Frequency")
    >>> plt.show()

    Inspection of the null distribution is essential if the statistic suffers
    from inaccuracy due to limited machine precision. Consider the following
    case:

    >>> from scipy.stats import pearsonr
    >>> x = [1, 2, 4, 3]
    >>> y = [2, 4, 6, 8]
    >>> def statistic(x, y):
    ...     return pearsonr(x, y).statistic
    >>> res = permutation_test((x, y), statistic, vectorized=False,
    ...                        permutation_type='pairings',
    ...                        alternative='greater')
    >>> r, pvalue, null = res.statistic, res.pvalue, res.null_distribution

    In this case, some elements of the null distribution differ from the
    observed value of the correlation coefficient ``r`` due to numerical noise.
    We manually inspect the elements of the null distribution that are nearly
    the same as the observed value of the test statistic.

    >>> r
    0.8
    >>> unique = np.unique(null)
    >>> unique
    array([-1. , -0.8, -0.8, -0.6, -0.4, -0.2, -0.2,  0. ,  0.2,  0.2,  0.4,
            0.6,  0.8,  0.8,  1. ]) # may vary
    >>> unique[np.isclose(r, unique)].tolist()
    [0.7999999999999999, 0.8]

    If `permutation_test` were to perform the comparison naively, the
    elements of the null distribution with value ``0.7999999999999999`` would
    not be considered as extreme or more extreme as the observed value of the
    statistic, so the calculated p-value would be too small.

    >>> incorrect_pvalue = np.count_nonzero(null >= r) / len(null)
    >>> incorrect_pvalue
    0.1111111111111111  # may vary

    Instead, `permutation_test` treats elements of the null distribution that
    are within ``max(1e-14, abs(r)*1e-14)`` of the observed value of the
    statistic ``r`` to be equal to ``r``.

    >>> correct_pvalue = np.count_nonzero(null >= r - 1e-14) / len(null)
    >>> correct_pvalue
    0.16666666666666666
    >>> res.pvalue == correct_pvalue
    True

    This method of comparison is expected to be accurate in most practical
    situations, but the user is advised to assess this by inspecting the
    elements of the null distribution that are close to the observed value
    of the statistic. Also, consider the use of statistics that can be
    calculated using exact arithmetic (e.g. integer statistics).

    """
    args = _permutation_test_iv(data, statistic, permutation_type, vectorized,
                                n_resamples, batch, alternative, axis,
                                random_state)
    (data, statistic, permutation_type, vectorized, n_resamples, batch,
     alternative, axis, random_state) = args

    observed = statistic(*data, axis=-1)

    null_calculators = {"pairings": _calculate_null_pairings,
                        "samples": _calculate_null_samples,
                        "independent": _calculate_null_both}
    null_calculator_args = (data, statistic, n_resamples,
                            batch, random_state)
    calculate_null = null_calculators[permutation_type]
    null_distribution, n_resamples, exact_test = (
        calculate_null(*null_calculator_args))

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return PermutationTestResult(observed, pvalues, null_distribution)


@dataclass
class ResamplingMethod:
    """Configuration information for a statistical resampling method.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a resampling or Monte Carlo version
    of the hypothesis test.

    Attributes
    ----------
    n_resamples : int
        The number of resamples to perform or Monte Carlo samples to draw.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    """
    n_resamples: int = 9999
    batch: int = None  # type: ignore[assignment]


@dataclass
class MonteCarloMethod(ResamplingMethod):
    """Configuration information for a Monte Carlo hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a Monte Carlo version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of Monte Carlo samples to draw. Default is 9999.
    batch : int, optional
        The number of Monte Carlo samples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all samples in a single batch.
    rvs : callable or tuple of callables, optional
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples passed
        to the hypothesis test in which the `MonteCarloMethod` is used. Default
        is ``None``, in which case the hypothesis test function chooses values
        to match the standard version of the hypothesis test. For example,
        the null hypothesis of `scipy.stats.pearsonr` is typically that the
        samples are drawn from the standard normal distribution, so
        ``rvs = (rng.normal, rng.normal)`` where
        ``rng = np.random.default_rng()``.
    """
    rvs: object = None

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    rvs=self.rvs)


@dataclass
class PermutationMethod(ResamplingMethod):
    """Configuration information for a permutation hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a permutation version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.
    """
    random_state: object = None

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state)


@dataclass
class BootstrapMethod(ResamplingMethod):
    """Configuration information for a bootstrap confidence interval.

    Instances of this class can be passed into the `method` parameter of some
    confidence interval methods to generate a bootstrap confidence interval.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.

    method : {'bca', 'percentile', 'basic'}
        Whether to use the 'percentile' bootstrap ('percentile'), the 'basic'
        (AKA 'reverse') bootstrap ('basic'), or the bias-corrected and
        accelerated bootstrap ('BCa', default).
    """
    random_state: object = None
    method: str = 'BCa'

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state, method=self.method)
