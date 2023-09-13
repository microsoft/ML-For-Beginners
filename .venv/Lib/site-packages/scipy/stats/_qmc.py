"""Quasi-Monte Carlo engines and helpers."""
from __future__ import annotations

import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
    Callable,
    ClassVar,
    Literal,
    overload,
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import (
        DecimalNumber, GeneratorType, IntNumber, SeedType
    )

import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
    _initialize_v, _cscramble, _fill_p_cumulative, _draw, _fast_forward,
    _categorize, _MAXDIM
)
from ._qmc_cy import (
    _cy_wrapper_centered_discrepancy,
    _cy_wrapper_wrap_around_discrepancy,
    _cy_wrapper_mixture_discrepancy,
    _cy_wrapper_l2_star_discrepancy,
    _cy_wrapper_update_discrepancy,
    _cy_van_der_corput_scrambled,
    _cy_van_der_corput,
)


__all__ = ['scale', 'discrepancy', 'update_discrepancy',
           'QMCEngine', 'Sobol', 'Halton', 'LatinHypercube', 'PoissonDisk',
           'MultinomialQMC', 'MultivariateNormalQMC']


@overload
def check_random_state(seed: IntNumber | None = ...) -> np.random.Generator:
    ...


@overload
def check_random_state(seed: GeneratorType) -> GeneratorType:
    ...


# Based on scipy._lib._util.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional  # noqa
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` or ``RandomState`` instance, then
        the provided instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a'
                         ' numpy.random.Generator instance')


def scale(
    sample: npt.ArrayLike,
    l_bounds: npt.ArrayLike,
    u_bounds: npt.ArrayLike,
    *,
    reverse: bool = False
) -> np.ndarray:
    r"""Sample scaling from unit hypercube to different bounds.

    To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
    with :math:`a` the lower bounds and :math:`b` the upper bounds.
    The following transformation is used:

    .. math::

        (b - a) \cdot \text{sample} + a

    Parameters
    ----------
    sample : array_like (n, d)
        Sample to scale.
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds (resp. :math:`a`, :math:`b`) of transformed
        data. If `reverse` is True, range of the original data to transform
        to the unit hypercube.
    reverse : bool, optional
        Reverse the transformation from different bounds to the unit hypercube.
        Default is False.

    Returns
    -------
    sample : array_like (n, d)
        Scaled sample.

    Examples
    --------
    Transform 3 samples in the unit hypercube to bounds:

    >>> from scipy.stats import qmc
    >>> l_bounds = [-2, 0]
    >>> u_bounds = [6, 5]
    >>> sample = [[0.5 , 0.75],
    ...           [0.5 , 0.5],
    ...           [0.75, 0.25]]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    >>> sample_scaled
    array([[2.  , 3.75],
           [2.  , 2.5 ],
           [4.  , 1.25]])

    And convert back to the unit hypercube:

    >>> sample_ = qmc.scale(sample_scaled, l_bounds, u_bounds, reverse=True)
    >>> sample_
    array([[0.5 , 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25]])

    """
    sample = np.asarray(sample)

    # Checking bounds and sample
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    lower, upper = _validate_bounds(
        l_bounds=l_bounds, u_bounds=u_bounds, d=sample.shape[1]
    )

    if not reverse:
        # Checking that sample is within the hypercube
        if (sample.max() > 1.) or (sample.min() < 0.):
            raise ValueError('Sample is not in unit hypercube')

        return sample * (upper - lower) + lower
    else:
        # Checking that sample is within the bounds
        if not (np.all(sample >= lower) and np.all(sample <= upper)):
            raise ValueError('Sample is out of bounds')

        return (sample - lower) / (upper - lower)


def discrepancy(
        sample: npt.ArrayLike,
        *,
        iterative: bool = False,
        method: Literal["CD", "WD", "MD", "L2-star"] = "CD",
        workers: IntNumber = 1) -> float:
    """Discrepancy of a given sample.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    iterative : bool, optional
        Must be False if not using it for updating the discrepancy.
        Default is False. Refer to the notes for more details.
    method : str, optional
        Type of discrepancy, can be ``CD``, ``WD``, ``MD`` or ``L2-star``.
        Refer to the notes for more details. Default is ``CD``.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is given all
        CPU threads are used. Default is 1.

    Returns
    -------
    discrepancy : float
        Discrepancy.

    Notes
    -----
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.

    The lower the value is, the better the coverage of the parameter space is.

    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    A measure of uniformity is reasonable if it satisfies the following
    criteria [1]_:

    1. It is invariant under permuting factors and/or runs.
    2. It is invariant under rotation of the coordinates.
    3. It can measure not only uniformity of the sample over the hypercube,
       but also the projection uniformity of the sample over non-empty
       subset of lower dimension hypercubes.
    4. There is some reasonable geometric meaning.
    5. It is easy to compute.
    6. It satisfies the Koksma-Hlawka-like inequality.
    7. It is consistent with other criteria in experimental design.

    Four methods are available:

    * ``CD``: Centered Discrepancy - subspace involves a corner of the
      hypercube
    * ``WD``: Wrap-around Discrepancy - subspace can wrap around bounds
    * ``MD``: Mixture Discrepancy - mix between CD/WD covering more criteria
    * ``L2-star``: L2-star discrepancy - like CD BUT variant to rotation

    See [2]_ for precise definitions of each method.

    Lastly, using ``iterative=True``, it is possible to compute the
    discrepancy as if we had :math:`n+1` samples. This is useful if we want
    to add a point to a sampling and check the candidate which would give the
    lowest discrepancy. Then you could just update the discrepancy with
    each candidate using `update_discrepancy`. This method is faster than
    computing the discrepancy for a large number of candidates.

    References
    ----------
    .. [1] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [2] Zhou Y.-D. et al. "Mixture discrepancy for quasi-random point sets."
       Journal of Complexity, 29 (3-4) , pp. 283-301, 2013.
    .. [3] T. T. Warnock. "Computational investigations of low discrepancy
       point sets." Applications of Number Theory to Numerical
       Analysis, Academic Press, pp. 319-343, 1972.

    Examples
    --------
    Calculate the quality of the sample using the discrepancy:

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> space
    array([[0.08333333, 0.41666667],
           [0.25      , 0.91666667],
           [0.41666667, 0.25      ],
           [0.58333333, 0.75      ],
           [0.75      , 0.08333333],
           [0.91666667, 0.58333333]])
    >>> qmc.discrepancy(space)
    0.008142039609053464

    We can also compute iteratively the ``CD`` discrepancy by using
    ``iterative=True``.

    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order="C")

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError("Sample is not in unit hypercube")

    workers = _validate_workers(workers)

    methods = {
        "CD": _cy_wrapper_centered_discrepancy,
        "WD": _cy_wrapper_wrap_around_discrepancy,
        "MD": _cy_wrapper_mixture_discrepancy,
        "L2-star": _cy_wrapper_l2_star_discrepancy,
    }

    if method in methods:
        return methods[method](sample, iterative, workers=workers)
    else:
        raise ValueError(f"{method!r} is not a valid method. It must be one of"
                         f" {set(methods)!r}")


def update_discrepancy(
        x_new: npt.ArrayLike,
        sample: npt.ArrayLike,
        initial_disc: DecimalNumber) -> float:
    """Update the centered discrepancy with a new sample.

    Parameters
    ----------
    x_new : array_like (1, d)
        The new sample to add in `sample`.
    sample : array_like (n, d)
        The initial sample.
    initial_disc : float
        Centered discrepancy of the `sample`.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the sample composed of `x_new` and `sample`.

    Examples
    --------
    We can also compute iteratively the discrepancy by using
    ``iterative=True``.

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order="C")
    x_new = np.asarray(x_new, dtype=np.float64, order="C")

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError('Sample is not in unit hypercube')

    # Checking that x_new is within the hypercube and 1D
    if not x_new.ndim == 1:
        raise ValueError('x_new is not a 1D array')

    if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
        raise ValueError('x_new is not in unit hypercube')

    if x_new.shape[0] != sample.shape[1]:
        raise ValueError("x_new and sample must be broadcastable")

    return _cy_wrapper_update_discrepancy(x_new, sample, initial_disc)


def _perturb_discrepancy(sample: np.ndarray, i1: int, i2: int, k: int,
                         disc: float):
    """Centered discrepancy after an elementary perturbation of a LHS.

    An elementary perturbation consists of an exchange of coordinates between
    two points: ``sample[i1, k] <-> sample[i2, k]``. By construction,
    this operation conserves the LHS properties.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample (before permutation) to compute the discrepancy from.
    i1 : int
        The first line of the elementary permutation.
    i2 : int
        The second line of the elementary permutation.
    k : int
        The column of the elementary permutation.
    disc : float
        Centered discrepancy of the design before permutation.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the design after permutation.

    References
    ----------
    .. [1] Jin et al. "An efficient algorithm for constructing optimal design
       of computer experiments", Journal of Statistical Planning and
       Inference, 2005.

    """
    n = sample.shape[0]

    z_ij = sample - 0.5

    # Eq (19)
    c_i1j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i1, :])
                              + abs(z_ij) - abs(z_ij[i1, :] - z_ij)), axis=1))
    c_i2j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i2, :])
                              + abs(z_ij) - abs(z_ij[i2, :] - z_ij)), axis=1))

    # Eq (20)
    c_i1i1 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i1, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i1, :])
                                 - 0.5 * z_ij[i1, :] ** 2))
    c_i2i2 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i2, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i2, :])
                                 - 0.5 * z_ij[i2, :] ** 2))

    # Eq (22), typo in the article in the denominator i2 -> i1
    num = (2 + abs(z_ij[i2, k]) + abs(z_ij[:, k])
           - abs(z_ij[i2, k] - z_ij[:, k]))
    denum = (2 + abs(z_ij[i1, k]) + abs(z_ij[:, k])
             - abs(z_ij[i1, k] - z_ij[:, k]))
    gamma = num / denum

    # Eq (23)
    c_p_i1j = gamma * c_i1j
    # Eq (24)
    c_p_i2j = c_i2j / gamma

    alpha = (1 + abs(z_ij[i2, k])) / (1 + abs(z_ij[i1, k]))
    beta = (2 - abs(z_ij[i2, k])) / (2 - abs(z_ij[i1, k]))

    g_i1 = np.prod(1. + abs(z_ij[i1, :]))
    g_i2 = np.prod(1. + abs(z_ij[i2, :]))
    h_i1 = np.prod(1. + 0.5 * abs(z_ij[i1, :]) - 0.5 * (z_ij[i1, :] ** 2))
    h_i2 = np.prod(1. + 0.5 * abs(z_ij[i2, :]) - 0.5 * (z_ij[i2, :] ** 2))

    # Eq (25), typo in the article g is missing
    c_p_i1i1 = ((g_i1 * alpha) / (n ** 2) - 2. * alpha * beta * h_i1 / n)
    # Eq (26), typo in the article n ** 2
    c_p_i2i2 = ((g_i2 / ((n ** 2) * alpha)) - (2. * h_i2 / (n * alpha * beta)))

    # Eq (26)
    sum_ = c_p_i1j - c_i1j + c_p_i2j - c_i2j

    mask = np.ones(n, dtype=bool)
    mask[[i1, i2]] = False
    sum_ = sum(sum_[mask])

    disc_ep = (disc + c_p_i1i1 - c_i1i1 + c_p_i2i2 - c_i2i2 + 2 * sum_)

    return disc_ep


def primes_from_2_to(n: int) -> np.ndarray:
    """Prime numbers from 2 to *n*.

    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.

    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.

    Notes
    -----
    Taken from [1]_ by P.T. Roy, written consent given on 23.04.2021
    by the original author, Bruno Astrolino, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        k = 3 * i + 1 | 1
        sieve[k * k // 3::2 * k] = False
        sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n: IntNumber) -> list[int]:
    """List of the n-first prime numbers.

    Parameters
    ----------
    n : int
        Number of prime numbers wanted.

    Returns
    -------
    primes : list(int)
        List of primes.

    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]  # type: ignore[misc]

    if len(primes) < n:
        big_number = 2000
        while 'Not enough primes':
            primes = primes_from_2_to(big_number)[:n]  # type: ignore
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def _van_der_corput_permutations(
    base: IntNumber, *, random_state: SeedType = None
) -> np.ndarray:
    """Permutations for scrambling a Van der Corput sequence.

    Parameters
    ----------
    base : int
        Base of the sequence.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Returns
    -------
    permutations : array_like
        Permutation indices.

    Notes
    -----
    In Algorithm 1 of Owen 2017, a permutation of `np.arange(base)` is
    created for each positive integer `k` such that `1 - base**-k < 1`
    using floating-point arithmetic. For double precision floats, the
    condition `1 - base**-k < 1` can also be written as `base**-k >
    2**-54`, which makes it more apparent how many permutations we need
    to create.
    """
    rng = check_random_state(random_state)
    count = math.ceil(54 / math.log2(base)) - 1
    permutations = np.repeat(np.arange(base)[None], count, axis=0)
    for perm in permutations:
        rng.shuffle(perm)

    return permutations


def van_der_corput(
        n: IntNumber,
        base: IntNumber = 2,
        *,
        start_index: IntNumber = 0,
        scramble: bool = False,
        permutations: npt.ArrayLike | None = None,
        seed: SeedType = None,
        workers: IntNumber = 1) -> np.ndarray:
    """Van der Corput sequence.

    Pseudo-random number generator based on a b-adic expansion.

    Scrambling uses permutations of the remainders (see [1]_). Multiple
    permutations are applied to construct a point. The sequence of
    permutations has to be the same for all points of the sequence.

    Parameters
    ----------
    n : int
        Number of element of the sequence.
    base : int, optional
        Base of the sequence. Default is 2.
    start_index : int, optional
        Index to start the sequence from. Default is 0.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    permutations : array_like, optional
        Permutations used for scrambling.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    sequence : list (n,)
        Sequence of Van der Corput.

    References
    ----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    """
    if base < 2:
        raise ValueError("'base' must be at least 2")

    if scramble:
        if permutations is None:
            permutations = _van_der_corput_permutations(
                base=base, random_state=seed
            )
        else:
            permutations = np.asarray(permutations)

        return _cy_van_der_corput_scrambled(n, base, start_index,
                                            permutations, workers)

    else:
        return _cy_van_der_corput(n, base, start_index, workers)


class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.

    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator (used for the
    ``seed``).

    **Subclassing**

    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(d, seed=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``seed`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``_random(n, *, workers=1)``: draw ``n`` from the engine. ``workers``
      is used for parallelism. See `Halton` for example.

    Optionally, two other methods can be overwritten by subclasses:

    * ``reset``: Reset the engine to its original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.

    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:

    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def _random(self, n=1, *, workers=1):
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self

    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.

    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    We can also reset the state of the generator and resample again.

    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    """

    @abstractmethod
    def __init__(
        self,
        d: IntNumber,
        *,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        if not np.issubdtype(type(d), np.integer) or d < 0:
            raise ValueError('d must be a non-negative integer value')

        self.d = d

        if isinstance(seed, np.random.Generator):
            # Spawn a Generator that we can own and reset.
            self.rng = _rng_spawn(seed, 1)[0]
        else:
            # Create our instance of Generator, does not need spawning
            # Also catch RandomState which cannot be spawned
            self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(self.rng)

        self.num_generated = 0

        config = {
            # random-cd
            "n_nochange": 100,
            "n_iters": 10_000,
            "rng": self.rng,

            # lloyd
            "tol": 1e-5,
            "maxiter": 10,
            "qhull_options": None,
        }
        self.optimization_method = _select_optimizer(optimization, config)

    @abstractmethod
    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        ...

    def random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        workers : int, optional
            Only supported with `Halton`.
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        sample = self._random(n, workers=workers)
        if self.optimization_method is not None:
            sample = self.optimization_method(sample)

        self.num_generated += n
        return sample

    def integers(
        self,
        l_bounds: npt.ArrayLike,
        *,
        u_bounds: npt.ArrayLike | None = None,
        n: IntNumber = 1,
        endpoint: bool = False,
        workers: IntNumber = 1
    ) -> np.ndarray:
        r"""
        Draw `n` integers from `l_bounds` (inclusive) to `u_bounds`
        (exclusive), or if endpoint=True, `l_bounds` (inclusive) to
        `u_bounds` (inclusive).

        Parameters
        ----------
        l_bounds : int or array-like of ints
            Lowest (signed) integers to be drawn (unless ``u_bounds=None``,
            in which case this parameter is 0 and this value is used for
            `u_bounds`).
        u_bounds : int or array-like of ints, optional
            If provided, one above the largest (signed) integer to be drawn
            (see above for behavior if ``u_bounds=None``).
            If array-like, must contain integer values.
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        endpoint : bool, optional
            If true, sample from the interval ``[l_bounds, u_bounds]`` instead
            of the default ``[l_bounds, u_bounds)``. Defaults is False.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Only supported when using `Halton`
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        Notes
        -----
        It is safe to just use the same ``[0, 1)`` to integer mapping
        with QMC that you would use with MC. You still get unbiasedness,
        a strong law of large numbers, an asymptotically infinite variance
        reduction and a finite sample variance bound.

        To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
        with :math:`a` the lower bounds and :math:`b` the upper bounds,
        the following transformation is used:

        .. math::

            \text{floor}((b - a) \cdot \text{sample} + a)

        """
        if u_bounds is None:
            u_bounds = l_bounds
            l_bounds = 0

        u_bounds = np.atleast_1d(u_bounds)
        l_bounds = np.atleast_1d(l_bounds)

        if endpoint:
            u_bounds = u_bounds + 1

        if (not np.issubdtype(l_bounds.dtype, np.integer) or
                not np.issubdtype(u_bounds.dtype, np.integer)):
            message = ("'u_bounds' and 'l_bounds' must be integers or"
                       " array-like of integers")
            raise ValueError(message)

        if isinstance(self, Halton):
            sample = self.random(n=n, workers=workers)
        else:
            sample = self.random(n=n)

        sample = scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
        sample = np.floor(sample).astype(np.int64)

        return sample

    def reset(self) -> QMCEngine:
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n: IntNumber) -> QMCEngine:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        self.random(n=n)
        return self


class Halton(QMCEngine):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. The Halton sequence uses the base-two Van der
    Corput sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    The Halton sequence has severe striping artifacts for even modestly
    large dimensions. These can be ameliorated by scrambling. Scrambling
    also supports replication-based error estimates and extends
    applicabiltiy to unbounded integrands.

    References
    ----------
    .. [1] Halton, "On the efficiency of certain quasi-random sequences of
       points in evaluating multi-dimensional integrals", Numerische
       Mathematik, 1960.
    .. [2] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Halton(d=2, scramble=False)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111],
           [0.125     , 0.44444444]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.088893711419753

    If some wants to continue an existing design, extra points can be obtained
    by calling again `random`. Alternatively, you can skip some points like:

    >>> _ = sampler.fast_forward(5)
    >>> sample_continued = sampler.random(n=5)
    >>> sample_continued
    array([[0.3125    , 0.37037037],
           [0.8125    , 0.7037037 ],
           [0.1875    , 0.14814815],
           [0.6875    , 0.48148148],
           [0.4375    , 0.81481481]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.125     , 3.11111111],
           [8.125     , 4.11111111],
           [1.875     , 2.44444444],
           [6.875     , 3.44444444],
           [4.375     , 4.44444444]])

    """

    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True,
                           'optimization': optimization}
        super().__init__(d=d, optimization=optimization, seed=seed)
        self.seed = seed

        # important to have ``type(bdim) == int`` for performance reason
        self.base = [int(bdim) for bdim in n_primes(d)]
        self.scramble = scramble

        self._initialize_permutations()

    def _initialize_permutations(self) -> None:
        """Initialize permutations for all Van der Corput sequences.

        Permutations are only needed for scrambling.
        """
        self._permutations: list = [None] * len(self.base)
        if self.scramble:
            for i, bdim in enumerate(self.base):
                permutations = _van_der_corput_permutations(
                    base=bdim, random_state=self.rng
                )

                self._permutations[i] = permutations

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        workers = _validate_workers(workers)
        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [van_der_corput(n, bdim, start_index=self.num_generated,
                                 scramble=self.scramble,
                                 permutations=self._permutations[i],
                                 workers=workers)
                  for i, bdim in enumerate(self.base)]

        return np.array(sample).T.reshape(n, self.d)


class LatinHypercube(QMCEngine):
    r"""Latin hypercube sampling (LHS).

    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    centered : bool, optional
        Center samples within cells of a multi-dimensional grid.
        Default is False.

        .. deprecated:: 1.10.0
            `centered` is deprecated as of SciPy 1.10.0 and will be removed in
            1.12.0. Use `scramble` instead. ``centered=True`` corresponds to
            ``scramble=False``.

    scramble : bool, optional
        When False, center samples within cells of a multi-dimensional grid.
        Otherwise, samples are randomly placed within cells of the grid.

        .. note::
            Setting ``scramble=False`` does not ensure deterministic output.
            For that, use the `seed` parameter.

        Default is True.

        .. versionadded:: 1.10.0

    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.8.0
        .. versionchanged:: 1.10.0
            Add ``lloyd``.

    strength : {1, 2}, optional
        Strength of the LHS. ``strength=1`` produces a plain LHS while
        ``strength=2`` produces an orthogonal array based LHS of strength 2
        [7]_, [8]_. In that case, only ``n=p**2`` points can be sampled,
        with ``p`` a prime number. It also constrains ``d <= p + 1``.
        Default is 1.

        .. versionadded:: 1.8.0

    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----

    When LHS is used for integrating a function :math:`f` over :math:`n`,
    LHS is extremely effective on integrands that are nearly additive [2]_.
    With a LHS of :math:`n` points, the variance of the integral is always
    lower than plain MC on :math:`n-1` points [3]_. There is a central limit
    theorem for LHS on the mean and variance of the integral [4]_, but not
    necessarily for optimized LHS due to the randomization.

    :math:`A` is called an orthogonal array of strength :math:`t` if in each
    n-row-by-t-column submatrix of :math:`A`: all :math:`p^t` possible
    distinct rows occur the same number of times. The elements of :math:`A`
    are in the set :math:`\{0, 1, ..., p-1\}`, also called symbols.
    The constraint that :math:`p` must be a prime number is to allow modular
    arithmetic. Increasing strength adds some symmetry to the sub-projections
    of a sample. With strength 2, samples are symmetric along the diagonals of
    2D sub-projections. This may be undesirable, but on the other hand, the
    sample dispersion is improved.

    Strength 1 (plain LHS) brings an advantage over strength 0 (MC) and
    strength 2 is a useful increment over strength 1. Going to strength 3 is
    a smaller increment and scrambled QMC like Sobol', Halton are more
    performant [7]_.

    To create a LHS of strength 2, the orthogonal array :math:`A` is
    randomized by applying a random, bijective map of the set of symbols onto
    itself. For example, in column 0, all 0s might become 2; in column 1,
    all 0s might become 1, etc.
    Then, for each column :math:`i` and symbol :math:`j`, we add a plain,
    one-dimensional LHS of size :math:`p` to the subarray where
    :math:`A^i = j`. The resulting matrix is finally divided by :math:`p`.

    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code."
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    .. [5] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [6] Damblin et al., "Numerical studies of space filling designs:
       optimization of Latin Hypercube Samples and subprojection properties."
       Journal of Simulation, 2013.
    .. [7] A. B. Owen , "Orthogonal arrays for computer experiments,
       integration and visualization." Statistica Sinica, 1992.
    .. [8] B. Tang, "Orthogonal Array-Based Latin Hypercubes."
       Journal of the American Statistical Association, 1993.
    .. [9] Susan K. Seaholm et al. "Latin hypercube sampling and the
       sensitivity analysis of a Monte Carlo epidemic model".
       Int J Biomed Comput, 23(1-2), 97-112,
       :doi:`10.1016/0020-7101(88)90067-0`, 1988.

    Examples
    --------
    In [9]_, a Latin Hypercube sampling strategy was used to sample a
    parameter space to study the importance of each parameter of an epidemic
    model. Such analysis is also called a sensitivity analysis.

    Since the dimensionality of the problem is high (6), it is computationally
    expensive to cover the space. When numerical experiments are costly,
    QMC enables analysis that may not be possible if using a grid.

    The six parameters of the model represented the probability of illness,
    the probability of withdrawal, and four contact probabilities,
    The authors assumed uniform distributions for all parameters and generated
    50 samples.

    Using `scipy.stats.qmc.LatinHypercube` to replicate the protocol, the
    first step is to create a sample in the unit hypercube:

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=6)
    >>> sample = sampler.random(n=50)

    Then the sample can be scaled to the appropriate bounds:

    >>> l_bounds = [0.000125, 0.01, 0.0025, 0.05, 0.47, 0.7]
    >>> u_bounds = [0.000375, 0.03, 0.0075, 0.15, 0.87, 0.9]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    Such a sample was used to run the model 50 times, and a polynomial
    response surface was constructed. This allowed the authors to study the
    relative importance of each parameter across the range of
    possibilities of every other parameter.
    In this computer experiment, they showed a 14-fold reduction in the number
    of samples required to maintain an error below 2% on their response surface
    when compared to a grid sampling.

    Below are other examples showing alternative ways to construct LHS
    with even better coverage of the space.

    Using a base LHS as a baseline.

    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0196...  # random

    Use the `optimization` keyword argument to produce a LHS with
    lower discrepancy at higher computational cost.

    >>> sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0176...  # random

    Use the `strength` keyword argument to produce an orthogonal array based
    LHS of strength 2. In this case, the number of sample points must be the
    square of a prime number.

    >>> sampler = qmc.LatinHypercube(d=2, strength=2)
    >>> sample = sampler.random(n=9)
    >>> qmc.discrepancy(sample)
    0.00526...  # random

    Options could be combined to produce an optimized centered
    orthogonal array based LHS. After optimization, the result would not
    be guaranteed to be of strength 2.

    """

    def __init__(
        self, d: IntNumber, *, centered: bool = False,
        scramble: bool = True,
        strength: int = 1,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        if centered:
            scramble = False
            warnings.warn(
                "'centered' is deprecated and will be removed in SciPy 1.12."
                " Please use 'scramble' instead. 'centered=True' corresponds"
                " to 'scramble=False'.",
                stacklevel=2
            )

        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True, 'strength': strength,
                           'optimization': optimization}
        super().__init__(d=d, seed=seed, optimization=optimization)
        self.scramble = scramble

        lhs_method_strength = {
            1: self._random_lhs,
            2: self._random_oa_lhs
        }

        try:
            self.lhs_method: Callable = lhs_method_strength[strength]
        except KeyError as exc:
            message = (f"{strength!r} is not a valid strength. It must be one"
                       f" of {set(lhs_method_strength)!r}")
            raise ValueError(message) from exc

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        lhs = self.lhs_method(n)
        return lhs

    def _random_lhs(self, n: IntNumber = 1) -> np.ndarray:
        """Base LHS algorithm."""
        if not self.scramble:
            samples: np.ndarray | float = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))

        perms = np.tile(np.arange(1, n + 1),
                        (self.d, 1))  # type: ignore[arg-type]
        for i in range(self.d):
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        samples = (perms - samples) / n
        return samples

    def _random_oa_lhs(self, n: IntNumber = 4) -> np.ndarray:
        """Orthogonal array based LHS of strength 2."""
        p = np.sqrt(n).astype(int)
        n_row = p**2
        n_col = p + 1

        primes = primes_from_2_to(p + 1)
        if p not in primes or n != n_row:
            raise ValueError(
                "n is not the square of a prime number. Close"
                f" values are {primes[-2:]**2}"
            )
        if self.d > p + 1:
            raise ValueError("n is too small for d. Must be n > (d-1)**2")

        oa_sample = np.zeros(shape=(n_row, n_col), dtype=int)

        # OA of strength 2
        arrays = np.tile(np.arange(p), (2, 1))
        oa_sample[:, :2] = np.stack(np.meshgrid(*arrays),
                                    axis=-1).reshape(-1, 2)
        for p_ in range(1, p):
            oa_sample[:, 2+p_-1] = np.mod(oa_sample[:, 0]
                                          + p_*oa_sample[:, 1], p)

        # scramble the OA
        oa_sample_ = np.empty(shape=(n_row, n_col), dtype=int)
        for j in range(n_col):
            perms = self.rng.permutation(p)
            oa_sample_[:, j] = perms[oa_sample[:, j]]

        # following is making a scrambled OA into an OA-LHS
        oa_lhs_sample = np.zeros(shape=(n_row, n_col))
        lhs_engine = LatinHypercube(d=1, scramble=self.scramble, strength=1,
                                    seed=self.rng)  # type: QMCEngine
        for j in range(n_col):
            for k in range(p):
                idx = oa_sample[:, j] == k
                lhs = lhs_engine.random(p).flatten()
                oa_lhs_sample[:, j][idx] = lhs + oa_sample[:, j][idx]

                lhs_engine = lhs_engine.reset()

        oa_lhs_sample /= p

        return oa_lhs_sample[:, :self.d]  # type: ignore


class Sobol(QMCEngine):
    """Engine for generating (scrambled) Sobol' sequences.

    Sobol' sequences are low-discrepancy, quasi-random numbers. Points
    can be drawn using two methods:

    * `random_base2`: safely draw :math:`n=2^m` points. This method
      guarantees the balance properties of the sequence.
    * `random`: draw an arbitrary number of points from the
      sequence. See warning below.

    Parameters
    ----------
    d : int
        Dimensionality of the sequence. Max dimensionality is 21201.
    scramble : bool, optional
        If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
        Default is True.
    bits : int, optional
        Number of bits of the generator. Control the maximum number of points
        that can be generated, which is ``2**bits``. Maximal value is 64.
        It does not correspond to the return type, which is always
        ``np.float64`` to prevent points from repeating themselves.
        Default is None, which for backward compatibility, corresponds to 30.

        .. versionadded:: 1.9.0
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    Sobol' sequences [1]_ provide :math:`n=2^m` low discrepancy points in
    :math:`[0,1)^{d}`. Scrambling them [3]_ makes them suitable for singular
    integrands, provides a means of error estimation, and can improve their
    rate of convergence. The scrambling strategy which is implemented is a
    (left) linear matrix scramble (LMS) followed by a digital random shift
    (LMS+shift) [2]_.

    There are many versions of Sobol' sequences depending on their
    'direction numbers'. This code uses direction numbers from [4]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/.

    .. warning::

       Sobol' sequences are a quadrature rule and they lose their balance
       properties if one uses a sample size that is not a power of 2, or skips
       the first point, or thins the sequence [5]_.

       If :math:`n=2^m` points are not enough then one should take :math:`2^M`
       points for :math:`M>m`. When scrambling, the number R of independent
       replicates does not have to be a power of 2.

       Sobol' sequences are generated to some number :math:`B` of bits.
       After :math:`2^B` points have been generated, the sequence would
       repeat. Hence, an error is raised.
       The number of bits can be controlled with the parameter `bits`.

    References
    ----------
    .. [1] I. M. Sobol', "The distribution of points in a cube and the accurate
       evaluation of integrals." Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
       1967.
    .. [2] J. Matousek, "On the L2-discrepancy for anchored boxes."
       J. of Complexity 14, 527-556, 1998.
    .. [3] Art B. Owen, "Scrambling Sobol and Niederreiter-Xing points."
       Journal of Complexity, 14(4):466-489, December 1998.
    .. [4] S. Joe and F. Y. Kuo, "Constructing sobol sequences with better
       two-dimensional projections." SIAM Journal on Scientific Computing,
       30(5):2635-2654, 2008.
    .. [5] Art B. Owen, "On dropping the first Sobol' point."
       :arxiv:`2008.08051`, 2020.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Sobol'.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Sobol(d=2, scramble=False)
    >>> sample = sampler.random_base2(m=3)
    >>> sample
    array([[0.   , 0.   ],
           [0.5  , 0.5  ],
           [0.75 , 0.25 ],
           [0.25 , 0.75 ],
           [0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.013882107204860938

    To continue an existing design, extra points can be obtained
    by calling again `random_base2`. Alternatively, you can skip some
    points like:

    >>> _ = sampler.reset()
    >>> _ = sampler.fast_forward(4)
    >>> sample_continued = sampler.random_base2(m=2)
    >>> sample_continued
    array([[0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.75 , 3.125],
           [8.75 , 4.625],
           [6.25 , 2.375],
           [1.25 , 3.875]])

    """

    MAXDIM: ClassVar[int] = _MAXDIM

    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        bits: IntNumber | None = None, seed: SeedType = None,
        optimization: Literal["random-cd", "lloyd"] | None = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True, 'bits': bits,
                           'optimization': optimization}

        super().__init__(d=d, optimization=optimization, seed=seed)
        if d > self.MAXDIM:
            raise ValueError(
                f"Maximum supported dimensionality is {self.MAXDIM}."
            )

        self.bits = bits
        self.dtype_i: type

        if self.bits is None:
            self.bits = 30

        if self.bits <= 32:
            self.dtype_i = np.uint32
        elif 32 < self.bits <= 64:
            self.dtype_i = np.uint64
        else:
            raise ValueError("Maximum supported 'bits' is 64")

        self.maxn = 2**self.bits

        # v is d x maxbit matrix
        self._sv: np.ndarray = np.zeros((d, self.bits), dtype=self.dtype_i)
        _initialize_v(self._sv, dim=d, bits=self.bits)

        if not scramble:
            self._shift: np.ndarray = np.zeros(d, dtype=self.dtype_i)
        else:
            # scramble self._shift and self._sv
            self._scramble()

        self._quasi = self._shift.copy()

        # normalization constant with the largest possible number
        # calculate in Python to not overflow int with 2**64
        self._scale = 1.0 / 2 ** self.bits

        self._first_point = (self._quasi * self._scale).reshape(1, -1)
        # explicit casting to float64
        self._first_point = self._first_point.astype(np.float64)

    def _scramble(self) -> None:
        """Scramble the sequence using LMS+shift."""
        # Generate shift vector
        self._shift = np.dot(
            rng_integers(self.rng, 2, size=(self.d, self.bits),
                         dtype=self.dtype_i),
            2 ** np.arange(self.bits, dtype=self.dtype_i),
        )
        # Generate lower triangular matrices (stacked across dimensions)
        ltm = np.tril(rng_integers(self.rng, 2,
                                   size=(self.d, self.bits, self.bits),
                                   dtype=self.dtype_i))
        _cscramble(
            dim=self.d, bits=self.bits,  # type: ignore[arg-type]
            ltm=ltm, sv=self._sv
        )

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw next point(s) in the Sobol' sequence.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        sample: np.ndarray = np.empty((n, self.d), dtype=np.float64)

        if n == 0:
            return sample

        total_n = self.num_generated + n
        if total_n > self.maxn:
            msg = (
                f"At most 2**{self.bits}={self.maxn} distinct points can be "
                f"generated. {self.num_generated} points have been previously "
                f"generated, then: n={self.num_generated}+{n}={total_n}. "
            )
            if self.bits != 64:
                msg += "Consider increasing `bits`."
            raise ValueError(msg)

        if self.num_generated == 0:
            # verify n is 2**n
            if not (n & (n - 1) == 0):
                warnings.warn("The balance properties of Sobol' points require"
                              " n to be a power of 2.", stacklevel=2)

            if n == 1:
                sample = self._first_point
            else:
                _draw(
                    n=n - 1, num_gen=self.num_generated, dim=self.d,
                    scale=self._scale, sv=self._sv, quasi=self._quasi,
                    sample=sample
                )
                sample = np.concatenate(
                    [self._first_point, sample]
                )[:n]  # type: ignore[misc]
        else:
            _draw(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                scale=self._scale, sv=self._sv, quasi=self._quasi,
                sample=sample
            )

        return sample

    def random_base2(self, m: IntNumber) -> np.ndarray:
        """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        n = 2 ** m

        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError("The balance properties of Sobol' points require "
                             "n to be a power of 2. {0} points have been "
                             "previously generated, then: n={0}+2**{1}={2}. "
                             "If you still want to do this, the function "
                             "'Sobol.random()' can be used."
                             .format(self.num_generated, m, total_n))

        return self.random(n)

    def reset(self) -> Sobol:
        """Reset the engine to base state.

        Returns
        -------
        engine : Sobol
            Engine reset to its base state.

        """
        super().reset()
        self._quasi = self._shift.copy()
        return self

    def fast_forward(self, n: IntNumber) -> Sobol:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.

        """
        if self.num_generated == 0:
            _fast_forward(
                n=n - 1, num_gen=self.num_generated, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        else:
            _fast_forward(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        self.num_generated += n
        return self


class PoissonDisk(QMCEngine):
    """Poisson disk sampling.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    radius : float
        Minimal distance to keep between points when sampling new candidates.
    hypersphere : {"volume", "surface"}, optional
        Sampling strategy to generate potential candidates to be added in the
        final sample. Default is "volume".

        * ``volume``: original Bridson algorithm as described in [1]_.
          New candidates are sampled *within* the hypersphere.
        * ``surface``: only sample the surface of the hypersphere.
    ncandidates : int
        Number of candidates to sample per iteration. More candidates result
        in a denser sampling as more candidates can be accepted per iteration.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    Poisson disk sampling is an iterative sampling strategy. Starting from
    a seed sample, `ncandidates` are sampled in the hypersphere
    surrounding the seed. Candidates bellow a certain `radius` or outside the
    domain are rejected. New samples are added in a pool of sample seed. The
    process stops when the pool is empty or when the number of required
    samples is reached.

    The maximum number of point that a sample can contain is directly linked
    to the `radius`. As the dimension of the space increases, a higher radius
    spreads the points further and help overcome the curse of dimensionality.
    See the :ref:`quasi monte carlo tutorial <quasi-monte-carlo>` for more
    details.

    .. warning::

       The algorithm is more suitable for low dimensions and sampling size
       due to its iterative nature and memory requirements.
       Selecting a small radius with a high dimension would
       mean that the space could contain more samples than using lower
       dimension or a bigger radius.

    Some code taken from [2]_, written consent given on 31.03.2021
    by the original author, Shamis, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] Robert Bridson, "Fast Poisson Disk Sampling in Arbitrary
       Dimensions." SIGGRAPH, 2007.
    .. [2] `StackOverflow <https://stackoverflow.com/questions/66047540>`__.

    Examples
    --------
    Generate a 2D sample using a `radius` of 0.2.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.collections import PatchCollection
    >>> from scipy.stats import qmc
    >>>
    >>> rng = np.random.default_rng()
    >>> radius = 0.2
    >>> engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    >>> sample = engine.random(20)

    Visualizing the 2D sample and showing that no points are closer than
    `radius`. ``radius/2`` is used to visualize non-intersecting circles.
    If two samples are exactly at `radius` from each other, then their circle
    of radius ``radius/2`` will touch.

    >>> fig, ax = plt.subplots()
    >>> _ = ax.scatter(sample[:, 0], sample[:, 1])
    >>> circles = [plt.Circle((xi, yi), radius=radius/2, fill=False)
    ...            for xi, yi in sample]
    >>> collection = PatchCollection(circles, match_original=True)
    >>> ax.add_collection(collection)
    >>> _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
    ...            xlim=[0, 1], ylim=[0, 1])
    >>> plt.show()

    Such visualization can be seen as circle packing: how many circle can
    we put in the space. It is a np-hard problem. The method `fill_space`
    can be used to add samples until no more samples can be added. This is
    a hard problem and parameters may need to be adjusted manually. Beware of
    the dimension: as the dimensionality increases, the number of samples
    required to fill the space increases exponentially
    (curse-of-dimensionality).

    """

    def __init__(
        self,
        d: IntNumber,
        *,
        radius: DecimalNumber = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: IntNumber = 30,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        seed: SeedType = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'radius': radius,
                           'hypersphere': hypersphere,
                           'ncandidates': ncandidates,
                           'optimization': optimization}
        super().__init__(d=d, optimization=optimization, seed=seed)

        hypersphere_sample = {
            "volume": self._hypersphere_volume_sample,
            "surface": self._hypersphere_surface_sample
        }

        try:
            self.hypersphere_method = hypersphere_sample[hypersphere]
        except KeyError as exc:
            message = (
                f"{hypersphere!r} is not a valid hypersphere sampling"
                f" method. It must be one of {set(hypersphere_sample)!r}")
            raise ValueError(message) from exc

        # size of the sphere from which the samples are drawn relative to the
        # size of a disk (radius)
        # for the surface sampler, all new points are almost exactly 1 radius
        # away from at least one existing sample +eps to avoid rejection
        self.radius_factor = 2 if hypersphere == "volume" else 1.001
        self.radius = radius
        self.radius_squared = self.radius**2

        # sample to generate per iteration in the hypersphere around center
        self.ncandidates = ncandidates

        with np.errstate(divide='ignore'):
            self.cell_size = self.radius / np.sqrt(self.d)
            self.grid_size = (
                np.ceil(np.ones(self.d) / self.cell_size)
            ).astype(int)

        self._initialize_grid_pool()

    def _initialize_grid_pool(self):
        """Sampling pool and sample grid."""
        self.sample_pool = []
        # Positions of cells
        # n-dim value for each grid cell
        self.sample_grid = np.empty(
            np.append(self.grid_size, self.d),
            dtype=np.float32
        )
        # Initialise empty cells with NaNs
        self.sample_grid.fill(np.nan)

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the interval ``[0, 1]``.

        Note that it can return fewer samples if the space is full.
        See the note section of the class.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        if n == 0 or self.d == 0:
            return np.empty((n, self.d))

        def in_limits(sample: np.ndarray) -> bool:
            return (sample.max() <= 1.) and (sample.min() >= 0.)

        def in_neighborhood(candidate: np.ndarray, n: int = 2) -> bool:
            """
            Check if there are samples closer than ``radius_squared`` to the
            `candidate` sample.
            """
            indices = (candidate / self.cell_size).astype(int)
            ind_min = np.maximum(indices - n, np.zeros(self.d, dtype=int))
            ind_max = np.minimum(indices + n + 1, self.grid_size)

            # Check if the center cell is empty
            if not np.isnan(self.sample_grid[tuple(indices)][0]):
                return True

            a = [slice(ind_min[i], ind_max[i]) for i in range(self.d)]

            # guards against: invalid value encountered in less as we are
            # comparing with nan and returns False. Which is wanted.
            with np.errstate(invalid='ignore'):
                if np.any(
                    np.sum(
                        np.square(candidate - self.sample_grid[tuple(a)]),
                        axis=self.d
                    ) < self.radius_squared
                ):
                    return True

            return False

        def add_sample(candidate: np.ndarray) -> None:
            self.sample_pool.append(candidate)
            indices = (candidate / self.cell_size).astype(int)
            self.sample_grid[tuple(indices)] = candidate
            curr_sample.append(candidate)

        curr_sample: list[np.ndarray] = []

        if len(self.sample_pool) == 0:
            # the pool is being initialized with a single random sample
            add_sample(self.rng.random(self.d))
            num_drawn = 1
        else:
            num_drawn = 0

        # exhaust sample pool to have up to n sample
        while len(self.sample_pool) and num_drawn < n:
            # select a sample from the available pool
            idx_center = rng_integers(self.rng, len(self.sample_pool))
            center = self.sample_pool[idx_center]
            del self.sample_pool[idx_center]

            # generate candidates around the center sample
            candidates = self.hypersphere_method(
                center, self.radius * self.radius_factor, self.ncandidates
            )

            # keep candidates that satisfy some conditions
            for candidate in candidates:
                if in_limits(candidate) and not in_neighborhood(candidate):
                    add_sample(candidate)

                    num_drawn += 1
                    if num_drawn >= n:
                        break

        self.num_generated += num_drawn
        return np.array(curr_sample)

    def fill_space(self) -> np.ndarray:
        """Draw ``n`` samples in the interval ``[0, 1]``.

        Unlike `random`, this method will try to add points until
        the space is full. Depending on ``candidates`` (and to a lesser extent
        other parameters), some empty areas can still be present in the sample.

        .. warning::

           This can be extremely slow in high dimensions or if the
           ``radius`` is very small-with respect to the dimensionality.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        return self.random(np.inf)  # type: ignore[arg-type]

    def reset(self) -> PoissonDisk:
        """Reset the engine to base state.

        Returns
        -------
        engine : PoissonDisk
            Engine reset to its base state.

        """
        super().reset()
        self._initialize_grid_pool()
        return self

    def _hypersphere_volume_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling within hypersphere."""
        # should remove samples within r/2
        x = self.rng.standard_normal(size=(candidates, self.d))
        ssq = np.sum(x**2, axis=1)
        fr = radius * gammainc(self.d/2, ssq/2)**(1/self.d) / np.sqrt(ssq)
        fr_tiled = np.tile(
            fr.reshape(-1, 1), (1, self.d)  # type: ignore[arg-type]
        )
        p = center + np.multiply(x, fr_tiled)
        return p

    def _hypersphere_surface_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling on the hypersphere's surface."""
        vec = self.rng.standard_normal(size=(candidates, self.d))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        p = center + np.multiply(vec, radius)
        return p


class MultivariateNormalQMC:
    r"""QMC sampling from a multivariate Normal :math:`N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = dist.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()

    """

    def __init__(
            self, mean: npt.ArrayLike, cov: npt.ArrayLike | None = None, *,
            cov_root: npt.ArrayLike | None = None,
            inv_transform: bool = True,
            engine: QMCEngine | None = None,
            seed: SeedType = None
    ) -> None:
        mean = np.array(mean, copy=False, ndmin=1)
        d = mean.shape[0]
        if cov is not None:
            # covariance matrix provided
            cov = np.array(cov, copy=False, ndmin=2)
            # check for square/symmetric cov matrix and mean vector has the
            # same d
            if not mean.shape[0] == cov.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
            if not np.allclose(cov, cov.transpose()):
                raise ValueError("Covariance matrix is not symmetric.")
            # compute Cholesky decomp; if it fails, do the eigen decomposition
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1.0e-8):
                    raise ValueError("Covariance matrix not PSD.")
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
        elif cov_root is not None:
            # root decomposition provided
            cov_root = np.atleast_2d(cov_root)
            if not mean.shape[0] == cov_root.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
        else:
            # corresponds to identity covariance matrix
            cov_root = None

        self._inv_transform = inv_transform

        if not inv_transform:
            # to apply Box-Muller, we need an even number of dimensions
            engine_dim = 2 * math.ceil(d / 2)
        else:
            engine_dim = d
        if engine is None:
            self.engine = Sobol(
                d=engine_dim, scramble=True, bits=30, seed=seed
            )  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != engine_dim:
                raise ValueError("Dimension of `engine` must be consistent"
                                 " with dimensions of mean and covariance."
                                 " If `inv_transform` is False, it must be"
                                 " an even number.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

        self._mean = mean
        self._corr_matrix = cov_root

        self._d = d

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multivariate Normal.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        base_samples = self._standard_normal_samples(n)
        return self._correlate(base_samples)

    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            return base_samples @ self._corr_matrix + self._mean
        else:
            # avoid multiplying with identity here
            return base_samples + self._mean

    def _standard_normal_samples(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        # get base samples
        samples = self.engine.random(n)
        if self._inv_transform:
            # apply inverse transform
            # (values to close to 0/1 result in inf values)
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]
        else:
            # apply Box-Muller transform (note: indexes starting from 1)
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin],
                                      -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            return transf_samples[:, : self._d]


class MultinomialQMC:
    r"""QMC sampling from a multinomial distribution.

    Parameters
    ----------
    pvals : array_like (k,)
        Vector of probabilities of size ``k``, where ``k`` is the number
        of categories. Elements must be non-negative and sum to 1.
    n_trials : int
        Number of trials.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    Let's define 3 categories and for a given sample, the sum of the trials
    of each category is 8. The number of trials per category is determined
    by the `pvals` associated to each category.
    Then, we sample this distribution 64 times.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultinomialQMC(
    ...     pvals=[0.2, 0.4, 0.4], n_trials=10, engine=qmc.Halton(d=1)
    ... )
    >>> sample = dist.random(64)

    We can plot the sample and verify that the median of number of trials
    for each category is following the `pvals`. That would be
    ``pvals * n_trials = [2, 4, 4]``.

    >>> fig, ax = plt.subplots()
    >>> ax.yaxis.get_major_locator().set_params(integer=True)
    >>> _ = ax.boxplot(sample)
    >>> ax.set(xlabel="Categories", ylabel="Trials")
    >>> plt.show()

    """

    def __init__(
        self, pvals: npt.ArrayLike, n_trials: IntNumber,
        *, engine: QMCEngine | None = None,
        seed: SeedType = None
    ) -> None:
        self.pvals = np.array(pvals, copy=False, ndmin=1)
        if np.min(pvals) < 0:
            raise ValueError('Elements of pvals must be non-negative.')
        if not np.isclose(np.sum(pvals), 1):
            raise ValueError('Elements of pvals must sum to 1.')
        self.n_trials = n_trials
        if engine is None:
            self.engine = Sobol(
                d=1, scramble=True, bits=30, seed=seed
            )  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != 1:
                raise ValueError("Dimension of `engine` must be 1.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multinomial distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        samples : array_like (n, pvals)
            Sample.

        """
        sample = np.empty((n, len(self.pvals)))
        for i in range(n):
            base_draws = self.engine.random(self.n_trials).ravel()
            p_cumulative = np.empty_like(self.pvals, dtype=float)
            _fill_p_cumulative(np.array(self.pvals, dtype=float), p_cumulative)
            sample_ = np.zeros_like(self.pvals, dtype=int)
            _categorize(base_draws, p_cumulative, sample_)
            sample[i] = sample_
        return sample


def _select_optimizer(
    optimization: Literal["random-cd", "lloyd"] | None, config: dict
) -> Callable | None:
    """A factory for optimization methods."""
    optimization_method: dict[str, Callable] = {
        "random-cd": _random_cd,
        "lloyd": _lloyd_centroidal_voronoi_tessellation
    }

    optimizer: partial | None
    if optimization is not None:
        try:
            optimization = optimization.lower()  # type: ignore[assignment]
            optimizer_ = optimization_method[optimization]
        except KeyError as exc:
            message = (f"{optimization!r} is not a valid optimization"
                       f" method. It must be one of"
                       f" {set(optimization_method)!r}")
            raise ValueError(message) from exc

        # config
        optimizer = partial(optimizer_, **config)
    else:
        optimizer = None

    return optimizer


def _random_cd(
    best_sample: np.ndarray, n_iters: int, n_nochange: int, rng: GeneratorType,
    **kwargs: dict
) -> np.ndarray:
    """Optimal LHS on CD.

    Create a base LHS and do random permutations of coordinates to
    lower the centered discrepancy.
    Because it starts with a normal LHS, it also works with the
    `centered` keyword argument.

    Two stopping criterion are used to stop the algorithm: at most,
    `n_iters` iterations are performed; or if there is no improvement
    for `n_nochange` consecutive iterations.
    """
    del kwargs  # only use keywords which are defined, needed by factory

    n, d = best_sample.shape

    if d == 0 or n == 0:
        return np.empty((n, d))

    if d == 1 or n == 1:
        # discrepancy measures are invariant under permuting factors and runs
        return best_sample

    best_disc = discrepancy(best_sample)

    bounds = ([0, d - 1],
              [0, n - 1],
              [0, n - 1])

    n_nochange_ = 0
    n_iters_ = 0
    while n_nochange_ < n_nochange and n_iters_ < n_iters:
        n_iters_ += 1

        col = rng_integers(rng, *bounds[0], endpoint=True)  # type: ignore[misc]
        row_1 = rng_integers(rng, *bounds[1], endpoint=True)  # type: ignore[misc]
        row_2 = rng_integers(rng, *bounds[2], endpoint=True)  # type: ignore[misc]
        disc = _perturb_discrepancy(best_sample,
                                    row_1, row_2, col,
                                    best_disc)
        if disc < best_disc:
            best_sample[row_1, col], best_sample[row_2, col] = (
                best_sample[row_2, col], best_sample[row_1, col])

            best_disc = disc
            n_nochange_ = 0
        else:
            n_nochange_ += 1

    return best_sample


def _l1_norm(sample: np.ndarray) -> float:
    return distance.pdist(sample, 'cityblock').min()


def _lloyd_iteration(
    sample: np.ndarray,
    decay: float,
    qhull_options: str
) -> np.ndarray:
    """Lloyd-Max algorithm iteration.

    Based on the implementation of Stéfan van der Walt:

    https://github.com/stefanv/lloyd

    which is:

        Copyright (c) 2021-04-21 Stéfan van der Walt
        https://github.com/stefanv/lloyd
        MIT License

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on.
    decay : float
        Relaxation decay. A positive value would move the samples toward
        their centroid, and negative value would move them away.
        1 would move the samples to their centroid.
    qhull_options : str
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after an iteration of Lloyd's algorithm.

    """
    new_sample = np.empty_like(sample)

    voronoi = Voronoi(sample, qhull_options=qhull_options)

    for ii, idx in enumerate(voronoi.point_region):
        # the region is a series of indices into self.voronoi.vertices
        # remove samples at infinity, designated by index -1
        region = [i for i in voronoi.regions[idx] if i != -1]

        # get the vertices for this region
        verts = voronoi.vertices[region]

        # clipping would be wrong, we need to intersect
        # verts = np.clip(verts, 0, 1)

        # move samples towards centroids:
        # Centroid in n-D is the mean for uniformly distributed nodes
        # of a geometry.
        centroid = np.mean(verts, axis=0)
        new_sample[ii] = sample[ii] + (centroid - sample[ii]) * decay

    # only update sample to centroid within the region
    is_valid = np.all(np.logical_and(new_sample >= 0, new_sample <= 1), axis=1)
    sample[is_valid] = new_sample[is_valid]

    return sample


def _lloyd_centroidal_voronoi_tessellation(
    sample: npt.ArrayLike,
    *,
    tol: DecimalNumber = 1e-5,
    maxiter: IntNumber = 10,
    qhull_options: str | None = None,
    **kwargs: dict
) -> np.ndarray:
    """Approximate Centroidal Voronoi Tessellation.

    Perturb samples in N-dimensions using Lloyd-Max algorithm.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on. With ``n`` the number of samples and ``d``
        the dimension. Samples must be in :math:`[0, 1]^d`, with ``d>=2``.
    tol : float, optional
        Tolerance for termination. If the min of the L1-norm over the samples
        changes less than `tol`, it stops the algorithm. Default is 1e-5.
    maxiter : int, optional
        Maximum number of iterations. It will stop the algorithm even if
        `tol` is above the threshold.
        Too many iterations tend to cluster the samples as a hypersphere.
        Default is 10.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after being processed by Lloyd-Max algorithm.

    Notes
    -----
    Lloyd-Max algorithm is an iterative process with the purpose of improving
    the dispersion of samples. For given sample: (i) compute a Voronoi
    Tessellation; (ii) find the centroid of each Voronoi cell; (iii) move the
    samples toward the centroid of their respective cell. See [1]_, [2]_.

    A relaxation factor is used to control how fast samples can move at each
    iteration. This factor is starting at 2 and ending at 1 after `maxiter`
    following an exponential decay.

    The process converges to equally spaced samples. It implies that measures
    like the discrepancy could suffer from too many iterations. On the other
    hand, L1 and L2 distances should improve. This is especially true with
    QMC methods which tend to favor the discrepancy over other criteria.

    .. note::

        The current implementation does not intersect the Voronoi Tessellation
        with the boundaries. This implies that for a low number of samples,
        empirically below 20, no Voronoi cell is touching the boundaries.
        Hence, samples cannot be moved close to the boundaries.

        Further improvements could consider the samples at infinity so that
        all boundaries are segments of some Voronoi cells. This would fix
        the computation of the centroid position.

    .. warning::

       The Voronoi Tessellation step is expensive and quickly becomes
       intractable with dimensions as low as 10 even for a sample
       of size as low as 1000.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Lloyd. "Least Squares Quantization in PCM".
       IEEE Transactions on Information Theory, 1982.
    .. [2] Max J. "Quantizing for minimum distortion".
       IEEE Transactions on Information Theory, 1960.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import distance
    >>> rng = np.random.default_rng()
    >>> sample = rng.random((128, 2))

    .. note::

        The samples need to be in :math:`[0, 1]^d`. `scipy.stats.qmc.scale`
        can be used to scale the samples from their
        original bounds to :math:`[0, 1]^d`. And back to their original bounds.

    Compute the quality of the sample using the L1 criterion.

    >>> def l1_norm(sample):
    ...    return distance.pdist(sample, 'cityblock').min()

    >>> l1_norm(sample)
    0.00161...  # random

    Now process the sample using Lloyd's algorithm and check the improvement
    on the L1. The value should increase.

    >>> sample = _lloyd_centroidal_voronoi_tessellation(sample)
    >>> l1_norm(sample)
    0.0278...  # random

    """
    del kwargs  # only use keywords which are defined, needed by factory

    sample = np.asarray(sample).copy()

    if not sample.ndim == 2:
        raise ValueError('`sample` is not a 2D array')

    if not sample.shape[1] >= 2:
        raise ValueError('`sample` dimension is not >= 2')

    # Checking that sample is within the hypercube
    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError('`sample` is not in unit hypercube')

    if qhull_options is None:
        qhull_options = 'Qbb Qc Qz QJ'

        if sample.shape[1] >= 5:
            qhull_options += ' Qx'

    # Fit an exponential to be 2 at 0 and 1 at `maxiter`.
    # The decay is used for relaxation.
    # analytical solution for y=exp(-maxiter/x) - 0.1
    root = -maxiter / np.log(0.1)
    decay = [np.exp(-x / root)+0.9 for x in range(maxiter)]

    l1_old = _l1_norm(sample=sample)
    for i in range(maxiter):
        sample = _lloyd_iteration(
                sample=sample, decay=decay[i],
                qhull_options=qhull_options,
        )

        l1_new = _l1_norm(sample=sample)

        if abs(l1_new - l1_old) < tol:
            break
        else:
            l1_old = l1_new

    return sample


def _validate_workers(workers: IntNumber = 1) -> IntNumber:
    """Validate `workers` based on platform and value.

    Parameters
    ----------
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    Workers : int
        Number of CPU used by the algorithm

    """
    workers = int(workers)
    if workers == -1:
        workers = os.cpu_count()  # type: ignore[assignment]
        if workers is None:
            raise NotImplementedError(
                "Cannot determine the number of cpus using os.cpu_count(), "
                "cannot use -1 for the number of workers"
            )
    elif workers <= 0:
        raise ValueError(f"Invalid number of workers: {workers}, must be -1 "
                         "or > 0")

    return workers


def _validate_bounds(
    l_bounds: npt.ArrayLike, u_bounds: npt.ArrayLike, d: int
) -> tuple[np.ndarray, ...]:
    """Bounds input validation.

    Parameters
    ----------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.
    d : int
        Dimension to use for broadcasting.

    Returns
    -------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.

    """
    try:
        lower = np.broadcast_to(l_bounds, d)
        upper = np.broadcast_to(u_bounds, d)
    except ValueError as exc:
        msg = ("'l_bounds' and 'u_bounds' must be broadcastable and respect"
               " the sample dimension")
        raise ValueError(msg) from exc

    if not np.all(lower < upper):
        raise ValueError("Bounds are not consistent 'l_bounds' < 'u_bounds'")

    return lower, upper
