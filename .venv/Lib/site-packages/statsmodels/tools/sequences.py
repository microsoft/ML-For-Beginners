"""Low discrepancy sequence tools."""
import numpy as np


def discrepancy(sample, bounds=None):
    """Discrepancy.

    Compute the centered discrepancy on a given sample.
    It is a measure of the uniformity of the points in the parameter space.
    The lower the value is, the better the coverage of the parameter space is.

    Parameters
    ----------
    sample : array_like (n_samples, k_vars)
        The sample to compute the discrepancy from.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.

    Returns
    -------
    discrepancy : float
        Centered discrepancy.

    References
    ----------
    [1] Fang et al. "Design and modeling for computer experiments",
      Computer Science and Data Analysis Series Science and Data Analysis
      Series, 2006.
    """
    sample = np.asarray(sample)
    n_sample, dim = sample.shape

    # Sample scaling from bounds to unit hypercube
    if bounds is not None:
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)
        sample = (sample - min_) / (max_ - min_)

    abs_ = abs(sample - 0.5)
    disc1 = np.sum(np.prod(1 + 0.5 * abs_ - 0.5 * abs_ ** 2, axis=1))

    prod_arr = 1
    for i in range(dim):
        s0 = sample[:, i]
        prod_arr *= (1 +
                     0.5 * abs(s0[:, None] - 0.5) + 0.5 * abs(s0 - 0.5) -
                     0.5 * abs(s0[:, None] - s0))
    disc2 = prod_arr.sum()

    c2 = ((13.0 / 12.0) ** dim - 2.0 / n_sample * disc1 +
          1.0 / (n_sample ** 2) * disc2)

    return c2


def primes_from_2_to(n):
    """Prime numbers from 2 to *n*.

    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.

    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.

    References
    ----------
    [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n):
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
              953, 967, 971, 977, 983, 991, 997][:n]

    if len(primes) < n:
        big_number = 10
        while 'Not enought primes':
            primes = primes_from_2_to(big_number)[:n]
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def van_der_corput(n_sample, base=2, start_index=0):
    """Van der Corput sequence.

    Pseudo-random number generator based on a b-adic expansion.

    Parameters
    ----------
    n_sample : int
        Number of element of the sequence.
    base : int
        Base of the sequence.
    start_index : int
        Index to start the sequence from.

    Returns
    -------
    sequence : list (n_samples,)
        Sequence of Van der Corput.
    """
    sequence = []
    for i in range(start_index, start_index + n_sample):
        n_th_number, denom = 0., 1.
        quotient = i
        while quotient > 0:
            quotient, remainder = divmod(quotient, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample, bounds=None, start_index=0):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. Halton sequence use base-two Van der Corput
    sequence for the first dimension, base-three for its second and base-n for
    its n-dimension.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_sample : int
        Number of samples to generate in the parametr space.
    bounds : tuple or array_like ([min, k_vars], [max, k_vars])
        Desired range of transformed data. The transformation apply the bounds
        on the sample and not the theoretical space, unit cube. Thus min and
        max values of the sample will coincide with the bounds.
    start_index : int
        Index to start the sequence from.

    Returns
    -------
    sequence : array_like (n_samples, k_vars)
        Sequence of Halton.

    References
    ----------
    [1] Halton, "On the efficiency of certain quasi-random sequences of points
      in evaluating multi-dimensional integrals", Numerische Mathematik, 1960.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from statsmodels.tools import sequences
    >>> sample = sequences.halton(dim=2, n_sample=5)

    Compute the quality of the sample using the discrepancy criterion.

    >>> uniformity = sequences.discrepancy(sample)

    If some wants to continue an existing design, extra points can be obtained.

    >>> sample_continued = sequences.halton(dim=2, n_sample=5, start_index=5)
    """
    base = n_primes(dim)

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, bdim, start_index) for bdim in base]
    sample = np.array(sample).T[1:]

    # Sample scaling from unit hypercube to feature range
    if bounds is not None:
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)
        sample = sample * (max_ - min_) + min_

    return sample
