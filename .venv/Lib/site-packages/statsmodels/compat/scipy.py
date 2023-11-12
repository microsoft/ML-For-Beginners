from packaging.version import Version, parse

import numpy as np
import scipy

SP_VERSION = parse(scipy.__version__)
SP_LT_15 = SP_VERSION < Version("1.4.99")
SCIPY_GT_14 = not SP_LT_15
SP_LT_16 = SP_VERSION < Version("1.5.99")
SP_LT_17 = SP_VERSION < Version("1.6.99")
SP_LT_19 = SP_VERSION < Version("1.8.99")


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _valarray(shape, value=np.nan, typecode=None):
    """Return an array of all value."""

    out = np.ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


if SP_LT_16:
    # copied from scipy, added to scipy in 1.6.0
    from ._scipy_multivariate_t import multivariate_t  # noqa: F401
else:
    from scipy.stats import multivariate_t  # noqa: F401
