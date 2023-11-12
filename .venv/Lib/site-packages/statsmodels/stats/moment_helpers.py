"""helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix


Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy.special import comb


def _convert_to_multidim(x):
    if any([isinstance(x, list), isinstance(x, tuple)]):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        # something strange was passed and the function probably
        # will fall, maybe insert an exception?
        return x


def _convert_from_multidim(x, totype=list):
    if len(x.shape) < 2:
        return totype(x)
    return x.T


def mc2mnc(mc):
    """convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    x = _convert_to_multidim(mc)

    def _local_counts(mc):
        mean = mc[0]
        mc = [1] + list(mc)  # add zero moment = 1
        mc[1] = 0  # define central mean as zero for formula
        mnc = [1, mean]  # zero and first raw moments
        for nn, m in enumerate(mc[2:]):
            n = nn + 2
            mnc.append(0)
            for k in range(n + 1):
                mnc[n] += comb(n, k, exact=True) * mc[k] * mean ** (n - k)
        return mnc[1:]

    res = np.apply_along_axis(_local_counts, 0, x)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res)


def mnc2mc(mnc, wmean=True):
    """convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        mean = mnc[0]
        mnc = [1] + list(mnc)  # add zero moment = 1
        mu = []
        for n, m in enumerate(mnc):
            mu.append(0)
            for k in range(n + 1):
                sgn_comb = (-1) ** (n - k) * comb(n, k, exact=True)
                mu[n] += sgn_comb * mnc[k] * mean ** (n - k)
        if wmean:
            mu[1] = mean
        return mu[1:]

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res)


def cum2mc(kappa):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    """
    X = _convert_to_multidim(kappa)

    def _local_counts(kappa):
        mc = [1, 0.0]  # _kappa[0]]  #insert 0-moment and mean
        kappa0 = kappa[0]
        kappa = [1] + list(kappa)
        for nn, m in enumerate(kappa[2:]):
            n = nn + 2
            mc.append(0)
            for k in range(n - 1):
                mc[n] += comb(n - 1, k, exact=True) * kappa[n - k] * mc[k]
        mc[1] = kappa0  # insert mean as first moments by convention
        return mc[1:]

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res)


def mnc2cum(mnc):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    """
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        mnc = [1] + list(mnc)
        kappa = [1]
        for nn, m in enumerate(mnc[1:]):
            n = nn + 1
            kappa.append(m)
            for k in range(1, n):
                num_ways = comb(n - 1, k - 1, exact=True)
                kappa[n] -= num_ways * kappa[k] * mnc[n - k]
        return kappa[1:]

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res)


def mc2cum(mc):
    """
    just chained because I have still the test case
    """
    first_step = mc2mnc(mc)
    if isinstance(first_step, np.ndarray):
        first_step = first_step.T
    return mnc2cum(first_step)
    # return np.apply_along_axis(lambda x: mnc2cum(mc2mnc(x)), 0, mc)


def mvsk2mc(args):
    """convert mean, variance, skew, kurtosis to central moments"""
    X = _convert_to_multidim(args)

    def _local_counts(args):
        mu, sig2, sk, kur = args
        cnt = [None] * 4
        cnt[0] = mu
        cnt[1] = sig2
        cnt[2] = sk * sig2 ** 1.5
        cnt[3] = (kur + 3.0) * sig2 ** 2.0
        return tuple(cnt)

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res, tuple)


def mvsk2mnc(args):
    """convert mean, variance, skew, kurtosis to non-central moments"""
    X = _convert_to_multidim(args)

    def _local_counts(args):
        mc, mc2, skew, kurt = args
        mnc = mc
        mnc2 = mc2 + mc * mc
        mc3 = skew * (mc2 ** 1.5)  # 3rd central moment
        mnc3 = mc3 + 3 * mc * mc2 + mc ** 3  # 3rd non-central moment
        mc4 = (kurt + 3.0) * (mc2 ** 2.0)  # 4th central moment
        mnc4 = mc4 + 4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4
        return (mnc, mnc2, mnc3, mnc4)

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res, tuple)


def mc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis"""
    X = _convert_to_multidim(args)

    def _local_counts(args):
        mc, mc2, mc3, mc4 = args
        skew = np.divide(mc3, mc2 ** 1.5)
        kurt = np.divide(mc4, mc2 ** 2.0) - 3.0
        return (mc, mc2, skew, kurt)

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res, tuple)


def mnc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis
    """
    X = _convert_to_multidim(args)

    def _local_counts(args):
        # convert four non-central moments to central moments
        mnc, mnc2, mnc3, mnc4 = args
        mc = mnc
        mc2 = mnc2 - mnc * mnc
        mc3 = mnc3 - (3 * mc * mc2 + mc ** 3)  # 3rd central moment
        mc4 = mnc4 - (4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4)
        return mc2mvsk((mc, mc2, mc3, mc4))

    res = np.apply_along_axis(_local_counts, 0, X)
    # for backward compatibility convert 1-dim output to list/tuple
    return _convert_from_multidim(res, tuple)

# def mnc2mc(args):
#    """convert four non-central moments to central moments
#    """
#    mnc, mnc2, mnc3, mnc4 = args
#    mc = mnc
#    mc2 = mnc2 - mnc*mnc
#    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
#    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)
#    return mc, mc2, mc

# TODO: no return, did it get lost in cut-paste?


def cov2corr(cov, return_std=False):
    """
    convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that
    division is defined elementwise. np.ma.array and np.matrix are allowed.
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr


def corr2cov(corr, std):
    """
    convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.
    """
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov


def se_cov(cov):
    """
    get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    Parameters
    ----------
    cov : array_like, square
        covariance matrix

    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov
    """
    return np.sqrt(np.diag(cov))
