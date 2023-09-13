# Integration of multivariate normal and t distributions.

# Adapted from the MATLAB original implementations by Dr. Alan Genz.

#     http://www.math.wsu.edu/faculty/genz/software/software.html

# Copyright (C) 2013, Alan Genz,  All rights reserved.
# Python implementation is copyright (C) 2022, Robert Kern,  All rights
# reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided the following conditions are met:
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#   3. The contributor name(s) may not be used to endorse or promote
#      products derived from this software without specific prior
#      written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to


phi = ndtr
phinv = ndtri


def _factorize_int(n):
    """Return a sorted list of the unique prime factors of a positive integer.
    """
    # NOTE: There are lots faster ways to do this, but this isn't terrible.
    factors = set()
    for p in primes_from_2_to(int(np.sqrt(n)) + 1):
        while not (n % p):
            factors.add(p)
            n //= p
        if n == 1:
            break
    if n != 1:
        factors.add(n)
    return sorted(factors)


def _primitive_root(p):
    """Compute a primitive root of the prime number `p`.

    Used in the CBC lattice construction.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    """
    # p is prime
    pm = p - 1
    factors = _factorize_int(pm)
    n = len(factors)
    r = 2
    k = 0
    while k < n:
        d = pm // factors[k]
        # pow() doesn't like numpy scalar types.
        rd = pow(int(r), int(d), int(p))
        if rd == 1:
            r += 1
            k = 0
        else:
            k += 1
    return r


def _cbc_lattice(n_dim, n_qmc_samples):
    """Compute a QMC lattice generator using a Fast CBC construction.

    Parameters
    ----------
    n_dim : int > 0
        The number of dimensions for the lattice.
    n_qmc_samples : int > 0
        The desired number of QMC samples. This will be rounded down to the
        nearest prime to enable the CBC construction.

    Returns
    -------
    q : float array : shape=(n_dim,)
        The lattice generator vector. All values are in the open interval
        `(0, 1)`.
    actual_n_qmc_samples : int
        The prime number of QMC samples that must be used with this lattice,
        no more, no less.

    References
    ----------
    .. [1] Nuyens, D. and Cools, R. "Fast Component-by-Component Construction,
           a Reprise for Different Kernels", In H. Niederreiter and D. Talay,
           editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
           Springer-Verlag, 2006, 371-385.
    """
    # Round down to the nearest prime number.
    primes = primes_from_2_to(n_qmc_samples + 1)
    n_qmc_samples = primes[-1]

    bt = np.ones(n_dim)
    gm = np.hstack([1.0, 0.8 ** np.arange(n_dim - 1)])
    q = 1
    w = 0
    z = np.arange(1, n_dim + 1)
    m = (n_qmc_samples - 1) // 2
    g = _primitive_root(n_qmc_samples)
    # Slightly faster way to compute perm[j] = pow(g, j, n_qmc_samples)
    # Shame that we don't have modulo pow() implemented as a ufunc.
    perm = np.ones(m, dtype=int)
    for j in range(m - 1):
        perm[j + 1] = (g * perm[j]) % n_qmc_samples
    perm = np.minimum(n_qmc_samples - perm, perm)
    pn = perm / n_qmc_samples
    c = pn * pn - pn + 1.0 / 6
    fc = fft(c)
    for s in range(1, n_dim):
        reordered = np.hstack([
            c[:w+1][::-1],
            c[w+1:m][::-1],
        ])
        q = q * (bt[s-1] + gm[s-1] * reordered)
        w = ifft(fc * fft(q)).real.argmin()
        z[s] = perm[w]
    q = z / n_qmc_samples
    return q, n_qmc_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the development of a parameter for users
# to set the desired CDF accuracy, but must be reviewed and tested before use.
def _qauto(func, covar, low, high, rng, error=1e-3, limit=10_000, **kwds):
    """Automatically rerun the integration to get the required error bound.

    Parameters
    ----------
    func : callable
        Either :func:`_qmvn` or :func:`_qmvt`.
    covar, low, high : array
        As specified in :func:`_qmvn` and :func:`_qmvt`.
    rng : Generator, optional
        default_rng(), yada, yada
    error : float > 0
        The desired error bound.
    limit : int > 0:
        The rough limit of the number of integration points to consider. The
        integration will stop looping once this limit has been *exceeded*.
    **kwds :
        Other keyword arguments to pass to `func`. When using :func:`_qmvt`, be
        sure to include ``nu=`` as one of these.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    n_samples : int
        The number of integration points actually used.
    """
    n = len(covar)
    n_samples = 0
    if n == 1:
        prob = phi(high) - phi(low)
        # More or less
        est_error = 1e-15
    else:
        mi = min(limit, n * 1000)
        prob = 0.0
        est_error = 1.0
        ei = 0.0
        while est_error > error and n_samples < limit:
            mi = round(np.sqrt(2) * mi)
            pi, ei, ni = func(mi, covar, low, high, rng=rng, **kwds)
            n_samples += ni
            wt = 1.0 / (1 + (ei / est_error)**2)
            prob += wt * (pi - prob)
            est_error = np.sqrt(wt) * ei
    return prob, est_error, n_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _qmvn(m, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate normal integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n - 1, max(m // n_batches, 1))
    y = np.zeros((n - 1, n_qmc_samples))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i - 1] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples


# Note: this function is not currently used or tested by any SciPy code. It is
# included in this file to facilitate the resolution of gh-8367, gh-16142, and
# possibly gh-14286, but must be reviewed and tested before use.
def _mvn_qmc_integrand(covar, low, high, use_tent=False):
    """Transform the multivariate normal integration into a QMC integrand over
    a unit hypercube.

    The dimensionality of the resulting hypercube integration domain is one
    less than the dimensionality of the original integrand. Note that this
    transformation subsumes the integration bounds in order to account for
    infinite bounds. The QMC integration one does with the returned integrand
    should be on the unit hypercube.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    use_tent : bool, optional
        If True, then use tent periodization. Only helpful for lattice rules.

    Returns
    -------
    integrand : Callable[[NDArray], NDArray]
        The QMC-integrable integrand. It takes an
        ``(n_qmc_samples, ndim_integrand)`` array of QMC samples in the unit
        hypercube and returns the ``(n_qmc_samples,)`` evaluations of at these
        QMC points.
    ndim_integrand : int
        The dimensionality of the integrand. Equal to ``n-1``.
    """
    cho, lo, hi = _permuted_cholesky(covar, low, high)
    n = cho.shape[0]
    ndim_integrand = n - 1
    ct = cho[0, 0]
    c = phi(lo[0] / ct)
    d = phi(hi[0] / ct)
    ci = c
    dci = d - ci

    def integrand(*zs):
        ndim_qmc = len(zs)
        n_qmc_samples = len(np.atleast_1d(zs[0]))
        assert ndim_qmc == ndim_integrand
        y = np.zeros((ndim_qmc, n_qmc_samples))
        c = np.full(n_qmc_samples, ci)
        dc = np.full(n_qmc_samples, dci)
        pv = dc.copy()
        for i in range(1, n):
            if use_tent:
                # Tent periodization transform.
                x = abs(2 * zs[i-1] - 1)
            else:
                x = zs[i-1]
            y[i - 1, :] = phinv(c + x * dc)
            s = cho[i, :i] @ y[:i, :]
            ct = cho[i, i]
            c = phi((lo[i] - s) / ct)
            d = phi((hi[i] - s) / ct)
            dc = d - c
            pv = pv * dc
        return pv

    return integrand, ndim_integrand


def _qmvt(m, nu, covar, low, high, rng, lattice='cbc', n_batches=10):
    """Multivariate t integration over box bounds.

    Parameters
    ----------
    m : int > n_batches
        The number of points to sample. This number will be divided into
        `n_batches` batches that apply random offsets of the sampling lattice
        for each batch in order to estimate the error.
    nu : float >= 0
        The shape parameter of the multivariate t distribution.
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    rng : Generator, optional
        default_rng(), yada, yada
    lattice : 'cbc' or callable
        The type of lattice rule to use to construct the integration points.
    n_batches : int > 0, optional
        The number of QMC batches to apply.

    Returns
    -------
    prob : float
        The estimated probability mass within the bounds.
    est_error : float
        3 times the standard error of the batch estimates.
    n_samples : int
        The number of samples actually used.
    """
    sn = max(1.0, np.sqrt(nu))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    cho, lo, hi = _permuted_cholesky(covar, low / sn, high / sn)
    n = cho.shape[0]
    prob = 0.0
    error_var = 0.0
    q, n_qmc_samples = _cbc_lattice(n, max(m // n_batches, 1))
    i_samples = np.arange(n_qmc_samples) + 1
    for j in range(n_batches):
        pv = np.ones(n_qmc_samples)
        s = np.zeros((n, n_qmc_samples))
        for i in range(n):
            # Pseudorandomly-shifted lattice coordinate.
            z = q[i] * i_samples + rng.random()
            # Fast remainder(z, 1.0)
            z -= z.astype(int)
            # Tent periodization transform.
            x = abs(2 * z - 1)
            # FIXME: Lift the i==0 case out of the loop to make the logic
            # easier to follow.
            if i == 0:
                # We'll use one of the QR variates to pull out the
                # t-distribution scaling.
                if nu > 0:
                    r = np.sqrt(2 * gammaincinv(nu / 2, x))
                else:
                    r = np.ones_like(x)
            else:
                y = phinv(c + x * dc)  # noqa: F821
                with np.errstate(invalid='ignore'):
                    s[i:, :] += cho[i:, i - 1][:, np.newaxis] * y
            si = s[i, :]

            c = np.ones(n_qmc_samples)
            d = np.ones(n_qmc_samples)
            with np.errstate(invalid='ignore'):
                lois = lo[i] * r - si
                hiis = hi[i] * r - si
            c[lois < -9] = 0.0
            d[hiis < -9] = 0.0
            lo_mask = abs(lois) < 9
            hi_mask = abs(hiis) < 9
            c[lo_mask] = phi(lois[lo_mask])
            d[hi_mask] = phi(hiis[hi_mask])

            dc = d - c
            pv *= dc

        # Accumulate the mean and error variances with online formulations.
        d = (pv.mean() - prob) / (j + 1)
        prob += d
        error_var = (j - 1) * error_var / (j + 1) + d * d
    # Error bounds are 3 times the standard error of the estimates.
    est_error = 3 * np.sqrt(error_var)
    n_samples = n_qmc_samples * n_batches
    return prob, est_error, n_samples


def _permuted_cholesky(covar, low, high, tol=1e-10):
    """Compute a scaled, permuted Cholesky factor, with integration bounds.

    The scaling and permuting of the dimensions accomplishes part of the
    transformation of the original integration problem into a more numerically
    tractable form. The lower-triangular Cholesky factor will then be used in
    the subsequent integration. The integration bounds will be scaled and
    permuted as well.

    Parameters
    ----------
    covar : (n, n) float array
        Possibly singular, positive semidefinite symmetric covariance matrix.
    low, high : (n,) float array
        The low and high integration bounds.
    tol : float, optional
        The singularity tolerance.

    Returns
    -------
    cho : (n, n) float array
        Lower Cholesky factor, scaled and permuted.
    new_low, new_high : (n,) float array
        The scaled and permuted low and high integration bounds.
    """
    # Make copies for outputting.
    cho = np.array(covar, dtype=np.float64)
    new_lo = np.array(low, dtype=np.float64)
    new_hi = np.array(high, dtype=np.float64)
    n = cho.shape[0]
    if cho.shape != (n, n):
        raise ValueError("expected a square symmetric array")
    if new_lo.shape != (n,) or new_hi.shape != (n,):
        raise ValueError(
            "expected integration boundaries the same dimensions "
            "as the covariance matrix"
        )
    # Scale by the sqrt of the diagonal.
    dc = np.sqrt(np.maximum(np.diag(cho), 0.0))
    # But don't divide by 0.
    dc[dc == 0.0] = 1.0
    new_lo /= dc
    new_hi /= dc
    cho /= dc
    cho /= dc[:, np.newaxis]

    y = np.zeros(n)
    sqtp = np.sqrt(2 * np.pi)
    for k in range(n):
        epk = (k + 1) * tol
        im = k
        ck = 0.0
        dem = 1.0
        s = 0.0
        lo_m = 0.0
        hi_m = 0.0
        for i in range(k, n):
            if cho[i, i] > tol:
                ci = np.sqrt(cho[i, i])
                if i > 0:
                    s = cho[i, :k] @ y[:k]
                lo_i = (new_lo[i] - s) / ci
                hi_i = (new_hi[i] - s) / ci
                de = phi(hi_i) - phi(lo_i)
                if de <= dem:
                    ck = ci
                    dem = de
                    lo_m = lo_i
                    hi_m = hi_i
                    im = i
        if im > k:
            # Swap im and k
            cho[im, im] = cho[k, k]
            _swap_slices(cho, np.s_[im, :k], np.s_[k, :k])
            _swap_slices(cho, np.s_[im + 1:, im], np.s_[im + 1:, k])
            _swap_slices(cho, np.s_[k + 1:im, k], np.s_[im, k + 1:im])
            _swap_slices(new_lo, k, im)
            _swap_slices(new_hi, k, im)
        if ck > epk:
            cho[k, k] = ck
            cho[k, k + 1:] = 0.0
            for i in range(k + 1, n):
                cho[i, k] /= ck
                cho[i, k + 1:i + 1] -= cho[i, k] * cho[k + 1:i + 1, k]
            if abs(dem) > tol:
                y[k] = ((np.exp(-lo_m * lo_m / 2) - np.exp(-hi_m * hi_m / 2)) /
                        (sqtp * dem))
            else:
                y[k] = (lo_m + hi_m) / 2
                if lo_m < -10:
                    y[k] = hi_m
                elif hi_m > 10:
                    y[k] = lo_m
            cho[k, :k + 1] /= ck
            new_lo[k] /= ck
            new_hi[k] /= ck
        else:
            cho[k:, k] = 0.0
            y[k] = (new_lo[k] + new_hi[k]) / 2
    return cho, new_lo, new_hi


def _swap_slices(x, slc1, slc2):
    t = x[slc1].copy()
    x[slc1] = x[slc2].copy()
    x[slc2] = t
