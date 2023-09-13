# flake8: noqa
import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to


def _primes(n):
    # Defined to facilitate comparison between translation and source
    # In Matlab, primes(10.5) -> first four primes, primes(11.5) -> first five
    return primes_from_2_to(math.ceil(n))


def _gaminv(a, b):
    # Defined to facilitate comparison between translation and source
    # Matlab's `gaminv` is like `special.gammaincinv` but args are reversed
    return special.gammaincinv(b, a)


def _qsimvtv(m, nu, sigma, a, b, rng):
    """Estimates the multivariate t CDF using randomized QMC

    Parameters
    ----------
    m : int
        The number of points
    nu : float
        Degrees of freedom
    sigma : ndarray
        A 2D positive semidefinite covariance matrix
    a : ndarray
        Lower integration limits
    b : ndarray
        Upper integration limits.
    rng : Generator
        Pseudorandom number generator

    Returns
    -------
    p : float
        The estimated CDF.
    e : float
        An absolute error estimate.

    """
    # _qsimvtv is a Python translation of the Matlab function qsimvtv,
    # semicolons and all.
    #
    #   This function uses an algorithm given in the paper
    #      "Comparison of Methods for the Numerical Computation of
    #       Multivariate t Probabilities", in
    #      J. of Computational and Graphical Stat., 11(2002), pp. 950-971, by
    #          Alan Genz and Frank Bretz
    #
    #   The primary references for the numerical integration are
    #    "On a Number-Theoretical Integration Method"
    #    H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11.
    #    and
    #    "Randomization of Number Theoretic Methods for Multiple Integration"
    #     R. Cranley & T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
    #
    #   Alan Genz is the author of this function and following Matlab functions.
    #          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
    #          Email : alangenz@wsu.edu
    #
    # Copyright (C) 2013, Alan Genz,  All rights reserved.
    #
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

    # Initialization
    sn = max(1, math.sqrt(nu)); ch, az, bz = _chlrps(sigma, a/sn, b/sn)
    n = len(sigma); N = 10; P = math.ceil(m/N); on = np.ones(P); p = 0; e = 0
    ps = np.sqrt(_primes(5*n*math.log(n+4)/4)); q = ps[:, np.newaxis]  # Richtmyer gens.

    # Randomization loop for ns samples
    c = None; dc = None
    for S in range(N):
        vp = on.copy(); s = np.zeros((n, P))
        for i in range(n):
            x = np.abs(2*np.mod(q[i]*np.arange(1, P+1) + rng.random(), 1)-1)  # periodizing transform
            if i == 0:
                r = on
                if nu > 0:
                    r = np.sqrt(2*_gaminv(x, nu/2))
            else:
                y = _Phinv(c + x*dc)
                s[i:] += ch[i:, i-1:i] * y
            si = s[i, :]; c = on.copy(); ai = az[i]*r - si; d = on.copy(); bi = bz[i]*r - si
            c[ai <= -9] = 0; tl = abs(ai) < 9; c[tl] = _Phi(ai[tl])
            d[bi <= -9] = 0; tl = abs(bi) < 9; d[tl] = _Phi(bi[tl])
            dc = d - c; vp = vp * dc
        d = (np.mean(vp) - p)/(S + 1); p = p + d; e = (S - 1)*e/(S + 1) + d**2
    e = math.sqrt(e)  # error estimate is 3 times std error with N samples.
    return p, e


#  Standard statistical normal distribution functions
def _Phi(z):
    return special.ndtr(z)


def _Phinv(p):
    return special.ndtri(p)


def _chlrps(R, a, b):
    """
    Computes permuted and scaled lower Cholesky factor c for R which may be
    singular, also permuting and scaling integration limit vectors a and b.
    """
    ep = 1e-10  # singularity tolerance
    eps = np.finfo(R.dtype).eps

    n = len(R); c = R.copy(); ap = a.copy(); bp = b.copy(); d = np.sqrt(np.maximum(np.diag(c), 0))
    for i in range(n):
        if d[i] > 0:
            c[:, i] /= d[i]; c[i, :] /= d[i]
            ap[i] /= d[i]; bp[i] /= d[i]
    y = np.zeros((n, 1)); sqtp = math.sqrt(2*math.pi)

    for k in range(n):
        im = k; ckk = 0; dem = 1; s = 0
        for i in range(k, n):
            if c[i, i] > eps:
                cii = math.sqrt(max(c[i, i], 0))
                if i > 0: s = c[i, :k] @ y[:k]
                ai = (ap[i]-s)/cii; bi = (bp[i]-s)/cii; de = _Phi(bi)-_Phi(ai)
                if de <= dem:
                    ckk = cii; dem = de; am = ai; bm = bi; im = i
        if im > k:
            ap[[im, k]] = ap[[k, im]]; bp[[im, k]] = bp[[k, im]]; c[im, im] = c[k, k]
            t = c[im, :k].copy(); c[im, :k] = c[k, :k]; c[k, :k] = t
            t = c[im+1:, im].copy(); c[im+1:, im] = c[im+1:, k]; c[im+1:, k] = t
            t = c[k+1:im, k].copy(); c[k+1:im, k] = c[im, k+1:im].T; c[im, k+1:im] = t.T
        if ckk > ep*(k+1):
            c[k, k] = ckk; c[k, k+1:] = 0
            for i in range(k+1, n):
                c[i, k] = c[i, k]/ckk; c[i, k+1:i+1] = c[i, k+1:i+1] - c[i, k]*c[k+1:i+1, k].T
            if abs(dem) > ep:
                y[k] = (np.exp(-am**2/2) - np.exp(-bm**2/2)) / (sqtp*dem)
            else:
                y[k] = (am + bm) / 2
                if am < -10:
                    y[k] = bm
                elif bm > 10:
                    y[k] = am
            c[k, :k+1] /= ckk; ap[k] /= ckk; bp[k] /= ckk
        else:
            c[k:, k] = 0; y[k] = (ap[k] + bp[k])/2
        pass
    return c, ap, bp
