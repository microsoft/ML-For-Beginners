
import warnings

import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special

# TODO:
# * actually solve (31) of Blinnikov & Moessner
# * numerical stability: multiply factorials in logspace?
# * ppf & friends: Cornish & Fisher series, or tabulate/solve


_faa_di_bruno_cache = {
        1: [[(1, 1)]],
        2: [[(1, 2)], [(2, 1)]],
        3: [[(1, 3)], [(2, 1), (1, 1)], [(3, 1)]],
        4: [[(1, 4)], [(1, 2), (2, 1)], [(2, 2)], [(3, 1), (1, 1)], [(4, 1)]]}


def _faa_di_bruno_partitions(n):
    """
    Return all non-negative integer solutions of the diophantine equation

            n*k_n + ... + 2*k_2 + 1*k_1 = n   (1)

    Parameters
    ----------
    n : int
        the r.h.s. of Eq. (1)

    Returns
    -------
    partitions : list
        Each solution is itself a list of the form `[(m, k_m), ...]`
        for non-zero `k_m`. Notice that the index `m` is 1-based.

    Examples:
    ---------
    >>> _faa_di_bruno_partitions(2)
    [[(1, 2)], [(2, 1)]]
    >>> for p in _faa_di_bruno_partitions(4):
    ...     assert 4 == sum(m * k for (m, k) in p)
    """
    if n < 1:
        raise ValueError("Expected a positive integer; got %s instead" % n)
    try:
        return _faa_di_bruno_cache[n]
    except KeyError:
        # TODO: higher order terms
        # solve Eq. (31) from Blinninkov & Moessner here
        raise NotImplementedError('Higher order terms not yet implemented.')


def cumulant_from_moments(momt, n):
    """Compute n-th cumulant given moments.

    Parameters
    ----------
    momt : array_like
        `momt[j]` contains `(j+1)`-th moment.
        These can be raw moments around zero, or central moments
        (in which case, `momt[0]` == 0).
    n : int
        which cumulant to calculate (must be >1)

    Returns
    -------
    kappa : float
        n-th cumulant.
    """
    if n < 1:
        raise ValueError("Expected a positive integer. Got %s instead." % n)
    if len(momt) < n:
        raise ValueError("%s-th cumulant requires %s moments, "
                         "only got %s." % (n, n, len(momt)))
    kappa = 0.
    for p in _faa_di_bruno_partitions(n):
        r = sum(k for (m, k) in p)
        term = (-1)**(r - 1) * factorial(r - 1)
        for (m, k) in p:
            term *= np.power(momt[m - 1] / factorial(m), k) / factorial(k)
        kappa += term
    kappa *= factorial(n)
    return kappa

## copied from scipy.stats.distributions to avoid the overhead of
## the public methods
_norm_pdf_C = np.sqrt(2*np.pi)
def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C

def _norm_cdf(x):
    return special.ndtr(x)

def _norm_sf(x):
    return special.ndtr(-x)


class ExpandedNormal(rv_continuous):
    """Construct the Edgeworth expansion pdf given cumulants.

    Parameters
    ----------
    cum : array_like
        `cum[j]` contains `(j+1)`-th cumulant: cum[0] is the mean,
        cum[1] is the variance and so on.

    Notes
    -----
    This is actually an asymptotic rather than convergent series, hence
    higher orders of the expansion may or may not improve the result.
    In a strongly non-Gaussian case, it is possible that the density
    becomes negative, especially far out in the tails.

    Examples
    --------
    Construct the 4th order expansion for the chi-square distribution using
    the known values of the cumulants:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> from scipy.special import factorial
    >>> df = 12
    >>> chi2_c = [2**(j-1) * factorial(j-1) * df for j in range(1, 5)]
    >>> edgw_chi2 = ExpandedNormal(chi2_c, name='edgw_chi2', momtype=0)

    Calculate several moments:
    >>> m, v = edgw_chi2.stats(moments='mv')
    >>> np.allclose([m, v], [df, 2 * df])
    True

    Plot the density function:
    >>> mu, sigma = df, np.sqrt(2*df)
    >>> x = np.linspace(mu - 3*sigma, mu + 3*sigma)
    >>> fig1 = plt.plot(x, stats.chi2.pdf(x, df=df), 'g-', lw=4, alpha=0.5)
    >>> fig2 = plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', lw=4, alpha=0.5)
    >>> fig3 = plt.plot(x, edgw_chi2.pdf(x), 'r-', lw=2)
    >>> plt.show()

    References
    ----------
    .. [*] E.A. Cornish and R.A. Fisher, Moments and cumulants in the
         specification of distributions, Revue de l'Institut Internat.
         de Statistique. 5: 307 (1938), reprinted in
         R.A. Fisher, Contributions to Mathematical Statistics. Wiley, 1950.
    .. [*] https://en.wikipedia.org/wiki/Edgeworth_series
    .. [*] S. Blinnikov and R. Moessner, Expansions for nearly Gaussian
        distributions, Astron. Astrophys. Suppl. Ser. 130, 193 (1998)
    """
    def __init__(self, cum, name='Edgeworth expanded normal', **kwds):
        if len(cum) < 2:
            raise ValueError("At least two cumulants are needed.")
        self._coef, self._mu, self._sigma = self._compute_coefs_pdf(cum)
        self._herm_pdf = HermiteE(self._coef)
        if self._coef.size > 2:
            self._herm_cdf = HermiteE(-self._coef[1:])
        else:
            self._herm_cdf = lambda x: 0.

        # warn if pdf(x) < 0 for some values of x within 4 sigma
        r = np.real_if_close(self._herm_pdf.roots())
        r = (r - self._mu) / self._sigma
        if r[(np.imag(r) == 0) & (np.abs(r) < 4)].any():
            mesg = 'PDF has zeros at %s ' % r
            warnings.warn(mesg, RuntimeWarning)

        kwds.update({'name': name,
                     'momtype': 0})   # use pdf, not ppf in self.moment()
        super(ExpandedNormal, self).__init__(**kwds)

    def _pdf(self, x):
        y = (x - self._mu) / self._sigma
        return self._herm_pdf(y) * _norm_pdf(y) / self._sigma

    def _cdf(self, x):
        y = (x - self._mu) / self._sigma
        return (_norm_cdf(y) +
                self._herm_cdf(y) * _norm_pdf(y))

    def _sf(self, x):
        y = (x - self._mu) / self._sigma
        return (_norm_sf(y) -
                self._herm_cdf(y) * _norm_pdf(y))

    def _compute_coefs_pdf(self, cum):
        # scale cumulants by \sigma
        mu, sigma = cum[0], np.sqrt(cum[1])
        lam = np.asarray(cum)
        for j, l in enumerate(lam):
            lam[j] /= cum[1]**j

        coef = np.zeros(lam.size * 3 - 5)
        coef[0] = 1.
        for s in range(lam.size - 2):
            for p in _faa_di_bruno_partitions(s+1):
                term = sigma**(s+1)
                for (m, k) in p:
                    term *= np.power(lam[m+1] / factorial(m+2), k) / factorial(k)
                r = sum(k for (m, k) in p)
                coef[s + 1 + 2*r] += term
        return coef, mu, sigma
