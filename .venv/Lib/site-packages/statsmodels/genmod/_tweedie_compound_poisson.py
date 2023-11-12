"""
Private experimental module for miscellaneous Tweedie functions.

References
----------

Dunn, Peter K. and Smyth,  Gordon K. 2001. Tweedie family densities: methods of
    evaluation. In Proceedings of the 16th International Workshop on
    Statistical Modelling, Odense, Denmark, 2–6 July.

Jørgensen, B., Demétrio, C.G.B., Kristensen, E., Banta, G.T., Petersen, H.C.,
    Delefosse, M.: Bias-corrected Pearson estimating functions for Taylor’s
    power law applied to benthic macrofauna data. Stat. Probab. Lett. 81,
    749–758 (2011)

Smyth G.K. and Jørgensen B. 2002. Fitting Tweedie's compound Poisson model to
    insurance claims data: dispersion modelling. ASTIN Bulletin 32: 143–157
"""
import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln


def _theta(mu, p):
    return np.where(p == 1, np.log(mu), mu ** (1 - p) / (1 - p))


def _alpha(p):
    return (2 - p) / (1 - p)


def _logWj(y, j, p, phi):
    alpha = _alpha(p)
    logz = (-alpha * np.log(y) + alpha * np.log(p - 1) - (1 - alpha) *
            np.log(phi) - np.log(2 - p))
    return (j * logz - gammaln(1 + j) - gammaln(-alpha * j))


def kappa(mu, p):
    return mu ** (2 - p) / (2 - p)


@np.vectorize
def _sumw(y, j_l, j_u, logWmax, p, phi):
    j = np.arange(j_l, j_u + 1)
    sumw = np.sum(np.exp(_logWj(y, j, p, phi) - logWmax))
    return sumw


def logW(y, p, phi):
    alpha = _alpha(p)
    jmax = y ** (2 - p) / ((2 - p) * phi)
    logWmax = np.array((1 - alpha) * jmax)
    tol = logWmax - 37  # Machine accuracy for 64 bit.
    j = np.ceil(jmax)
    while (_logWj(y, np.ceil(j), p, phi) > tol).any():
        j = np.where(_logWj(y, j, p, phi) > tol, j + 1, j)
    j_u = j
    j = np.floor(jmax)
    j = np.where(j > 1, j, 1)
    while (_logWj(y, j, p, phi) > tol).any() and (j > 1).any():
        j = np.where(_logWj(y, j, p, phi) > tol, j - 1, 1)
    j_l = j
    sumw = _sumw(y, j_l, j_u, logWmax, p, phi)
    return logWmax + np.log(sumw)


def density_at_zero(y, mu, p, phi):
    return np.exp(-(mu ** (2 - p)) / (phi * (2 - p)))


def density_otherwise(y, mu, p, phi):
    theta = _theta(mu, p)
    logd = logW(y, p, phi) - np.log(y) + (1 / phi * (y * theta - kappa(mu, p)))
    return np.exp(logd)


def series_density(y, mu, p, phi):
    density = _lazywhere(np.array(y) > 0,
                         (y, mu, p, phi),
                         f=density_otherwise,
                         f2=density_at_zero)
    return density


if __name__ == '__main__':
    from scipy import stats
    n = stats.poisson.rvs(.1, size=10000000)
    y = stats.gamma.rvs(.1, scale=30000, size=10000000)
    y = n * y
    mu = stats.gamma.rvs(10, scale=30, size=10000000)
    import time
    t = time.time()
    out = series_density(y=y, mu=mu, p=1.5, phi=20)
    print('That took {} seconds'.format(time.time() - t))
