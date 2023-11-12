"""
Test functions for genmod.families.links
"""
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
from scipy import stats
import pytest

import statsmodels.genmod.families as families
from statsmodels.tools import numdiff as nd

# Family instances
links = families.links
logit = links.Logit()
inverse_power = links.InversePower()
sqrt = links.Sqrt()
inverse_squared = links.InverseSquared()
identity = links.Identity()
log = links.Log()
logc = links.LogC()
probit = links.Probit()
cauchy = links.Cauchy()
cloglog = links.CLogLog()
loglog = links.LogLog()
negbinom = links.NegativeBinomial()

# TODO: parametrize all these tess
Links = [logit, inverse_power, sqrt, inverse_squared, identity,
         log, logc, probit, cauchy, cloglog, loglog, negbinom]

# links with defined second derivative of inverse link.
LinksISD = [inverse_power, sqrt, inverse_squared, identity,
            logc, cauchy, probit, loglog]


def get_domainvalue(link):
    """
    Get a value in the domain for a given family.
    """
    z = -np.log(np.random.uniform(0, 1))
    if isinstance(link, links.CLogLog):  # prone to overflow
        z = min(z, 3)
    elif isinstance(link, links.LogLog):
        z = max(z, -3)
    elif isinstance(link, (links.NegativeBinomial, links.LogC)):
        # domain is negative numbers
        z = -z
    return z


def test_inverse():
    # Logic check that link.inverse(link) and link(link.inverse)
    # are the identity.
    np.random.seed(3285)

    for link in Links:
        for k in range(10):
            p = np.random.uniform(0, 1)  # In domain for all families
            d = link.inverse(link(p))
            assert_allclose(d, p, atol=1e-8, err_msg=str(link))

            z = get_domainvalue(link)
            d = link(link.inverse(z))
            assert_allclose(d, z, atol=1e-8, err_msg=str(link))


def test_deriv():
    # Check link function derivatives using numeric differentiation.

    np.random.seed(24235)

    for link in Links:
        for k in range(10):
            p = np.random.uniform(0, 1)
            if isinstance(link, links.Cauchy):
                p = np.clip(p, 0.03, 0.97)
            d = link.deriv(p)
            da = nd.approx_fprime(np.r_[p], link)
            assert_allclose(d, da, rtol=1e-6, atol=1e-6,
                            err_msg=str(link))
            if not isinstance(link, (type(inverse_power),
                                     type(inverse_squared),
                                     type(logc))):
                # check monotonically increasing
                assert_array_less(-d, 0)


def test_deriv2():
    # Check link function second derivatives using numeric differentiation.

    np.random.seed(24235)

    for link in Links:
        for k in range(10):
            p = np.random.uniform(0, 1)
            p = np.clip(p, 0.01, 0.99)
            if isinstance(link, links.cauchy):
                p = np.clip(p, 0.03, 0.97)
            d = link.deriv2(p)
            da = nd.approx_fprime(np.r_[p], link.deriv)
            assert_allclose(d, da, rtol=5e-6, atol=1e-6,
                            err_msg=str(link))


def test_inverse_deriv():
    # Logic check that inverse_deriv equals 1/link.deriv(link.inverse)

    np.random.seed(24235)

    for link in Links:
        for k in range(10):
            z = get_domainvalue(link)
            d = link.inverse_deriv(z)
            f = 1 / link.deriv(link.inverse(z))
            assert_allclose(d, f, rtol=1e-8, atol=1e-10,
                            err_msg=str(link))


def test_inverse_deriv2():
    # Check second derivative of inverse link using numeric differentiation.

    np.random.seed(24235)

    for link in LinksISD:
        for k in range(10):
            z = get_domainvalue(link)
            d2 = link.inverse_deriv2(z)
            d2a = nd.approx_fprime(np.r_[z], link.inverse_deriv)
            assert_allclose(d2, d2a, rtol=5e-6, atol=1e-6,
                            err_msg=str(link))


def test_invlogit_stability():
    z = [1123.4910007309222, 1483.952316802719, 1344.86033748641,
         706.339159002542, 1167.9986375146532, 663.8345826933115,
         1496.3691686913917, 1563.0763842182257, 1587.4309332296314,
         697.1173174974248, 1333.7256198289665, 1388.7667560586933,
         819.7605431778434, 1479.9204150555015, 1078.5642245164856,
         480.10338454985896, 1112.691659145772, 534.1061908007274,
         918.2011296406588, 1280.8808515887802, 758.3890788775948,
         673.503699841035, 1556.7043357878208, 819.5269028006679,
         1262.5711060356423, 1098.7271535253608, 1482.811928490097,
         796.198809756532, 893.7946963941745, 470.3304989319786,
         1427.77079226037, 1365.2050226373822, 1492.4193201661922,
         871.9922191949931, 768.4735925445908, 732.9222777654679,
         812.2382651982667, 495.06449978924525]
    zinv = logit.inverse(z)
    assert_equal(zinv, np.ones_like(z))


class MyCLogLog(links.Link):

    def __call__(self, p):
        # p = self._clean(p)
        return np.log(-np.log(1 - p))

    def inverse(self, z):
        return 1 - np.exp(-np.exp(z))

    def deriv(self, p):
        # p = self._clean(p)
        return 1. / ((p - 1) * (np.log(1 - p)))


class CasesCDFLink():
    # just as namespace to hold cases for test_cdflink

    link_pairs = [
        (links.CDFLink(dbn=stats.gumbel_l), links.CLogLog()),
        (links.CDFLink(dbn=stats.gumbel_r), links.LogLog()),
        (links.CDFLink(dbn=stats.norm), links.Probit()),
        (links.CDFLink(dbn=stats.logistic), links.Logit()),
        (links.CDFLink(dbn=stats.t(1)), links.Cauchy()),
        # approximation of t by normal is not good enough for rtol, atol
        # (links.CDFLink(dbn=stats.t(1000000)), links.Probit()),

        (MyCLogLog(), links.CLogLog()),  # not a cdflink, but compares
        ]

    methods = ['__call__', 'deriv', 'inverse', 'inverse_deriv', 'deriv2',
               'inverse_deriv2']

    p = np.linspace(0, 1, 6)
    eps = 1e-3
    p = np.clip(p, eps, 1 - eps)


@pytest.mark.parametrize("m", CasesCDFLink.methods)
@pytest.mark.parametrize("link1, link2", CasesCDFLink.link_pairs)
def test_cdflink(m, link1, link2):
    p = CasesCDFLink.p
    res1 = getattr(link1, m)(p)
    res2 = getattr(link2, m)(p)

    assert_allclose(res1, res2, atol=1e-8, rtol=1e-8)
