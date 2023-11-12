"""
Test functions for genmod.families.family
"""
import warnings

import pytest

import numpy as np
from numpy.testing import assert_allclose

from scipy import integrate

from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
    ValueWarning,
    )
import statsmodels.genmod.families as F
from statsmodels.genmod.families.family import Tweedie
import statsmodels.genmod.families.links as L

all_links = {
    L.Logit, L.logit, L.Power, L.inverse_power, L.sqrt, L.inverse_squared,
    L.identity, L.Log, L.log, L.CDFLink, L.probit, L.cauchy, L.LogLog,
    L.loglog, L.CLogLog, L.cloglog, L.NegativeBinomial, L.nbinom
}
poisson_links = {L.Log, L.log, L.identity, L.sqrt}
gaussian_links = {L.Log, L.log, L.identity, L.inverse_power}
gamma_links = {L.Log, L.log, L.identity, L.inverse_power}
binomial_links = {
    L.Logit, L.logit, L.probit, L.cauchy, L.Log, L.log, L.CLogLog,
    L.cloglog, L.LogLog, L.loglog, L.identity
}
inverse_gaussian_links = {
    L.inverse_squared, L.inverse_power, L.identity, L.Log, L.log
}
negative_bionomial_links = {
    L.Log, L.log, L.CLogLog, L.cloglog, L.identity, L.NegativeBinomial,
    L.nbinom, L.Power
}
tweedie_links = {L.Log, L.log, L.Power}

link_cases = [
    (F.Poisson, poisson_links),
    (F.Gaussian, gaussian_links),
    (F.Gamma, gamma_links),
    (F.Binomial, binomial_links),
    (F.InverseGaussian, inverse_gaussian_links),
    (F.NegativeBinomial, negative_bionomial_links),
    (F.Tweedie, tweedie_links)
]


@pytest.mark.parametrize("family, links", link_cases)
def test_invalid_family_link(family, links):
    invalid_links = all_links - links
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            msg = ("Negative binomial dispersion parameter alpha not set. "
                   "Using default value alpha=1.0.")
            warnings.filterwarnings("ignore", message=msg,
                                    category=UserWarning)
            warnings.filterwarnings("ignore",
                                    category=FutureWarning)
            for link in invalid_links:
                family(link())


@pytest.mark.parametrize("family, links", link_cases)
def test_family_link(family, links):
    with warnings.catch_warnings():
        msg = ("Negative binomial dispersion parameter alpha not set. "
               "Using default value alpha=1.0.")
        warnings.filterwarnings("ignore", message=msg,
                                category=ValueWarning)
        warnings.filterwarnings("ignore",
                                category=FutureWarning)
        for link in links:
            assert family(link())


@pytest.mark.parametrize("family, links", link_cases)
def test_family_link_check(family, links):
    # check that we can turn of all link checks
    class Hugo():
        pass
    with warnings.catch_warnings():
        msg = ("Negative binomial dispersion parameter alpha not set. "
               "Using default value alpha=1.0.")
        warnings.filterwarnings("ignore", message=msg,
                                category=ValueWarning)
        assert family(Hugo(), check_link=False)


@pytest.mark.skipif(SP_LT_17, reason="Scipy too old, function not available")
@pytest.mark.parametrize("power", (1.1, 1.5, 1.9))
def test_tweedie_loglike_obs(power):
    """Test that Tweedie loglike is normalized to 1."""
    tweedie = Tweedie(var_power=power, eql=False)
    mu = 2.0
    scale = 2.9

    def pdf(y):
        return np.squeeze(
            np.exp(
                tweedie.loglike_obs(endog=y, mu=mu, scale=scale)
            )
        )

    assert_allclose(pdf(0) + integrate.quad(pdf, 0, 1e2)[0], 1, atol=1e-4)
