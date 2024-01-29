import warnings
import numpy as np
import pytest

from scipy.fft._fftlog import fht, ifht, fhtoffset
from scipy.special import poch

from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close


@array_api_compatible
def test_fht_agrees_with_fftlog(xp):
    # check that fht numerically agrees with the output from Fortran FFTLog,
    # the results were generated with the provided `fftlogtest` program,
    # after fixing how the k array is generated (divide range by n-1, not n)

    # test function, analytical Hankel transform is of the same form
    def f(r, mu):
        return r**(mu+1)*np.exp(-r**2/2)

    r = np.logspace(-4, 4, 16)

    dln = np.log(r[1]/r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = xp.asarray(f(r, mu))

    # test 1: compute as given
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-0.1159922613593045E-02, +0.1625822618458832E-02,
              -0.1949518286432330E-02, +0.3789220182554077E-02,
              +0.5093959119952945E-03, +0.2785387803618774E-01,
              +0.9944952700848897E-01, +0.4599202164586588E+00,
              +0.3157462160881342E+00, -0.8201236844404755E-03,
              -0.7834031308271878E-03, +0.3931444945110708E-03,
              -0.2697710625194777E-03, +0.3568398050238820E-03,
              -0.5554454827797206E-03, +0.8286331026468585E-03]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)

    # test 2: change to optimal offset
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+0.4353768523152057E-04, -0.9197045663594285E-05,
              +0.3150140927838524E-03, +0.9149121960963704E-03,
              +0.5808089753959363E-02, +0.2548065256377240E-01,
              +0.1339477692089897E+00, +0.4821530509479356E+00,
              +0.2659899781579785E+00, -0.1116475278448113E-01,
              +0.1791441617592385E-02, -0.4181810476548056E-03,
              +0.1314963536765343E-03, -0.5422057743066297E-04,
              +0.3208681804170443E-04, -0.2696849476008234E-04]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)

    # test 3: positive bias
    bias = 0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-7.3436673558316850E+00, +0.1710271207817100E+00,
              +0.1065374386206564E+00, -0.5121739602708132E-01,
              +0.2636649319269470E-01, +0.1697209218849693E-01,
              +0.1250215614723183E+00, +0.4739583261486729E+00,
              +0.2841149874912028E+00, -0.8312764741645729E-02,
              +0.1024233505508988E-02, -0.1644902767389120E-03,
              +0.3305775476926270E-04, -0.7786993194882709E-05,
              +0.1962258449520547E-05, -0.8977895734909250E-06]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)

    # test 4: negative bias
    bias = -0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+0.8985777068568745E-05, +0.4074898209936099E-04,
              +0.2123969254700955E-03, +0.1009558244834628E-02,
              +0.5131386375222176E-02, +0.2461678673516286E-01,
              +0.1235812845384476E+00, +0.4719570096404403E+00,
              +0.2893487490631317E+00, -0.1686570611318716E-01,
              +0.2231398155172505E-01, -0.1480742256379873E-01,
              +0.1692387813500801E+00, +0.3097490354365797E+00,
              +2.7593607182401860E+00, 10.5251075070045800E+00]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)


@array_api_compatible
@pytest.mark.parametrize('optimal', [True, False])
@pytest.mark.parametrize('offset', [0.0, 1.0, -1.0])
@pytest.mark.parametrize('bias', [0, 0.1, -0.1])
@pytest.mark.parametrize('n', [64, 63])
def test_fht_identity(n, bias, offset, optimal, xp):
    rng = np.random.RandomState(3491349965)

    a = xp.asarray(rng.standard_normal(n))
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)

    if optimal:
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)

    A = fht(a, dln, mu, offset=offset, bias=bias)
    a_ = ifht(A, dln, mu, offset=offset, bias=bias)

    xp_assert_close(a_, a)


@array_api_compatible
def test_fht_special_cases(xp):
    rng = np.random.RandomState(3491349965)

    a = xp.asarray(rng.standard_normal(64))
    dln = rng.uniform(-1, 1)

    # let x = (mu+1+q)/2, y = (mu+1-q)/2, M = {0, -1, -2, ...}

    # case 1: x in M, y in M => well-defined transform
    mu, bias = -4.0, 1.0
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'

    # case 2: x not in M, y in M => well-defined transform
    mu, bias = -2.5, 0.5
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'

    # case 3: x in M, y not in M => singular transform
    mu, bias = -3.5, 0.5
    with pytest.warns(Warning) as record:
        fht(a, dln, mu, bias=bias)
        assert record, 'fht did not warn about a singular transform'

    # case 4: x not in M, y in M => singular inverse transform
    mu, bias = -2.5, 0.5
    with pytest.warns(Warning) as record:
        ifht(a, dln, mu, bias=bias)
        assert record, 'ifht did not warn about a singular transform'


@array_api_compatible
@pytest.mark.parametrize('n', [64, 63])
def test_fht_exact(n, xp):
    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1-mu, 1/2)

    r = np.logspace(-2, 2, n)
    a = xp.asarray(r**gamma)

    dln = np.log(r[1]/r[0])

    offset = fhtoffset(dln, mu, initial=0.0, bias=gamma)

    A = fht(a, dln, mu, offset=offset, bias=gamma)

    k = np.exp(offset)/r[::-1]

    # analytical result
    At = xp.asarray((2/k)**gamma * poch((mu+1-gamma)/2, gamma))

    xp_assert_close(A, At)
