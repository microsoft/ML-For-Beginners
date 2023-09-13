import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools

from scipy import stats
from .common_tests import (check_normalization, check_moment,
                           check_mean_expect,
                           check_var_expect, check_skew_expect,
                           check_kurt_expect, check_entropy,
                           check_private_entropy, check_entropy_vect_scale,
                           check_edge_support, check_named_args,
                           check_random_state_property,
                           check_meth_dtype, check_ppf_dtype,
                           check_cmplx_deriv,
                           check_pickling, check_rvs_broadcast,
                           check_freezing, check_munp_expect,)
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen

"""
Test all continuous distributions.

Parameters were chosen for those distributions that pass the
Kolmogorov-Smirnov test.  This provides safe parameters for each
distributions so that we can perform further testing of class methods.

These tests currently check only/mostly for serious errors and exceptions,
not for numerically exact results.
"""

# Note that you need to add new distributions you want tested
# to _distr_params

DECIMAL = 5  # specify the precision of the tests  # increased from 0 to 5
_IS_32BIT = (sys.maxsize < 2**32)

# For skipping test_cont_basic
distslow = ['recipinvgauss', 'vonmises', 'kappa4', 'vonmises_line',
            'gausshyper', 'norminvgauss', 'geninvgauss', 'genhyperbolic',
            'truncnorm', 'truncweibull_min']

# distxslow are sorted by speed (very slow to slow)
distxslow = ['studentized_range', 'kstwo', 'ksone', 'wrapcauchy', 'genexpon']

# For skipping test_moments, which is already marked slow
distxslow_test_moments = ['studentized_range', 'vonmises', 'vonmises_line',
                          'ksone', 'kstwo', 'recipinvgauss', 'genexpon']

# skip check_fit_args (test is slow)
skip_fit_test_mle = ['exponpow', 'exponweib', 'gausshyper', 'genexpon',
                     'halfgennorm', 'gompertz', 'johnsonsb', 'johnsonsu',
                     'kappa4', 'ksone', 'kstwo', 'kstwobign', 'mielke', 'ncf',
                     'nct', 'powerlognorm', 'powernorm', 'recipinvgauss',
                     'trapezoid', 'vonmises', 'vonmises_line', 'levy_stable',
                     'rv_histogram_instance', 'studentized_range']

# these were really slow in `test_fit`.py.
# note that this list is used to skip both fit_test and fit_fix tests
slow_fit_test_mm = ['argus', 'exponpow', 'exponweib', 'gausshyper', 'genexpon',
                    'genhalflogistic', 'halfgennorm', 'gompertz', 'johnsonsb',
                    'kappa4', 'kstwobign', 'recipinvgauss',
                    'trapezoid', 'truncexpon', 'vonmises', 'vonmises_line',
                    'studentized_range']
# pearson3 fails due to something weird
# the first list fails due to non-finite distribution moments encountered
# most of the rest fail due to integration warnings
# pearson3 is overriden as not implemented due to gh-11746
fail_fit_test_mm = (['alpha', 'betaprime', 'bradford', 'burr', 'burr12',
                     'cauchy', 'crystalball', 'f', 'fisk', 'foldcauchy',
                     'genextreme', 'genpareto', 'halfcauchy', 'invgamma',
                     'kappa3', 'levy', 'levy_l', 'loglaplace', 'lomax',
                     'mielke', 'nakagami', 'ncf', 'skewcauchy', 't',
                     'tukeylambda', 'invweibull', 'rel_breitwigner']
                     + ['genhyperbolic', 'johnsonsu', 'ksone', 'kstwo',
                        'nct', 'pareto', 'powernorm', 'powerlognorm']
                     + ['pearson3'])

skip_fit_test = {"MLE": skip_fit_test_mle,
                 "MM": slow_fit_test_mm + fail_fit_test_mm}

# skip check_fit_args_fix (test is slow)
skip_fit_fix_test_mle = ['burr', 'exponpow', 'exponweib', 'gausshyper',
                         'genexpon', 'halfgennorm', 'gompertz', 'johnsonsb',
                         'johnsonsu', 'kappa4', 'ksone', 'kstwo', 'kstwobign',
                         'levy_stable', 'mielke', 'ncf', 'ncx2',
                         'powerlognorm', 'powernorm', 'rdist', 'recipinvgauss',
                         'trapezoid', 'truncpareto', 'vonmises', 'vonmises_line',
                         'studentized_range']
# the first list fails due to non-finite distribution moments encountered
# most of the rest fail due to integration warnings
# pearson3 is overriden as not implemented due to gh-11746
fail_fit_fix_test_mm = (['alpha', 'betaprime', 'burr', 'burr12', 'cauchy',
                         'crystalball', 'f', 'fisk', 'foldcauchy',
                         'genextreme', 'genpareto', 'halfcauchy', 'invgamma',
                         'kappa3', 'levy', 'levy_l', 'loglaplace', 'lomax',
                         'mielke', 'nakagami', 'ncf', 'nct', 'skewcauchy', 't',
                         'truncpareto', 'invweibull']
                        + ['genhyperbolic', 'johnsonsu', 'ksone', 'kstwo',
                           'pareto', 'powernorm', 'powerlognorm']
                        + ['pearson3'])
skip_fit_fix_test = {"MLE": skip_fit_fix_test_mle,
                     "MM": slow_fit_test_mm + fail_fit_fix_test_mm}

# These distributions fail the complex derivative test below.
# Here 'fail' mean produce wrong results and/or raise exceptions, depending
# on the implementation details of corresponding special functions.
# cf https://github.com/scipy/scipy/pull/4979 for a discussion.
fails_cmplx = {'argus', 'beta', 'betaprime', 'chi', 'chi2', 'cosine',
               'dgamma', 'dweibull', 'erlang', 'f', 'foldcauchy', 'gamma',
               'gausshyper', 'gengamma', 'genhyperbolic',
               'geninvgauss', 'gennorm', 'genpareto',
               'halfcauchy', 'halfgennorm', 'invgamma',
               'ksone', 'kstwo', 'kstwobign', 'levy_l', 'loggamma',
               'logistic', 'loguniform', 'maxwell', 'nakagami',
               'ncf', 'nct', 'ncx2', 'norminvgauss', 'pearson3',
               'powerlaw', 'rdist', 'reciprocal', 'rice',
               'skewnorm', 't', 'truncweibull_min',
               'tukeylambda', 'vonmises', 'vonmises_line',
               'rv_histogram_instance', 'truncnorm', 'studentized_range',
               'johnsonsb', 'halflogistic', 'rel_breitwigner'}


# rv_histogram instances, with uniform and non-uniform bins;
# stored as (dist, arg) tuples for cases_test_cont_basic
# and cases_test_moments.
histogram_test_instances = []
case1 = {'a': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
               6, 6, 6, 7, 7, 7, 8, 8, 9], 'bins': 8}  # equal width bins
case2 = {'a': [1, 1], 'bins': [0, 1, 10]}  # unequal width bins
for case, density in itertools.product([case1, case2], [True, False]):
    _hist = np.histogram(**case, density=density)
    _rv_hist = stats.rv_histogram(_hist, density=density)
    histogram_test_instances.append((_rv_hist, tuple()))


def cases_test_cont_basic():
    for distname, arg in distcont[:] + histogram_test_instances:
        if distname == 'levy_stable':
            continue
        elif distname in distslow:
            yield pytest.param(distname, arg, marks=pytest.mark.slow)
        elif distname in distxslow:
            yield pytest.param(distname, arg, marks=pytest.mark.xslow)
        else:
            yield distname, arg


@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
@pytest.mark.parametrize('sn, n_fit_samples', [(500, 200)])
def test_cont_basic(distname, arg, sn, n_fit_samples):
    # this test skips slow distributions

    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'

    rng = np.random.RandomState(765456)
    rvs = distfn.rvs(size=sn, *arg, random_state=rng)
    m, v = distfn.stats(*arg)

    if distname not in {'laplace_asymmetric'}:
        check_sample_meanvar_(m, v, rvs)
    check_cdf_ppf(distfn, arg, distname)
    check_sf_isf(distfn, arg, distname)
    check_pdf(distfn, arg, distname)
    check_pdf_logpdf(distfn, arg, distname)
    check_pdf_logpdf_at_endpoints(distfn, arg, distname)
    check_cdf_logcdf(distfn, arg, distname)
    check_sf_logsf(distfn, arg, distname)
    check_ppf_broadcast(distfn, arg, distname)

    alpha = 0.01
    if distname == 'rv_histogram_instance':
        check_distribution_rvs(distfn.cdf, arg, alpha, rvs)
    elif distname != 'geninvgauss':
        # skip kstest for geninvgauss since cdf is too slow; see test for
        # rv generation in TestGenInvGauss in test_distributions.py
        check_distribution_rvs(distname, arg, alpha, rvs)

    locscale_defaults = (0, 1)
    meths = [distfn.pdf, distfn.logpdf, distfn.cdf, distfn.logcdf,
             distfn.logsf]
    # make sure arguments are within support
    spec_x = {'weibull_max': -0.5, 'levy_l': -0.5,
              'pareto': 1.5, 'truncpareto': 3.2, 'tukeylambda': 0.3,
              'rv_histogram_instance': 5.0}
    x = spec_x.get(distname, 0.5)
    if distname == 'invweibull':
        arg = (1,)
    elif distname == 'ksone':
        arg = (3,)

    check_named_args(distfn, x, arg, locscale_defaults, meths)
    check_random_state_property(distfn, arg)

    if distname in ['rel_breitwigner'] and _IS_32BIT:
        # gh18414
        pytest.skip("fails on Linux 32-bit")
    else:
        check_pickling(distfn, arg)
    check_freezing(distfn, arg)

    # Entropy
    if distname not in ['kstwobign', 'kstwo', 'ncf']:
        check_entropy(distfn, arg, distname)

    if distfn.numargs == 0:
        check_vecentropy(distfn, arg)

    if (distfn.__class__._entropy != stats.rv_continuous._entropy
            and distname != 'vonmises'):
        check_private_entropy(distfn, arg, stats.rv_continuous)

    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        sup.filter(IntegrationWarning, "Extremely bad integrand")
        sup.filter(RuntimeWarning, "invalid value")
        check_entropy_vect_scale(distfn, arg)

    check_retrieving_support(distfn, arg)
    check_edge_support(distfn, arg)

    check_meth_dtype(distfn, arg, meths)
    check_ppf_dtype(distfn, arg)

    if distname not in fails_cmplx:
        check_cmplx_deriv(distfn, arg)

    if distname != 'truncnorm':
        check_ppf_private(distfn, arg, distname)

    for method in ["MLE", "MM"]:
        if distname not in skip_fit_test[method]:
            check_fit_args(distfn, arg, rvs[:n_fit_samples], method)

        if distname not in skip_fit_fix_test[method]:
            check_fit_args_fix(distfn, arg, rvs[:n_fit_samples], method)


@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
def test_rvs_scalar(distname, arg):
    # rvs should return a scalar when given scalar arguments (gh-12428)
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'

    assert np.isscalar(distfn.rvs(*arg))
    assert np.isscalar(distfn.rvs(*arg, size=()))
    assert np.isscalar(distfn.rvs(*arg, size=None))


def test_levy_stable_random_state_property():
    # levy_stable only implements rvs(), so it is skipped in the
    # main loop in test_cont_basic(). Here we apply just the test
    # check_random_state_property to levy_stable.
    check_random_state_property(stats.levy_stable, (0.5, 0.1))


def cases_test_moments():
    fail_normalization = set()
    fail_higher = {'ncf'}
    fail_moment = {'johnsonsu'}  # generic `munp` is inaccurate for johnsonsu

    for distname, arg in distcont[:] + histogram_test_instances:
        if distname == 'levy_stable':
            continue

        if distname in distxslow_test_moments:
            yield pytest.param(distname, arg, True, True, True, True,
                               marks=pytest.mark.xslow(reason="too slow"))
            continue

        cond1 = distname not in fail_normalization
        cond2 = distname not in fail_higher
        cond3 = distname not in fail_moment

        marks = list()
        # Currently unused, `marks` can be used to add a timeout to a test of
        # a specific distribution.  For example, this shows how a timeout could
        # be added for the 'skewnorm' distribution:
        #
        #     marks = list()
        #     if distname == 'skewnorm':
        #         marks.append(pytest.mark.timeout(300))

        yield pytest.param(distname, arg, cond1, cond2, cond3,
                           False, marks=marks)

        if not cond1 or not cond2 or not cond3:
            # Run the distributions that have issues twice, once skipping the
            # not_ok parts, once with the not_ok parts but marked as knownfail
            yield pytest.param(distname, arg, True, True, True, True,
                               marks=[pytest.mark.xfail] + marks)


@pytest.mark.slow
@pytest.mark.parametrize('distname,arg,normalization_ok,higher_ok,moment_ok,'
                         'is_xfailing',
                         cases_test_moments())
def test_moments(distname, arg, normalization_ok, higher_ok, moment_ok,
                 is_xfailing):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'

    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning,
                   "The integral is probably divergent, or slowly convergent.")
        sup.filter(IntegrationWarning,
                   "The maximum number of subdivisions.")
        sup.filter(IntegrationWarning,
                   "The algorithm does not converge.")

        if is_xfailing:
            sup.filter(IntegrationWarning)

        m, v, s, k = distfn.stats(*arg, moments='mvsk')

        with np.errstate(all="ignore"):
            if normalization_ok:
                check_normalization(distfn, arg, distname)

            if higher_ok:
                check_mean_expect(distfn, arg, m, distname)
                check_skew_expect(distfn, arg, m, v, s, distname)
                check_var_expect(distfn, arg, m, v, distname)
                check_kurt_expect(distfn, arg, m, v, k, distname)
                check_munp_expect(distfn, arg, distname)

        check_loc_scale(distfn, arg, m, v, distname)

        if moment_ok:
            check_moment(distfn, arg, m, v, distname)


@pytest.mark.parametrize('dist,shape_args', distcont)
def test_rvs_broadcast(dist, shape_args):
    if dist in ['gausshyper', 'studentized_range']:
        pytest.skip("too slow")

    if dist in ['rel_breitwigner'] and _IS_32BIT:
        # gh18414
        pytest.skip("fails on Linux 32-bit")

    # If shape_only is True, it means the _rvs method of the
    # distribution uses more than one random number to generate a random
    # variate.  That means the result of using rvs with broadcasting or
    # with a nontrivial size will not necessarily be the same as using the
    # numpy.vectorize'd version of rvs(), so we can only compare the shapes
    # of the results, not the values.
    # Whether or not a distribution is in the following list is an
    # implementation detail of the distribution, not a requirement.  If
    # the implementation the rvs() method of a distribution changes, this
    # test might also have to be changed.
    shape_only = dist in ['argus', 'betaprime', 'dgamma', 'dweibull',
                          'exponnorm', 'genhyperbolic', 'geninvgauss',
                          'levy_stable', 'nct', 'norminvgauss', 'rice',
                          'skewnorm', 'semicircular', 'gennorm', 'loggamma']

    distfunc = getattr(stats, dist)
    loc = np.zeros(2)
    scale = np.ones((3, 1))
    nargs = distfunc.numargs
    allargs = []
    bshape = [3, 2]
    # Generate shape parameter arguments...
    for k in range(nargs):
        shp = (k + 4,) + (1,)*(k + 2)
        allargs.append(shape_args[k]*np.ones(shp))
        bshape.insert(0, k + 4)
    allargs.extend([loc, scale])
    # bshape holds the expected shape when loc, scale, and the shape
    # parameters are all broadcast together.

    check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, 'd')


# Expected values of the SF, CDF, PDF were computed using
# mpmath with mpmath.mp.dps = 50 and output at 20:
#
# def ks(x, n):
#     x = mpmath.mpf(x)
#     logp = -mpmath.power(6.0*n*x+1.0, 2)/18.0/n
#     sf, cdf = mpmath.exp(logp), -mpmath.expm1(logp)
#     pdf = (6.0*n*x+1.0) * 2 * sf/3
#     print(mpmath.nstr(sf, 20), mpmath.nstr(cdf, 20), mpmath.nstr(pdf, 20))
#
# Tests use 1/n < x < 1-1/n and n > 1e6 to use the asymptotic computation.
# Larger x has a smaller sf.
@pytest.mark.parametrize('x,n,sf,cdf,pdf,rtol',
                         [(2.0e-5, 1000000000,
                           0.44932297307934442379, 0.55067702692065557621,
                           35946.137394996276407, 5e-15),
                          (2.0e-9, 1000000000,
                           0.99999999061111115519, 9.3888888448132728224e-9,
                           8.6666665852962971765, 5e-14),
                          (5.0e-4, 1000000000,
                           7.1222019433090374624e-218, 1.0,
                           1.4244408634752704094e-211, 5e-14)])
def test_gh17775_regression(x, n, sf, cdf, pdf, rtol):
    # Regression test for gh-17775. In scipy 1.9.3 and earlier,
    # these test would fail.
    #
    # KS one asymptotic sf ~ e^(-(6nx+1)^2 / 18n)
    # Given a large 32-bit integer n, 6n will overflow in the c implementation.
    # Example of broken behaviour:
    # ksone.sf(2.0e-5, 1000000000) == 0.9374359693473666
    ks = stats.ksone
    vals = np.array([ks.sf(x, n), ks.cdf(x, n), ks.pdf(x, n)])
    expected = np.array([sf, cdf, pdf])
    npt.assert_allclose(vals, expected, rtol=rtol)
    # The sf+cdf must sum to 1.0.
    npt.assert_equal(vals[0] + vals[1], 1.0)
    # Check inverting the (potentially very small) sf (uses a lower tolerance)
    npt.assert_allclose([ks.isf(sf, n)], [x], rtol=1e-8)


def test_rvs_gh2069_regression():
    # Regression tests for gh-2069.  In scipy 0.17 and earlier,
    # these tests would fail.
    #
    # A typical example of the broken behavior:
    # >>> norm.rvs(loc=np.zeros(5), scale=np.ones(5))
    # array([-2.49613705, -2.49613705, -2.49613705, -2.49613705, -2.49613705])
    rng = np.random.RandomState(123)
    vals = stats.norm.rvs(loc=np.zeros(5), scale=1, random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=0, scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=np.zeros(5), scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=np.array([[0], [0]]), scale=np.ones(5),
                          random_state=rng)
    d = np.diff(vals.ravel())
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")

    assert_raises(ValueError, stats.norm.rvs, [[0, 0], [0, 0]],
                  [[1, 1], [1, 1]], 1)
    assert_raises(ValueError, stats.gamma.rvs, [2, 3, 4, 5], 0, 1, (2, 2))
    assert_raises(ValueError, stats.gamma.rvs, [1, 1, 1, 1], [0, 0, 0, 0],
                  [[1], [2]], (4,))


def test_nomodify_gh9900_regression():
    # Regression test for gh-9990
    # Prior to gh-9990, calls to stats.truncnorm._cdf() use what ever was
    # set inside the stats.truncnorm instance during stats.truncnorm.cdf().
    # This could cause issues wth multi-threaded code.
    # Since then, the calls to cdf() are not permitted to modify the global
    # stats.truncnorm instance.
    tn = stats.truncnorm
    # Use the right-half truncated normal
    # Check that the cdf and _cdf return the same result.
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf), 0.6826894921370859)
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]), 0.6826894921370859)

    # Now use the left-half truncated normal
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0), 0.31731050786291415)
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]), 0.31731050786291415)

    # Check that the right-half truncated normal _cdf hasn't changed
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]), 0.6826894921370859)  # noqa, NOT 1.6826894921370859
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf), 0.6826894921370859)

    # Check that the left-half truncated normal _cdf hasn't changed
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]), 0.31731050786291415)  # noqa, Not -0.6826894921370859
    npt.assert_almost_equal(tn.cdf(1, -np.inf, 0), 1)                     # Not 1.6826894921370859
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0), 0.31731050786291415)  # Not -0.6826894921370859


def test_broadcast_gh9990_regression():
    # Regression test for gh-9990
    # The x-value 7 only lies within the support of 4 of the supplied
    # distributions.  Prior to 9990, one array passed to
    # stats.reciprocal._cdf would have 4 elements, but an array
    # previously stored by stats.reciprocal_argcheck() would have 6, leading
    # to a broadcast error.
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([8, 16, 1, 32, 1, 48])
    ans = [stats.reciprocal.cdf(7, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(7, a, b), ans)

    ans = [stats.reciprocal.cdf(1, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(1, a, b), ans)

    ans = [stats.reciprocal.cdf(_a, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(a, a, b), ans)

    ans = [stats.reciprocal.cdf(_b, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(b, a, b), ans)


def test_broadcast_gh7933_regression():
    # Check broadcast works
    stats.truncnorm.logpdf(
        np.array([3.0, 2.0, 1.0]),
        a=(1.5 - np.array([6.0, 5.0, 4.0])) / 3.0,
        b=np.inf,
        loc=np.array([6.0, 5.0, 4.0]),
        scale=3.0
    )


def test_gh2002_regression():
    # Add a check that broadcast works in situations where only some
    # x-values are compatible with some of the shape arguments.
    x = np.r_[-2:2:101j]
    a = np.r_[-np.ones(50), np.ones(51)]
    expected = [stats.truncnorm.pdf(_x, _a, np.inf) for _x, _a in zip(x, a)]
    ans = stats.truncnorm.pdf(x, a, np.inf)
    npt.assert_array_almost_equal(ans, expected)


def test_gh1320_regression():
    # Check that the first example from gh-1320 now works.
    c = 2.62
    stats.genextreme.ppf(0.5, np.array([[c], [c + 0.5]]))
    # The other examples in gh-1320 appear to have stopped working
    # some time ago.
    # ans = stats.genextreme.moment(2, np.array([c, c + 0.5]))
    # expected = np.array([25.50105963, 115.11191437])
    # stats.genextreme.moment(5, np.array([[c], [c + 0.5]]))
    # stats.genextreme.moment(5, np.array([c, c + 0.5]))


def test_method_of_moments():
    # example from https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
    np.random.seed(1234)
    x = [0, 0, 0, 0, 1]
    a = 1/5 - 2*np.sqrt(3)/5
    b = 1/5 + 2*np.sqrt(3)/5
    # force use of method of moments (uniform.fit is overriden)
    loc, scale = super(type(stats.uniform), stats.uniform).fit(x, method="MM")
    npt.assert_almost_equal(loc, a, decimal=4)
    npt.assert_almost_equal(loc+scale, b, decimal=4)


def check_sample_meanvar_(popmean, popvar, sample):
    if np.isfinite(popmean):
        check_sample_mean(sample, popmean)
    if np.isfinite(popvar):
        check_sample_var(sample, popvar)


def check_sample_mean(sample, popmean):
    # Checks for unlikely difference between sample mean and population mean
    prob = stats.ttest_1samp(sample, popmean).pvalue
    assert prob > 0.01


def check_sample_var(sample, popvar):
    # check that population mean lies within the CI bootstrapped from the
    # sample. This used to be a chi-squared test for variance, but there were
    # too many false positives
    res = stats.bootstrap(
        (sample,),
        lambda x, axis: x.var(ddof=1, axis=axis),
        confidence_level=0.995,
    )
    conf = res.confidence_interval
    low, high = conf.low, conf.high
    assert low <= popvar <= high


def check_cdf_ppf(distfn, arg, msg):
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg),
                            values, decimal=DECIMAL, err_msg=msg +
                            ' - cdf-ppf roundtrip')


def check_sf_isf(distfn, arg, msg):
    npt.assert_almost_equal(distfn.sf(distfn.isf([0.1, 0.5, 0.9], *arg), *arg),
                            [0.1, 0.5, 0.9], decimal=DECIMAL, err_msg=msg +
                            ' - sf-isf roundtrip')
    npt.assert_almost_equal(distfn.cdf([0.1, 0.9], *arg),
                            1.0 - distfn.sf([0.1, 0.9], *arg),
                            decimal=DECIMAL, err_msg=msg +
                            ' - cdf-sf relationship')


def check_pdf(distfn, arg, msg):
    # compares pdf at median with numerical derivative of cdf
    median = distfn.ppf(0.5, *arg)
    eps = 1e-6
    pdfv = distfn.pdf(median, *arg)
    if (pdfv < 1e-4) or (pdfv > 1e4):
        # avoid checking a case where pdf is close to zero or
        # huge (singularity)
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) -
               distfn.cdf(median - eps, *arg))/eps/2.0
    # replace with better diff and better test (more points),
    # actually, this works pretty well
    msg += ' - cdf-pdf relationship'
    npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg)


def check_pdf_logpdf(distfn, args, msg):
    # compares pdf at several points with the log of the pdf
    points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    logpdf = distfn.logpdf(vals, *args)
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    logpdf = logpdf[np.isfinite(logpdf)]
    msg += " - logpdf-log(pdf) relationship"
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)


def check_pdf_logpdf_at_endpoints(distfn, args, msg):
    # compares pdf with the log of the pdf at the (finite) end points
    points = np.array([0, 1])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    logpdf = distfn.logpdf(vals, *args)
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    logpdf = logpdf[np.isfinite(logpdf)]
    msg += " - logpdf-log(pdf) relationship"
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)


def check_sf_logsf(distfn, args, msg):
    # compares sf at several points with the log of the sf
    points = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    sf = distfn.sf(vals, *args)
    logsf = distfn.logsf(vals, *args)
    sf = sf[sf != 0]
    logsf = logsf[np.isfinite(logsf)]
    msg += " - logsf-log(sf) relationship"
    npt.assert_almost_equal(np.log(sf), logsf, decimal=7, err_msg=msg)


def check_cdf_logcdf(distfn, args, msg):
    # compares cdf at several points with the log of the cdf
    points = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    cdf = distfn.cdf(vals, *args)
    logcdf = distfn.logcdf(vals, *args)
    cdf = cdf[cdf != 0]
    logcdf = logcdf[np.isfinite(logcdf)]
    msg += " - logcdf-log(cdf) relationship"
    npt.assert_almost_equal(np.log(cdf), logcdf, decimal=7, err_msg=msg)


def check_ppf_broadcast(distfn, arg, msg):
    # compares ppf for multiple argsets.
    num_repeats = 5
    args = [] * num_repeats
    if arg:
        args = [np.array([_] * num_repeats) for _ in arg]

    median = distfn.ppf(0.5, *arg)
    medians = distfn.ppf(0.5, *args)
    msg += " - ppf multiple"
    npt.assert_almost_equal(medians, [median] * num_repeats, decimal=7, err_msg=msg)


def check_distribution_rvs(dist, args, alpha, rvs):
    # dist is either a cdf function or name of a distribution in scipy.stats.
    # args are the args for scipy.stats.dist(*args)
    # alpha is a significance level, ~0.01
    # rvs is array_like of random variables
    # test from scipy.stats.tests
    # this version reuses existing random variables
    D, pval = stats.kstest(rvs, dist, args=args, N=1000)
    if (pval < alpha):
        # The rvs passed in failed the K-S test, which _could_ happen
        # but is unlikely if alpha is small enough.
        # Repeat the test with a new sample of rvs.
        # Generate 1000 rvs, perform a K-S test that the new sample of rvs
        # are distributed according to the distribution.
        D, pval = stats.kstest(dist, dist, args=args, N=1000)
        npt.assert_(pval > alpha, "D = " + str(D) + "; pval = " + str(pval) +
                    "; alpha = " + str(alpha) + "\nargs = " + str(args))


def check_vecentropy(distfn, args):
    npt.assert_equal(distfn.vecentropy(*args), distfn._entropy(*args))


def check_loc_scale(distfn, arg, m, v, msg):
    # Make `loc` and `scale` arrays to catch bugs like gh-13580 where
    # `loc` and `scale` arrays improperly broadcast with shapes.
    loc, scale = np.array([10.0, 20.0]), np.array([10.0, 20.0])
    mt, vt = distfn.stats(*arg, loc=loc, scale=scale)
    npt.assert_allclose(m*scale + loc, mt)
    npt.assert_allclose(v*scale*scale, vt)


def check_ppf_private(distfn, arg, msg):
    # fails by design for truncnorm self.nb not defined
    ppfs = distfn._ppf(np.array([0.1, 0.5, 0.9]), *arg)
    npt.assert_(not np.any(np.isnan(ppfs)), msg + 'ppf private is nan')


def check_retrieving_support(distfn, args):
    loc, scale = 1, 2
    supp = distfn.support(*args)
    supp_loc_scale = distfn.support(*args, loc=loc, scale=scale)
    npt.assert_almost_equal(np.array(supp)*scale + loc,
                            np.array(supp_loc_scale))


def check_fit_args(distfn, arg, rvs, method):
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning,
                   message="The shape parameter of the erlang")
        sup.filter(category=RuntimeWarning,
                   message="floating point number truncated")
        vals = distfn.fit(rvs, method=method)
        vals2 = distfn.fit(rvs, optimizer='powell', method=method)
    # Only check the length of the return; accuracy tested in test_fit.py
    npt.assert_(len(vals) == 2+len(arg))
    npt.assert_(len(vals2) == 2+len(arg))


def check_fit_args_fix(distfn, arg, rvs, method):
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning,
                   message="The shape parameter of the erlang")

        vals = distfn.fit(rvs, floc=0, method=method)
        vals2 = distfn.fit(rvs, fscale=1, method=method)
        npt.assert_(len(vals) == 2+len(arg))
        npt.assert_(vals[-2] == 0)
        npt.assert_(vals2[-1] == 1)
        npt.assert_(len(vals2) == 2+len(arg))
        if len(arg) > 0:
            vals3 = distfn.fit(rvs, f0=arg[0], method=method)
            npt.assert_(len(vals3) == 2+len(arg))
            npt.assert_(vals3[0] == arg[0])
        if len(arg) > 1:
            vals4 = distfn.fit(rvs, f1=arg[1], method=method)
            npt.assert_(len(vals4) == 2+len(arg))
            npt.assert_(vals4[1] == arg[1])
        if len(arg) > 2:
            vals5 = distfn.fit(rvs, f2=arg[2], method=method)
            npt.assert_(len(vals5) == 2+len(arg))
            npt.assert_(vals5[2] == arg[2])


@pytest.mark.parametrize('method', ['pdf', 'logpdf', 'cdf', 'logcdf',
                                    'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distcont)
def test_methods_with_lists(method, distname, args):
    # Test that the continuous distributions can accept Python lists
    # as arguments.
    dist = getattr(stats, distname)
    f = getattr(dist, method)
    if distname == 'invweibull' and method.startswith('log'):
        x = [1.5, 2]
    else:
        x = [0.1, 0.2]

    shape2 = [[a]*2 for a in args]
    loc = [0, 0.1]
    scale = [1, 1.01]
    result = f(x, *shape2, loc=loc, scale=scale)
    npt.assert_allclose(result,
                        [f(*v) for v in zip(x, *shape2, loc, scale)],
                        rtol=1e-14, atol=5e-14)


def test_burr_fisk_moment_gh13234_regression():
    vals0 = stats.burr.moment(1, 5, 4)
    assert isinstance(vals0, float)

    vals1 = stats.fisk.moment(1, 8)
    assert isinstance(vals1, float)


def test_moments_with_array_gh12192_regression():
    # array loc and scalar scale
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=1)
    expected0 = np.array([1., 2., 3.])
    npt.assert_equal(vals0, expected0)

    # array loc and invalid scalar scale
    vals1 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=-1)
    expected1 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals1, expected1)

    # array loc and array scale with invalid entries
    vals2 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]),
                              scale=[-3, 1, 0])
    expected2 = np.array([np.nan, 2., np.nan])
    npt.assert_equal(vals2, expected2)

    # (loc == 0) & (scale < 0)
    vals3 = stats.norm.moment(order=2, loc=0, scale=-4)
    expected3 = np.nan
    npt.assert_equal(vals3, expected3)
    assert isinstance(vals3, expected3.__class__)

    # array loc with 0 entries and scale with invalid entries
    vals4 = stats.norm.moment(order=2, loc=[1, 0, 2], scale=[3, -4, -5])
    expected4 = np.array([10., np.nan, np.nan])
    npt.assert_equal(vals4, expected4)

    # all(loc == 0) & (array scale with invalid entries)
    vals5 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[5., -2, 100.])
    expected5 = np.array([25., np.nan, 10000.])
    npt.assert_equal(vals5, expected5)

    # all( (loc == 0) & (scale < 0) )
    vals6 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[-5., -2, -100.])
    expected6 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals6, expected6)

    # scalar args, loc, and scale
    vals7 = stats.chi.moment(order=2, df=1, loc=0, scale=0)
    expected7 = np.nan
    npt.assert_equal(vals7, expected7)
    assert isinstance(vals7, expected7.__class__)

    # array args, scalar loc, and scalar scale
    vals8 = stats.chi.moment(order=2, df=[1, 2, 3], loc=0, scale=0)
    expected8 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals8, expected8)

    # array args, array loc, and array scale
    vals9 = stats.chi.moment(order=2, df=[1, 2, 3], loc=[1., 0., 2.],
                             scale=[1., -3., 0.])
    expected9 = np.array([3.59576912, np.nan, np.nan])
    npt.assert_allclose(vals9, expected9, rtol=1e-8)

    # (n > 4), all(loc != 0), and all(scale != 0)
    vals10 = stats.norm.moment(5, [1., 2.], [1., 2.])
    expected10 = np.array([26., 832.])
    npt.assert_allclose(vals10, expected10, rtol=1e-13)

    # test broadcasting and more
    a = [-1.1, 0, 1, 2.2, np.pi]
    b = [-1.1, 0, 1, 2.2, np.pi]
    loc = [-1.1, 0, np.sqrt(2)]
    scale = [-2.1, 0, 1, 2.2, np.pi]

    a = np.array(a).reshape((-1, 1, 1, 1))
    b = np.array(b).reshape((-1, 1, 1))
    loc = np.array(loc).reshape((-1, 1))
    scale = np.array(scale)

    vals11 = stats.beta.moment(order=2, a=a, b=b, loc=loc, scale=scale)

    a, b, loc, scale = np.broadcast_arrays(a, b, loc, scale)

    for i in np.ndenumerate(a):
        with np.errstate(invalid='ignore', divide='ignore'):
            i = i[0]  # just get the index
            # check against same function with scalar input
            expected = stats.beta.moment(order=2, a=a[i], b=b[i],
                                         loc=loc[i], scale=scale[i])
            np.testing.assert_equal(vals11[i], expected)


def test_broadcasting_in_moments_gh12192_regression():
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[[1]])
    expected0 = np.array([[1., 2., 3.]])
    npt.assert_equal(vals0, expected0)
    assert vals0.shape == expected0.shape

    vals1 = stats.norm.moment(order=1, loc=np.array([[1], [2], [3]]),
                              scale=[1, 2, 3])
    expected1 = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    npt.assert_equal(vals1, expected1)
    assert vals1.shape == expected1.shape

    vals2 = stats.chi.moment(order=1, df=[1., 2., 3.], loc=0., scale=1.)
    expected2 = np.array([0.79788456, 1.25331414, 1.59576912])
    npt.assert_allclose(vals2, expected2, rtol=1e-8)
    assert vals2.shape == expected2.shape

    vals3 = stats.chi.moment(order=1, df=[[1.], [2.], [3.]], loc=[0., 1., 2.],
                             scale=[-1., 0., 3.])
    expected3 = np.array([[np.nan, np.nan, 4.39365368],
                          [np.nan, np.nan, 5.75994241],
                          [np.nan, np.nan, 6.78730736]])
    npt.assert_allclose(vals3, expected3, rtol=1e-8)
    assert vals3.shape == expected3.shape


def test_kappa3_array_gh13582():
    # https://github.com/scipy/scipy/pull/15140#issuecomment-994958241
    shapes = [0.5, 1.5, 2.5, 3.5, 4.5]
    moments = 'mvsk'
    res = np.array([[stats.kappa3.stats(shape, moments=moment)
                   for shape in shapes] for moment in moments])
    res2 = np.array(stats.kappa3.stats(shapes, moments=moments))
    npt.assert_allclose(res, res2)


@pytest.mark.xslow
def test_kappa4_array_gh13582():
    h = np.array([-0.5, 2.5, 3.5, 4.5, -3])
    k = np.array([-0.5, 1, -1.5, 0, 3.5])
    moments = 'mvsk'
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment)
                   for i in range(5)] for moment in moments])
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    npt.assert_allclose(res, res2)

    # https://github.com/scipy/scipy/pull/15250#discussion_r775112913
    h = np.array([-1, -1/4, -1/4, 1, -1, 0])
    k = np.array([1, 1, 1/2, -1/3, -1, 0])
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment)
                   for i in range(6)] for moment in moments])
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    npt.assert_allclose(res, res2)

    # https://github.com/scipy/scipy/pull/15250#discussion_r775115021
    h = np.array([-1, -0.5, 1])
    k = np.array([-1, -0.5, 0, 1])[:, None]
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    assert res2.shape == (4, 4, 3)


def test_frozen_attributes():
    # gh-14827 reported that all frozen distributions had both pmf and pdf
    # attributes; continuous should have pdf and discrete should have pmf.
    message = "'rv_continuous_frozen' object has no attribute"
    with pytest.raises(AttributeError, match=message):
        stats.norm().pmf
    with pytest.raises(AttributeError, match=message):
        stats.norm().logpmf
    stats.norm.pmf = "herring"
    frozen_norm = stats.norm()
    assert isinstance(frozen_norm, rv_continuous_frozen)
    delattr(stats.norm, 'pmf')


def test_skewnorm_pdf_gh16038():
    rng = np.random.default_rng(0)
    x, a = -np.inf, 0
    npt.assert_equal(stats.skewnorm.pdf(x, a), stats.norm.pdf(x))
    x, a = rng.random(size=(3, 3)), rng.random(size=(3, 3))
    mask = rng.random(size=(3, 3)) < 0.5
    a[mask] = 0
    x_norm = x[mask]
    res = stats.skewnorm.pdf(x, a)
    npt.assert_equal(res[mask], stats.norm.pdf(x_norm))
    npt.assert_equal(res[~mask], stats.skewnorm.pdf(x[~mask], a[~mask]))


# for scalar input, these functions should return scalar output
scalar_out = [['rvs', []], ['pdf', [0]], ['logpdf', [0]], ['cdf', [0]],
              ['logcdf', [0]], ['sf', [0]], ['logsf', [0]], ['ppf', [0]],
              ['isf', [0]], ['moment', [1]], ['entropy', []], ['expect', []],
              ['median', []], ['mean', []], ['std', []], ['var', []]]
scalars_out = [['interval', [0.95]], ['support', []], ['stats', ['mv']]]


@pytest.mark.parametrize('case', scalar_out + scalars_out)
def test_scalar_for_scalar(case):
    # Some rv_continuous functions returned 0d array instead of NumPy scalar
    # Guard against regression
    method_name, args = case
    method = getattr(stats.norm(), method_name)
    res = method(*args)
    if case in scalar_out:
        assert isinstance(res, np.number)
    else:
        assert isinstance(res[0], np.number)
        assert isinstance(res[1], np.number)


def test_scalar_for_scalar2():
    # test methods that are not attributes of frozen distributions
    res = stats.norm.fit([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.fit_loc_scale([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.nnlf((0, 1), [1, 2, 3])
    assert isinstance(res, np.number)
