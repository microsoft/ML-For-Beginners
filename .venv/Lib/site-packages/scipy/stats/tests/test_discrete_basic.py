import numpy.testing as npt
from numpy.testing import assert_allclose

import numpy as np
import pytest

from scipy import stats
from .common_tests import (check_normalization, check_moment,
                           check_mean_expect,
                           check_var_expect, check_skew_expect,
                           check_kurt_expect, check_entropy,
                           check_private_entropy, check_edge_support,
                           check_named_args, check_random_state_property,
                           check_pickling, check_rvs_broadcast,
                           check_freezing,)
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen

vals = ([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
distdiscrete += [[stats.rv_discrete(values=vals), ()]]

# For these distributions, test_discrete_basic only runs with test mode full
distslow = {'zipfian', 'nhypergeom'}


def cases_test_discrete_basic():
    seen = set()
    for distname, arg in distdiscrete:
        if distname in distslow:
            yield pytest.param(distname, arg, distname, marks=pytest.mark.slow)
        else:
            yield distname, arg, distname not in seen
        seen.add(distname)


@pytest.mark.parametrize('distname,arg,first_case', cases_test_discrete_basic())
def test_discrete_basic(distname, arg, first_case):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'sample distribution'
    np.random.seed(9765456)
    rvs = distfn.rvs(size=2000, *arg)
    supp = np.unique(rvs)
    m, v = distfn.stats(*arg)
    check_cdf_ppf(distfn, arg, supp, distname + ' cdf_ppf')

    check_pmf_cdf(distfn, arg, distname)
    check_oth(distfn, arg, supp, distname + ' oth')
    check_edge_support(distfn, arg)

    alpha = 0.01
    check_discrete_chisquare(distfn, arg, rvs, alpha,
                             distname + ' chisquare')

    if first_case:
        locscale_defaults = (0,)
        meths = [distfn.pmf, distfn.logpmf, distfn.cdf, distfn.logcdf,
                 distfn.logsf]
        # make sure arguments are within support
        # for some distributions, this needs to be overridden
        spec_k = {'randint': 11, 'hypergeom': 4, 'bernoulli': 0,
                  'nchypergeom_wallenius': 6}
        k = spec_k.get(distname, 1)
        check_named_args(distfn, k, arg, locscale_defaults, meths)
        if distname != 'sample distribution':
            check_scale_docstring(distfn)
        check_random_state_property(distfn, arg)
        check_pickling(distfn, arg)
        check_freezing(distfn, arg)

        # Entropy
        check_entropy(distfn, arg, distname)
        if distfn.__class__._entropy != stats.rv_discrete._entropy:
            check_private_entropy(distfn, arg, stats.rv_discrete)


@pytest.mark.parametrize('distname,arg', distdiscrete)
def test_moments(distname, arg):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'sample distribution'
    m, v, s, k = distfn.stats(*arg, moments='mvsk')
    check_normalization(distfn, arg, distname)

    # compare `stats` and `moment` methods
    check_moment(distfn, arg, m, v, distname)
    check_mean_expect(distfn, arg, m, distname)
    check_var_expect(distfn, arg, m, v, distname)
    check_skew_expect(distfn, arg, m, v, s, distname)
    if distname not in ['zipf', 'yulesimon', 'betanbinom']:
        check_kurt_expect(distfn, arg, m, v, k, distname)

    # frozen distr moments
    check_moment_frozen(distfn, arg, m, 1)
    check_moment_frozen(distfn, arg, v+m*m, 2)


@pytest.mark.parametrize('dist,shape_args', distdiscrete)
def test_rvs_broadcast(dist, shape_args):
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
    shape_only = dist in ['betabinom', 'betanbinom', 'skellam', 'yulesimon',
                          'dlaplace', 'nchypergeom_fisher',
                          'nchypergeom_wallenius']

    try:
        distfunc = getattr(stats, dist)
    except TypeError:
        distfunc = dist
        dist = f'rv_discrete(values=({dist.xk!r}, {dist.pk!r}))'
    loc = np.zeros(2)
    nargs = distfunc.numargs
    allargs = []
    bshape = []
    # Generate shape parameter arguments...
    for k in range(nargs):
        shp = (k + 3,) + (1,)*(k + 1)
        param_val = shape_args[k]
        allargs.append(np.full(shp, param_val))
        bshape.insert(0, shp[0])
    allargs.append(loc)
    bshape.append(loc.size)
    # bshape holds the expected shape when loc, scale, and the shape
    # parameters are all broadcast together.
    check_rvs_broadcast(
        distfunc, dist, allargs, bshape, shape_only, [np.dtype(int)]
    )


@pytest.mark.parametrize('dist,args', distdiscrete)
def test_ppf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    #check with a negative, no and positive relocation.
    np.random.seed(1942349)
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        npt.assert_array_equal(
            [_a-1+loc, _b+loc],
            [distfn.ppf(0.0, *args, loc=loc), distfn.ppf(1.0, *args, loc=loc)]
            )


@pytest.mark.parametrize('dist, args', distdiscrete)
def test_isf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    # check with a negative, no and positive relocation.
    np.random.seed(1942349)
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        expected = _b + loc, _a - 1 + loc
        res = distfn.isf(0., *args, loc=loc), distfn.isf(1., *args, loc=loc)
        npt.assert_array_equal(expected, res)
    # test broadcasting behaviour
    re_locs = [np.random.randint(-10, -1, size=(5, 3)),
               np.zeros((5, 3)),
               np.random.randint(1, 10, size=(5, 3))]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        expected = _b + loc, _a - 1 + loc
        res = distfn.isf(0., *args, loc=loc), distfn.isf(1., *args, loc=loc)
        npt.assert_array_equal(expected, res)


def check_cdf_ppf(distfn, arg, supp, msg):
    # supp is assumed to be an array of integers in the support of distfn
    # (but not necessarily all the integers in the support).
    # This test assumes that the PMF of any value in the support of the
    # distribution is greater than 1e-8.

    # cdf is a step function, and ppf(q) = min{k : cdf(k) >= q, k integer}
    cdf_supp = distfn.cdf(supp, *arg)
    # In very rare cases, the finite precision calculation of ppf(cdf(supp))
    # can produce an array in which an element is off by one.  We nudge the
    # CDF values down by 15 ULPs help to avoid this.
    cdf_supp0 = cdf_supp - 15*np.spacing(cdf_supp)
    npt.assert_array_equal(distfn.ppf(cdf_supp0, *arg),
                           supp, msg + '-roundtrip')
    # Repeat the same calculation, but with the CDF values decreased by 1e-8.
    npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg) - 1e-8, *arg),
                           supp, msg + '-roundtrip')

    if not hasattr(distfn, 'xk'):
        _a, _b = distfn.support(*arg)
        supp1 = supp[supp < _b]
        npt.assert_array_equal(distfn.ppf(distfn.cdf(supp1, *arg) + 1e-8, *arg),
                               supp1 + distfn.inc, msg + ' ppf-cdf-next')


def check_pmf_cdf(distfn, arg, distname):
    if hasattr(distfn, 'xk'):
        index = distfn.xk
    else:
        startind = int(distfn.ppf(0.01, *arg) - 1)
        index = list(range(startind, startind + 10))
    cdfs = distfn.cdf(index, *arg)
    pmfs_cum = distfn.pmf(index, *arg).cumsum()

    atol, rtol = 1e-10, 1e-10
    if distname == 'skellam':    # ncx2 accuracy
        atol, rtol = 1e-5, 1e-5
    npt.assert_allclose(cdfs - cdfs[0], pmfs_cum - pmfs_cum[0],
                        atol=atol, rtol=rtol)

    # also check that pmf at non-integral k is zero
    k = np.asarray(index)
    k_shifted = k[:-1] + np.diff(k)/2
    npt.assert_equal(distfn.pmf(k_shifted, *arg), 0)

    # better check frozen distributions, and also when loc != 0
    loc = 0.5
    dist = distfn(loc=loc, *arg)
    npt.assert_allclose(dist.pmf(k[1:] + loc), np.diff(dist.cdf(k + loc)))
    npt.assert_equal(dist.pmf(k_shifted + loc), 0)


def check_moment_frozen(distfn, arg, m, k):
    npt.assert_allclose(distfn(*arg).moment(k), m,
                        atol=1e-10, rtol=1e-10)


def check_oth(distfn, arg, supp, msg):
    # checking other methods of distfn
    npt.assert_allclose(distfn.sf(supp, *arg), 1. - distfn.cdf(supp, *arg),
                        atol=1e-10, rtol=1e-10)

    q = np.linspace(0.01, 0.99, 20)
    npt.assert_allclose(distfn.isf(q, *arg), distfn.ppf(1. - q, *arg),
                        atol=1e-10, rtol=1e-10)

    median_sf = distfn.isf(0.5, *arg)
    npt.assert_(distfn.sf(median_sf - 1, *arg) > 0.5)
    npt.assert_(distfn.cdf(median_sf + 1, *arg) > 0.5)


def check_discrete_chisquare(distfn, arg, rvs, alpha, msg):
    """Perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    """
    wsupp = 0.05

    # construct intervals with minimum mass `wsupp`.
    # intervals are left-half-open as in a cdf difference
    _a, _b = distfn.support(*arg)
    lo = int(max(_a, -1000))
    high = int(min(_b, 1000)) + 1
    distsupport = range(lo, high)
    last = 0
    distsupp = [lo]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii, *arg)
        if current - last >= wsupp - 1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > (1 - wsupp):
                break
    if distsupp[-1] < _b:
        distsupp.append(_b)
        distmass.append(1 - last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)

    # convert intervals to right-half-open as required by histogram
    histsupp = distsupp + 1e-8
    histsupp[0] = _a

    # find sample frequencies and perform chisquare test
    freq, hsupp = np.histogram(rvs, histsupp)
    chis, pval = stats.chisquare(np.array(freq), len(rvs)*distmass)

    npt.assert_(
        pval > alpha,
        f'chisquare - test for {msg} at arg = {str(arg)} with pval = {str(pval)}'
    )


def check_scale_docstring(distfn):
    if distfn.__doc__ is not None:
        # Docstrings can be stripped if interpreter is run with -OO
        npt.assert_('scale' not in distfn.__doc__)


@pytest.mark.parametrize('method', ['pmf', 'logpmf', 'cdf', 'logcdf',
                                    'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distdiscrete)
def test_methods_with_lists(method, distname, args):
    # Test that the discrete distributions can accept Python lists
    # as arguments.
    try:
        dist = getattr(stats, distname)
    except TypeError:
        return
    if method in ['ppf', 'isf']:
        z = [0.1, 0.2]
    else:
        z = [0, 1]
    p2 = [[p]*2 for p in args]
    loc = [0, 1]
    result = dist.pmf(z, *p2, loc=loc)
    npt.assert_allclose(result,
                        [dist.pmf(*v) for v in zip(z, *p2, loc)],
                        rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('distname, args', invdistdiscrete)
def test_cdf_gh13280_regression(distname, args):
    # Test for nan output when shape parameters are invalid
    dist = getattr(stats, distname)
    x = np.arange(-2, 15)
    vals = dist.cdf(x, *args)
    expected = np.nan
    npt.assert_equal(vals, expected)


def cases_test_discrete_integer_shapes():
    # distributions parameters that are only allowed to be integral when
    # fitting, but are allowed to be real as input to PDF, etc.
    integrality_exceptions = {'nbinom': {'n'}, 'betanbinom': {'n'}}

    seen = set()
    for distname, shapes in distdiscrete:
        if distname in seen:
            continue
        seen.add(distname)

        try:
            dist = getattr(stats, distname)
        except TypeError:
            continue

        shape_info = dist._shape_info()

        for i, shape in enumerate(shape_info):
            if (shape.name in integrality_exceptions.get(distname, set()) or
                    not shape.integrality):
                continue

            yield distname, shape.name, shapes


@pytest.mark.parametrize('distname, shapename, shapes',
                         cases_test_discrete_integer_shapes())
def test_integer_shapes(distname, shapename, shapes):
    dist = getattr(stats, distname)
    shape_info = dist._shape_info()
    shape_names = [shape.name for shape in shape_info]
    i = shape_names.index(shapename)  # this element of params must be integral

    shapes_copy = list(shapes)

    valid_shape = shapes[i]
    invalid_shape = valid_shape - 0.5  # arbitrary non-integral value
    new_valid_shape = valid_shape - 1
    shapes_copy[i] = [[valid_shape], [invalid_shape], [new_valid_shape]]

    a, b = dist.support(*shapes)
    x = np.round(np.linspace(a, b, 5))

    pmf = dist.pmf(x, *shapes_copy)
    assert not np.any(np.isnan(pmf[0, :]))
    assert np.all(np.isnan(pmf[1, :]))
    assert not np.any(np.isnan(pmf[2, :]))


def test_frozen_attributes():
    # gh-14827 reported that all frozen distributions had both pmf and pdf
    # attributes; continuous should have pdf and discrete should have pmf.
    message = "'rv_discrete_frozen' object has no attribute"
    with pytest.raises(AttributeError, match=message):
        stats.binom(10, 0.5).pdf
    with pytest.raises(AttributeError, match=message):
        stats.binom(10, 0.5).logpdf
    stats.binom.pdf = "herring"
    frozen_binom = stats.binom(10, 0.5)
    assert isinstance(frozen_binom, rv_discrete_frozen)
    delattr(stats.binom, 'pdf')


@pytest.mark.parametrize('distname, shapes', distdiscrete)
def test_interval(distname, shapes):
    # gh-11026 reported that `interval` returns incorrect values when
    # `confidence=1`. The values were not incorrect, but it was not intuitive
    # that the left end of the interval should extend beyond the support of the
    # distribution. Confirm that this is the behavior for all distributions.
    if isinstance(distname, str):
        dist = getattr(stats, distname)
    else:
        dist = distname
    a, b = dist.support(*shapes)
    npt.assert_equal(dist.ppf([0, 1], *shapes), (a-1, b))
    npt.assert_equal(dist.isf([1, 0], *shapes), (a-1, b))
    npt.assert_equal(dist.interval(1, *shapes), (a-1, b))


@pytest.mark.xfail_on_32bit("Sensible to machine precision")
def test_rv_sample():
    # Thoroughly test rv_sample and check that gh-3758 is resolved

    # Generate a random discrete distribution
    rng = np.random.default_rng(98430143469)
    xk = np.sort(rng.random(10) * 10)
    pk = rng.random(10)
    pk /= np.sum(pk)
    dist = stats.rv_discrete(values=(xk, pk))

    # Generate points to the left and right of xk
    xk_left = (np.array([0] + xk[:-1].tolist()) + xk)/2
    xk_right = (np.array(xk[1:].tolist() + [xk[-1]+1]) + xk)/2

    # Generate points to the left and right of cdf
    cdf2 = np.cumsum(pk)
    cdf2_left = (np.array([0] + cdf2[:-1].tolist()) + cdf2)/2
    cdf2_right = (np.array(cdf2[1:].tolist() + [1]) + cdf2)/2

    # support - leftmost and rightmost xk
    a, b = dist.support()
    assert_allclose(a, xk[0])
    assert_allclose(b, xk[-1])

    # pmf - supported only on the xk
    assert_allclose(dist.pmf(xk), pk)
    assert_allclose(dist.pmf(xk_right), 0)
    assert_allclose(dist.pmf(xk_left), 0)

    # logpmf is log of the pmf; log(0) = -np.inf
    with np.errstate(divide='ignore'):
        assert_allclose(dist.logpmf(xk), np.log(pk))
        assert_allclose(dist.logpmf(xk_right), -np.inf)
        assert_allclose(dist.logpmf(xk_left), -np.inf)

    # cdf - the cumulative sum of the pmf
    assert_allclose(dist.cdf(xk), cdf2)
    assert_allclose(dist.cdf(xk_right), cdf2)
    assert_allclose(dist.cdf(xk_left), [0]+cdf2[:-1].tolist())

    with np.errstate(divide='ignore'):
        assert_allclose(dist.logcdf(xk), np.log(dist.cdf(xk)),
                        atol=1e-15)
        assert_allclose(dist.logcdf(xk_right), np.log(dist.cdf(xk_right)),
                        atol=1e-15)
        assert_allclose(dist.logcdf(xk_left), np.log(dist.cdf(xk_left)),
                        atol=1e-15)

    # sf is 1-cdf
    assert_allclose(dist.sf(xk), 1-dist.cdf(xk))
    assert_allclose(dist.sf(xk_right), 1-dist.cdf(xk_right))
    assert_allclose(dist.sf(xk_left), 1-dist.cdf(xk_left))

    with np.errstate(divide='ignore'):
        assert_allclose(dist.logsf(xk), np.log(dist.sf(xk)),
                        atol=1e-15)
        assert_allclose(dist.logsf(xk_right), np.log(dist.sf(xk_right)),
                        atol=1e-15)
        assert_allclose(dist.logsf(xk_left), np.log(dist.sf(xk_left)),
                        atol=1e-15)

    # ppf
    assert_allclose(dist.ppf(cdf2), xk)
    assert_allclose(dist.ppf(cdf2_left), xk)
    assert_allclose(dist.ppf(cdf2_right)[:-1], xk[1:])
    assert_allclose(dist.ppf(0), a - 1)
    assert_allclose(dist.ppf(1), b)

    # isf
    sf2 = dist.sf(xk)
    assert_allclose(dist.isf(sf2), xk)
    assert_allclose(dist.isf(1-cdf2_left), dist.ppf(cdf2_left))
    assert_allclose(dist.isf(1-cdf2_right), dist.ppf(cdf2_right))
    assert_allclose(dist.isf(0), b)
    assert_allclose(dist.isf(1), a - 1)

    # interval is (ppf(alpha/2), isf(alpha/2))
    ps = np.linspace(0.01, 0.99, 10)
    int2 = dist.ppf(ps/2), dist.isf(ps/2)
    assert_allclose(dist.interval(1-ps), int2)
    assert_allclose(dist.interval(0), dist.median())
    assert_allclose(dist.interval(1), (a-1, b))

    # median is simply ppf(0.5)
    med2 = dist.ppf(0.5)
    assert_allclose(dist.median(), med2)

    # all four stats (mean, var, skew, and kurtosis) from the definitions
    mean2 = np.sum(xk*pk)
    var2 = np.sum((xk - mean2)**2 * pk)
    skew2 = np.sum((xk - mean2)**3 * pk) / var2**(3/2)
    kurt2 = np.sum((xk - mean2)**4 * pk) / var2**2 - 3
    assert_allclose(dist.mean(), mean2)
    assert_allclose(dist.std(), np.sqrt(var2))
    assert_allclose(dist.var(), var2)
    assert_allclose(dist.stats(moments='mvsk'), (mean2, var2, skew2, kurt2))

    # noncentral moment against definition
    mom3 = np.sum((xk**3) * pk)
    assert_allclose(dist.moment(3), mom3)

    # expect - check against moments
    assert_allclose(dist.expect(lambda x: 1), 1)
    assert_allclose(dist.expect(), mean2)
    assert_allclose(dist.expect(lambda x: x**3), mom3)

    # entropy is the negative of the expected value of log(p)
    with np.errstate(divide='ignore'):
        assert_allclose(-dist.expect(lambda x: dist.logpmf(x)), dist.entropy())

    # RVS is just ppf of uniform random variates
    rng = np.random.default_rng(98430143469)
    rvs = dist.rvs(size=100, random_state=rng)
    rng = np.random.default_rng(98430143469)
    rvs0 = dist.ppf(rng.random(size=100))
    assert_allclose(rvs, rvs0)
