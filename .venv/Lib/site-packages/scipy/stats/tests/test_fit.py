import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution

from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit


# this is not a proper statistical test for convergence, but only
# verifies that the estimate and true values don't differ by too much

fit_sizes = [1000, 5000, 10000]  # sample sizes to try

thresh_percent = 0.25  # percent of true parameters for fail cut-off
thresh_min = 0.75  # minimum difference estimate - true to fail test

mle_failing_fits = [
        'gausshyper',
        'genexpon',
        'gengamma',
        'kappa4',
        'ksone',
        'kstwo',
        'ncf',
        'ncx2',
        'truncexpon',
        'tukeylambda',
        'vonmises',
        'levy_stable',
        'trapezoid',
        'truncweibull_min',
        'studentized_range',
]

# The MLE fit method of these distributions doesn't perform well when all
# parameters are fit, so test them with the location fixed at 0.
mle_use_floc0 = [
    'burr',
    'chi',
    'chi2',
    'mielke',
    'pearson3',
    'genhalflogistic',
    'rdist',
    'pareto',
    'powerlaw',  # distfn.nnlf(est2, rvs) > distfn.nnlf(est1, rvs) otherwise
    'powerlognorm',
    'wrapcauchy',
    'rel_breitwigner',
]

mm_failing_fits = ['alpha', 'betaprime', 'burr', 'burr12', 'cauchy', 'chi',
                   'chi2', 'crystalball', 'dgamma', 'dweibull', 'f',
                   'fatiguelife', 'fisk', 'foldcauchy', 'genextreme',
                   'gengamma', 'genhyperbolic', 'gennorm', 'genpareto',
                   'halfcauchy', 'invgamma', 'invweibull', 'johnsonsu',
                   'kappa3', 'ksone', 'kstwo', 'levy', 'levy_l',
                   'levy_stable', 'loglaplace', 'lomax', 'mielke', 'nakagami',
                   'ncf', 'nct', 'ncx2', 'pareto', 'powerlognorm', 'powernorm',
                   'rel_breitwigner', 'skewcauchy', 't', 'trapezoid', 'triang',
                   'truncpareto', 'truncweibull_min', 'tukeylambda',
                   'studentized_range']

# not sure if these fail, but they caused my patience to fail
mm_slow_fits = ['argus', 'exponpow', 'exponweib', 'gausshyper', 'genexpon',
                'genhalflogistic', 'halfgennorm', 'gompertz', 'johnsonsb',
                'kappa4', 'kstwobign', 'recipinvgauss',
                'truncexpon', 'vonmises', 'vonmises_line']

failing_fits = {"MM": mm_failing_fits + mm_slow_fits, "MLE": mle_failing_fits}
fail_interval_censored = {"truncpareto"}

# Don't run the fit test on these:
skip_fit = [
    'erlang',  # Subclass of gamma, generates a warning.
    'genhyperbolic',  # too slow
]


def cases_test_cont_fit():
    # this tests the closeness of the estimated parameters to the true
    # parameters with fit method of continuous distributions
    # Note: is slow, some distributions don't converge with sample
    # size <= 10000
    for distname, arg in distcont:
        if distname not in skip_fit:
            yield distname, arg


@pytest.mark.slow
@pytest.mark.parametrize('distname,arg', cases_test_cont_fit())
@pytest.mark.parametrize('method', ["MLE", "MM"])
def test_cont_fit(distname, arg, method):
    if distname in failing_fits[method]:
        # Skip failing fits unless overridden
        try:
            xfail = not int(os.environ['SCIPY_XFAIL'])
        except Exception:
            xfail = True
        if xfail:
            msg = "Fitting %s doesn't work reliably yet" % distname
            msg += (" [Set environment variable SCIPY_XFAIL=1 to run this"
                    " test nevertheless.]")
            pytest.xfail(msg)

    distfn = getattr(stats, distname)

    truearg = np.hstack([arg, [0.0, 1.0]])
    diffthreshold = np.max(np.vstack([truearg*thresh_percent,
                                      np.full(distfn.numargs+2, thresh_min)]),
                           0)

    for fit_size in fit_sizes:
        # Note that if a fit succeeds, the other fit_sizes are skipped
        np.random.seed(1234)

        with np.errstate(all='ignore'):
            rvs = distfn.rvs(size=fit_size, *arg)
            if method == 'MLE' and distfn.name in mle_use_floc0:
                kwds = {'floc': 0}
            else:
                kwds = {}
            # start with default values
            est = distfn.fit(rvs, method=method, **kwds)
            if method == 'MLE':
                # Trivial test of the use of CensoredData.  The fit() method
                # will check that data contains no actual censored data, and
                # do a regular uncensored fit.
                data1 = stats.CensoredData(rvs)
                est1 = distfn.fit(data1, **kwds)
                msg = ('Different results fitting uncensored data wrapped as'
                       f' CensoredData: {distfn.name}: est={est} est1={est1}')
                assert_allclose(est1, est, rtol=1e-10, err_msg=msg)
            if method == 'MLE' and distname not in fail_interval_censored:
                # Convert the first `nic` values in rvs to interval-censored
                # values. The interval is small, so est2 should be close to
                # est.
                nic = 15
                interval = np.column_stack((rvs, rvs))
                interval[:nic, 0] *= 0.99
                interval[:nic, 1] *= 1.01
                interval.sort(axis=1)
                data2 = stats.CensoredData(interval=interval)
                est2 = distfn.fit(data2, **kwds)
                msg = ('Different results fitting interval-censored'
                       f' data: {distfn.name}: est={est} est2={est2}')
                assert_allclose(est2, est, rtol=0.05, err_msg=msg)

        diff = est - truearg

        # threshold for location
        diffthreshold[-2] = np.max([np.abs(rvs.mean())*thresh_percent,
                                    thresh_min])

        if np.any(np.isnan(est)):
            raise AssertionError('nan returned in fit')
        else:
            if np.all(np.abs(diff) <= diffthreshold):
                break
    else:
        txt = 'parameter: %s\n' % str(truearg)
        txt += 'estimated: %s\n' % str(est)
        txt += 'diff     : %s\n' % str(diff)
        raise AssertionError('fit not very good in %s\n' % distfn.name + txt)


def _check_loc_scale_mle_fit(name, data, desired, atol=None):
    d = getattr(stats, name)
    actual = d.fit(data)[-2:]
    assert_allclose(actual, desired, atol=atol,
                    err_msg='poor mle fit of (loc, scale) in %s' % name)


def test_non_default_loc_scale_mle_fit():
    data = np.array([1.01, 1.78, 1.78, 1.78, 1.88, 1.88, 1.88, 2.00])
    _check_loc_scale_mle_fit('uniform', data, [1.01, 0.99], 1e-3)
    _check_loc_scale_mle_fit('expon', data, [1.01, 0.73875], 1e-3)


def test_expon_fit():
    """gh-6167"""
    data = [0, 0, 0, 0, 2, 2, 2, 2]
    phat = stats.expon.fit(data, floc=0)
    assert_allclose(phat, [0, 1.0], atol=1e-3)


def test_fit_error():
    data = np.concatenate([np.zeros(29), np.ones(21)])
    message = "Optimization converged to parameters that are..."
    with pytest.raises(FitError, match=message), \
            pytest.warns(RuntimeWarning):
        stats.beta.fit(data)


@pytest.mark.parametrize("dist, params",
                         [(stats.norm, (0.5, 2.5)),  # type: ignore[attr-defined] # noqa
                          (stats.binom, (10, 0.3, 2))])  # type: ignore[attr-defined] # noqa
def test_nnlf_and_related_methods(dist, params):
    rng = np.random.default_rng(983459824)

    if hasattr(dist, 'pdf'):
        logpxf = dist.logpdf
    else:
        logpxf = dist.logpmf

    x = dist.rvs(*params, size=100, random_state=rng)
    ref = -logpxf(x, *params).sum()
    res1 = dist.nnlf(params, x)
    res2 = dist._penalized_nnlf(params, x)
    assert_allclose(res1, ref)
    assert_allclose(res2, ref)


def cases_test_fit_mle():
    # These fail default test or hang
    skip_basic_fit = {'argus', 'foldnorm', 'truncpareto', 'truncweibull_min',
                      'ksone', 'levy_stable', 'studentized_range', 'kstwo'}

    # Please keep this list in alphabetical order...
    slow_basic_fit = {'alpha', 'arcsine',
                      'betaprime', 'binom', 'bradford', 'burr12',
                      'chi', 'crystalball', 'dweibull', 'exponpow',
                      'f', 'fatiguelife', 'fisk', 'foldcauchy',
                      'genexpon', 'genextreme', 'gennorm', 'genpareto',
                      'gompertz', 'halfgennorm', 'invgauss', 'invweibull',
                      'johnsonsb', 'johnsonsu', 'kappa3', 'kstwobign',
                      'loglaplace', 'lognorm', 'lomax', 'mielke',
                      'nakagami', 'nbinom', 'norminvgauss',
                      'pareto', 'pearson3', 'powerlaw', 'powernorm',
                      'randint', 'rdist', 'recipinvgauss', 'rice',
                      't', 'uniform', 'weibull_max', 'wrapcauchy'}

    # Please keep this list in alphabetical order...
    xslow_basic_fit = {'beta', 'betabinom', 'burr', 'exponweib',
                       'gausshyper', 'gengamma', 'genhalflogistic',
                       'genhyperbolic', 'geninvgauss',
                       'hypergeom', 'kappa4', 'loguniform',
                       'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius',
                       'nct', 'ncx2', 'nhypergeom',
                       'powerlognorm', 'reciprocal', 'rel_breitwigner',
                       'skellam', 'trapezoid', 'triang', 'truncnorm',
                       'tukeylambda', 'zipfian'}

    for dist in dict(distdiscrete + distcont):
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = "tested separately"
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        elif dist in slow_basic_fit:
            reason = "too slow (>= 0.25s)"
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        elif dist in xslow_basic_fit:
            reason = "too slow (>= 1.0s)"
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        else:
            yield dist


def cases_test_fit_mse():
    # the first four are so slow that I'm not sure whether they would pass
    skip_basic_fit = {'levy_stable', 'studentized_range', 'ksone', 'skewnorm',
                      'norminvgauss',  # super slow (~1 hr) but passes
                      'kstwo',  # very slow (~25 min) but passes
                      'geninvgauss',  # quite slow (~4 minutes) but passes
                      'gausshyper', 'genhyperbolic',  # integration warnings
                      'argus',  # close, but doesn't meet tolerance
                      'vonmises'}  # can have negative CDF; doesn't play nice

    # Please keep this list in alphabetical order...
    slow_basic_fit = {'alpha', 'anglit', 'arcsine', 'betabinom', 'bradford',
                      'chi', 'chi2', 'crystalball', 'dgamma', 'dweibull',
                      'erlang', 'exponnorm', 'exponpow', 'exponweib',
                      'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                      'gamma', 'genexpon', 'genextreme', 'genhalflogistic',
                      'genlogistic', 'genpareto', 'gompertz',
                      'hypergeom', 'invweibull', 'johnsonsb', 'johnsonsu',
                      'kappa3', 'kstwobign',
                      'laplace_asymmetric', 'loggamma', 'loglaplace',
                      'lognorm', 'lomax',
                      'maxwell', 'mielke', 'nakagami', 'nhypergeom',
                      'pareto', 'powernorm', 'randint', 'recipinvgauss',
                      'semicircular',
                      't', 'triang', 'truncexpon', 'truncpareto',
                      'truncweibull_min',
                      'uniform', 'vonmises_line',
                      'wald', 'weibull_max', 'weibull_min', 'wrapcauchy'}

    # Please keep this list in alphabetical order...
    xslow_basic_fit = {'beta', 'betaprime', 'burr', 'burr12',
                       'f', 'gengamma', 'gennorm',
                       'halfgennorm', 'invgamma', 'invgauss',
                       'kappa4', 'loguniform',
                       'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius',
                       'nct', 'ncx2',
                       'pearson3', 'powerlaw', 'powerlognorm',
                       'rdist', 'reciprocal', 'rel_breitwigner', 'rice',
                       'trapezoid', 'truncnorm', 'tukeylambda',
                       'zipfian'}

    warns_basic_fit = {'skellam'}  # can remove mark after gh-14901 is resolved

    for dist in dict(distdiscrete + distcont):
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = "Fails. Oh well."
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        elif dist in slow_basic_fit:
            reason = "too slow (>= 0.25s)"
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        elif dist in xslow_basic_fit:
            reason = "too slow (>= 1.0s)"
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        elif dist in warns_basic_fit:
            mark = pytest.mark.filterwarnings('ignore::RuntimeWarning')
            yield pytest.param(dist, marks=mark)
        else:
            yield dist


def cases_test_fitstart():
    for distname, shapes in dict(distcont).items():
        if (not isinstance(distname, str) or
                distname in {'studentized_range', 'recipinvgauss'}):  # slow
            continue
        yield distname, shapes


@pytest.mark.parametrize('distname, shapes', cases_test_fitstart())
def test_fitstart(distname, shapes):
    dist = getattr(stats, distname)
    rng = np.random.default_rng(216342614)
    data = rng.random(10)

    with np.errstate(invalid='ignore', divide='ignore'):  # irrelevant to test
        guess = dist._fitstart(data)

    assert dist._argcheck(*guess[:-2])


def assert_nlff_less_or_close(dist, data, params1, params0, rtol=1e-7, atol=0,
                              nlff_name='nnlf'):
    nlff = getattr(dist, nlff_name)
    nlff1 = nlff(params1, data)
    nlff0 = nlff(params0, data)
    if not (nlff1 < nlff0):
        np.testing.assert_allclose(nlff1, nlff0, rtol=rtol, atol=atol)


class TestFit:
    dist = stats.binom  # type: ignore[attr-defined]
    seed = 654634816187
    rng = np.random.default_rng(seed)
    data = stats.binom.rvs(5, 0.5, size=100, random_state=rng)  # type: ignore[attr-defined] # noqa
    shape_bounds_a = [(1, 10), (0, 1)]
    shape_bounds_d = {'n': (1, 10), 'p': (0, 1)}
    atol = 5e-2
    rtol = 1e-2
    tols = {'atol': atol, 'rtol': rtol}

    def opt(self, *args, **kwds):
        return differential_evolution(*args, seed=0, **kwds)

    def test_dist_iv(self):
        message = "`dist` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            stats.fit(10, self.data, self.shape_bounds_a)

    def test_data_iv(self):
        message = "`data` must be exactly one-dimensional."
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [[1, 2, 3]], self.shape_bounds_a)

        message = "All elements of `data` must be finite numbers."
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [1, 2, 3, np.nan], self.shape_bounds_a)
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [1, 2, 3, np.inf], self.shape_bounds_a)
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, ['1', '2', '3'], self.shape_bounds_a)

    def test_bounds_iv(self):
        message = "Bounds provided for the following unrecognized..."
        shape_bounds = {'n': (1, 10), 'p': (0, 1), '1': (0, 10)}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "Each element of a `bounds` sequence must be a tuple..."
        shape_bounds = [(1, 10, 3), (0, 1)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "Each element of `bounds` must be a tuple specifying..."
        shape_bounds = [(1, 10, 3), (0, 1, 0.5)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        shape_bounds = [1, 0]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "A `bounds` sequence must contain at least 2 elements..."
        shape_bounds = [(1, 10)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "A `bounds` sequence may not contain more than 3 elements..."
        bounds = [(1, 10), (1, 10), (1, 10), (1, 10)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, bounds)

        message = "There are no values for `p` on the interval..."
        shape_bounds = {'n': (1, 10), 'p': (1, 0)}
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "There are no values for `n` on the interval..."
        shape_bounds = [(10, 1), (0, 1)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "There are no integer values for `n` on the interval..."
        shape_bounds = [(1.4, 1.6), (0, 1)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        message = "The intersection of user-provided bounds for `n`"
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data)
        shape_bounds = [(-np.inf, np.inf), (0, 1)]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

    def test_guess_iv(self):
        message = "Guesses provided for the following unrecognized..."
        guess = {'n': 1, 'p': 0.5, '1': 255}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "Each element of `guess` must be a scalar..."
        guess = {'n': 1, 'p': 'hi'}
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
        guess = [1, 'f']
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
        guess = [[1, 2]]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "A `guess` sequence must contain at least 2..."
        guess = [1]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "A `guess` sequence may not contain more than 3..."
        guess = [1, 2, 3, 4]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "Guess for parameter `n` rounded..."
        guess = {'n': 4.5, 'p': -0.5}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "Guess for parameter `loc` rounded..."
        guess = [5, 0.5, 0.5]
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "Guess for parameter `p` clipped..."
        guess = {'n': 5, 'p': -0.5}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        message = "Guess for parameter `loc` clipped..."
        guess = [5, 0.5, 1]
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

    def basic_fit_test(self, dist_name, method):

        N = 5000
        dist_data = dict(distcont + distdiscrete)
        rng = np.random.default_rng(self.seed)
        dist = getattr(stats, dist_name)
        shapes = np.array(dist_data[dist_name])
        bounds = np.empty((len(shapes) + 2, 2), dtype=np.float64)
        bounds[:-2, 0] = shapes/10.**np.sign(shapes)
        bounds[:-2, 1] = shapes*10.**np.sign(shapes)
        bounds[-2] = (0, 10)
        bounds[-1] = (1e-16, 10)
        loc = rng.uniform(*bounds[-2])
        scale = rng.uniform(*bounds[-1])
        ref = list(dist_data[dist_name]) + [loc, scale]

        if getattr(dist, 'pmf', False):
            ref = ref[:-1]
            ref[-1] = np.floor(loc)
            data = dist.rvs(*ref, size=N, random_state=rng)
            bounds = bounds[:-1]
        if getattr(dist, 'pdf', False):
            data = dist.rvs(*ref, size=N, random_state=rng)

        with npt.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "overflow encountered")
            res = stats.fit(dist, data, bounds, method=method,
                            optimizer=self.opt)

        nlff_names = {'mle': 'nnlf', 'mse': '_penalized_nlpsf'}
        nlff_name = nlff_names[method]
        assert_nlff_less_or_close(dist, data, res.params, ref, **self.tols,
                                  nlff_name=nlff_name)

    @pytest.mark.parametrize("dist_name", cases_test_fit_mle())
    def test_basic_fit_mle(self, dist_name):
        self.basic_fit_test(dist_name, "mle")

    @pytest.mark.parametrize("dist_name", cases_test_fit_mse())
    def test_basic_fit_mse(self, dist_name):
        self.basic_fit_test(dist_name, "mse")

    def test_argus(self):
        # Can't guarantee that all distributions will fit all data with
        # arbitrary bounds. This distribution just happens to fail above.
        # Try something slightly different.
        N = 1000
        rng = np.random.default_rng(self.seed)
        dist = stats.argus
        shapes = (1., 2., 3.)
        data = dist.rvs(*shapes, size=N, random_state=rng)
        shape_bounds = {'chi': (0.1, 10), 'loc': (0.1, 10), 'scale': (0.1, 10)}
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_foldnorm(self):
        # Can't guarantee that all distributions will fit all data with
        # arbitrary bounds. This distribution just happens to fail above.
        # Try something slightly different.
        N = 1000
        rng = np.random.default_rng(self.seed)
        dist = stats.foldnorm
        shapes = (1.952125337355587, 2., 3.)
        data = dist.rvs(*shapes, size=N, random_state=rng)
        shape_bounds = {'c': (0.1, 10), 'loc': (0.1, 10), 'scale': (0.1, 10)}
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_truncpareto(self):
        # Can't guarantee that all distributions will fit all data with
        # arbitrary bounds. This distribution just happens to fail above.
        # Try something slightly different.
        N = 1000
        rng = np.random.default_rng(self.seed)
        dist = stats.truncpareto
        shapes = (1.8, 5.3, 2.3, 4.1)
        data = dist.rvs(*shapes, size=N, random_state=rng)
        shape_bounds = [(0.1, 10)]*4
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_truncweibull_min(self):
        # Can't guarantee that all distributions will fit all data with
        # arbitrary bounds. This distribution just happens to fail above.
        # Try something slightly different.
        N = 1000
        rng = np.random.default_rng(self.seed)
        dist = stats.truncweibull_min
        shapes = (2.5, 0.25, 1.75, 2., 3.)
        data = dist.rvs(*shapes, size=N, random_state=rng)
        shape_bounds = [(0.1, 10)]*5
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_missing_shape_bounds(self):
        # some distributions have a small domain w.r.t. a parameter, e.g.
        # $p \in [0, 1]$ for binomial distribution
        # User does not need to provide these because the intersection of the
        # user's bounds (none) and the distribution's domain is finite
        N = 1000
        rng = np.random.default_rng(self.seed)

        dist = stats.binom
        n, p, loc = 10, 0.65, 0
        data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
        shape_bounds = {'n': np.array([0, 20])}  # check arrays are OK, too
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
        assert_allclose(res.params, (n, p, loc), **self.tols)

        dist = stats.bernoulli
        p, loc = 0.314159, 0
        data = dist.rvs(p, loc=loc, size=N, random_state=rng)
        res = stats.fit(dist, data, optimizer=self.opt)
        assert_allclose(res.params, (p, loc), **self.tols)

    def test_fit_only_loc_scale(self):
        # fit only loc
        N = 5000
        rng = np.random.default_rng(self.seed)

        dist = stats.norm
        loc, scale = 1.5, 1
        data = dist.rvs(loc=loc, size=N, random_state=rng)
        loc_bounds = (0, 5)
        bounds = {'loc': loc_bounds}
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        assert_allclose(res.params, (loc, scale), **self.tols)

        # fit only scale
        loc, scale = 0, 2.5
        data = dist.rvs(scale=scale, size=N, random_state=rng)
        scale_bounds = (0, 5)
        bounds = {'scale': scale_bounds}
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        assert_allclose(res.params, (loc, scale), **self.tols)

        # fit only loc and scale
        dist = stats.norm
        loc, scale = 1.5, 2.5
        data = dist.rvs(loc=loc, scale=scale, size=N, random_state=rng)
        bounds = {'loc': loc_bounds, 'scale': scale_bounds}
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        assert_allclose(res.params, (loc, scale), **self.tols)

    def test_everything_fixed(self):
        N = 5000
        rng = np.random.default_rng(self.seed)

        dist = stats.norm
        loc, scale = 1.5, 2.5
        data = dist.rvs(loc=loc, scale=scale, size=N, random_state=rng)

        # loc, scale fixed to 0, 1 by default
        res = stats.fit(dist, data)
        assert_allclose(res.params, (0, 1), **self.tols)

        # loc, scale explicitly fixed
        bounds = {'loc': (loc, loc), 'scale': (scale, scale)}
        res = stats.fit(dist, data, bounds)
        assert_allclose(res.params, (loc, scale), **self.tols)

        # `n` gets fixed during polishing
        dist = stats.binom
        n, p, loc = 10, 0.65, 0
        data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
        shape_bounds = {'n': (0, 20), 'p': (0.65, 0.65)}
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
        assert_allclose(res.params, (n, p, loc), **self.tols)

    def test_failure(self):
        N = 5000
        rng = np.random.default_rng(self.seed)

        dist = stats.nbinom
        shapes = (5, 0.5)
        data = dist.rvs(*shapes, size=N, random_state=rng)

        assert data.min() == 0
        # With lower bounds on location at 0.5, likelihood is zero
        bounds = [(0, 30), (0, 1), (0.5, 10)]
        res = stats.fit(dist, data, bounds)
        message = "Optimization converged to parameter values that are"
        assert res.message.startswith(message)
        assert res.success is False

    @pytest.mark.xslow
    def test_guess(self):
        # Test that guess helps DE find the desired solution
        N = 2000
        rng = np.random.default_rng(self.seed)
        dist = stats.nhypergeom
        params = (20, 7, 12, 0)
        bounds = [(2, 200), (0.7, 70), (1.2, 120), (0, 10)]

        data = dist.rvs(*params, size=N, random_state=rng)

        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        assert not np.allclose(res.params, params, **self.tols)

        res = stats.fit(dist, data, bounds, guess=params, optimizer=self.opt)
        assert_allclose(res.params, params, **self.tols)

    def test_mse_accuracy_1(self):
        # Test maximum spacing estimation against example from Wikipedia
        # https://en.wikipedia.org/wiki/Maximum_spacing_estimation#Examples
        data = [2, 4]
        dist = stats.expon
        bounds = {'loc': (0, 0), 'scale': (1e-8, 10)}
        res_mle = stats.fit(dist, data, bounds=bounds, method='mle')
        assert_allclose(res_mle.params.scale, 3, atol=1e-3)
        res_mse = stats.fit(dist, data, bounds=bounds, method='mse')
        assert_allclose(res_mse.params.scale, 3.915, atol=1e-3)

    def test_mse_accuracy_2(self):
        # Test maximum spacing estimation against example from Wikipedia
        # https://en.wikipedia.org/wiki/Maximum_spacing_estimation#Examples
        rng = np.random.default_rng(9843212616816518964)

        dist = stats.uniform
        n = 10
        data = dist(3, 6).rvs(size=n, random_state=rng)
        bounds = {'loc': (0, 10), 'scale': (1e-8, 10)}
        res = stats.fit(dist, data, bounds=bounds, method='mse')
        # (loc=3.608118420015416, scale=5.509323262055043)

        x = np.sort(data)
        a = (n*x[0] - x[-1])/(n - 1)
        b = (n*x[-1] - x[0])/(n - 1)
        ref = a, b-a  # (3.6081133632151503, 5.509328130317254)
        assert_allclose(res.params, ref, rtol=1e-4)


# Data from Matlab: https://www.mathworks.com/help/stats/lillietest.html
examgrades = [65, 61, 81, 88, 69, 89, 55, 84, 86, 84, 71, 81, 84, 81, 78, 67,
              96, 66, 73, 75, 59, 71, 69, 63, 79, 76, 63, 85, 87, 88, 80, 71,
              65, 84, 71, 75, 81, 79, 64, 65, 84, 77, 70, 75, 84, 75, 73, 92,
              90, 79, 80, 71, 73, 71, 58, 79, 73, 64, 77, 82, 81, 59, 54, 82,
              57, 79, 79, 73, 74, 82, 63, 64, 73, 69, 87, 68, 81, 73, 83, 73,
              80, 73, 73, 71, 66, 78, 64, 74, 68, 67, 75, 75, 80, 85, 74, 76,
              80, 77, 93, 70, 86, 80, 81, 83, 68, 60, 85, 64, 74, 82, 81, 77,
              66, 85, 75, 81, 69, 60, 83, 72]


class TestGoodnessOfFit:

    def test_gof_iv(self):
        dist = stats.norm
        x = [1, 2, 3]

        message = r"`dist` must be a \(non-frozen\) instance of..."
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(stats.norm(), x)

        message = "`data` must be a one-dimensional array of numbers."
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, [[1, 2, 3]])

        message = "`statistic` must be one of..."
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, statistic='mm')

        message = "`n_mc_samples` must be an integer."
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(dist, x, n_mc_samples=1000.5)

        message = "'herring' cannot be used to seed a"
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, random_state='herring')

    def test_against_ks(self):
        rng = np.random.default_rng(8517426291317196949)
        x = examgrades
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ks', random_state=rng)
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        assert_allclose(res.statistic, ref.statistic)  # ~0.0848
        assert_allclose(res.pvalue, ref.pvalue, atol=5e-3)  # ~0.335

    def test_against_lilliefors(self):
        rng = np.random.default_rng(2291803665717442724)
        x = examgrades
        res = goodness_of_fit(stats.norm, x, statistic='ks', random_state=rng)
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        assert_allclose(res.statistic, ref.statistic)  # ~0.0848
        assert_allclose(res.pvalue, 0.0348, atol=5e-3)

    def test_against_cvm(self):
        rng = np.random.default_rng(8674330857509546614)
        x = examgrades
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='cvm', random_state=rng)
        ref = stats.cramervonmises(x, stats.norm(**known_params).cdf)
        assert_allclose(res.statistic, ref.statistic)  # ~0.090
        assert_allclose(res.pvalue, ref.pvalue, atol=5e-3)  # ~0.636

    def test_against_anderson_case_0(self):
        # "Case 0" is where loc and scale are known [1]
        rng = np.random.default_rng(7384539336846690410)
        x = np.arange(1, 101)
        # loc that produced critical value of statistic found w/ root_scalar
        known_params = {'loc': 45.01575354024957, 'scale': 30}
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 2.492)  # See [1] Table 1A 1.0
        assert_allclose(res.pvalue, 0.05, atol=5e-3)

    def test_against_anderson_case_1(self):
        # "Case 1" is where scale is known and loc is fit [1]
        rng = np.random.default_rng(5040212485680146248)
        x = np.arange(1, 101)
        # scale that produced critical value of statistic found w/ root_scalar
        known_params = {'scale': 29.957112639101933}
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 0.908)  # See [1] Table 1B 1.1
        assert_allclose(res.pvalue, 0.1, atol=5e-3)

    def test_against_anderson_case_2(self):
        # "Case 2" is where loc is known and scale is fit [1]
        rng = np.random.default_rng(726693985720914083)
        x = np.arange(1, 101)
        # loc that produced critical value of statistic found w/ root_scalar
        known_params = {'loc': 44.5680212261933}
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 2.904)  # See [1] Table 1B 1.2
        assert_allclose(res.pvalue, 0.025, atol=5e-3)

    def test_against_anderson_case_3(self):
        # "Case 3" is where both loc and scale are fit [1]
        rng = np.random.default_rng(6763691329830218206)
        # c that produced critical value of statistic found w/ root_scalar
        x = stats.skewnorm.rvs(1.4477847789132101, loc=1, scale=2, size=100,
                               random_state=rng)
        res = goodness_of_fit(stats.norm, x, statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 0.559)  # See [1] Table 1B 1.2
        assert_allclose(res.pvalue, 0.15, atol=5e-3)

    @pytest.mark.slow
    def test_against_anderson_gumbel_r(self):
        rng = np.random.default_rng(7302761058217743)
        # c that produced critical value of statistic found w/ root_scalar
        x = stats.genextreme(0.051896837188595134, loc=0.5,
                             scale=1.5).rvs(size=1000, random_state=rng)
        res = goodness_of_fit(stats.gumbel_r, x, statistic='ad',
                              random_state=rng)
        ref = stats.anderson(x, dist='gumbel_r')
        assert_allclose(res.statistic, ref.critical_values[0])
        assert_allclose(res.pvalue, ref.significance_level[0]/100, atol=5e-3)

    def test_against_filliben_norm(self):
        # Test against `stats.fit` ref. [7] Section 8 "Example"
        rng = np.random.default_rng(8024266430745011915)
        y = [6, 1, -4, 8, -2, 5, 0]
        known_params = {'loc': 0, 'scale': 1}
        res = stats.goodness_of_fit(stats.norm, y, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        # Slight discrepancy presumably due to roundoff in Filliben's
        # calculation. Using exact order statistic medians instead of
        # Filliben's approximation doesn't account for it.
        assert_allclose(res.statistic, 0.98538, atol=1e-4)
        assert 0.75 < res.pvalue < 0.9

        # Using R's ppcc library:
        # library(ppcc)
        # options(digits=16)
        # x < - c(6, 1, -4, 8, -2, 5, 0)
        # set.seed(100)
        # ppccTest(x, "qnorm", ppos="Filliben")
        # Discrepancy with
        assert_allclose(res.statistic, 0.98540957187084, rtol=2e-5)
        assert_allclose(res.pvalue, 0.8875, rtol=2e-3)

    def test_filliben_property(self):
        # Filliben's statistic should be independent of data location and scale
        rng = np.random.default_rng(8535677809395478813)
        x = rng.normal(loc=10, scale=0.5, size=100)
        res = stats.goodness_of_fit(stats.norm, x,
                                    statistic="filliben", random_state=rng)
        known_params = {'loc': 0, 'scale': 1}
        ref = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        assert_allclose(res.statistic, ref.statistic, rtol=1e-15)

    @pytest.mark.parametrize('case', [(25, [.928, .937, .950, .958, .966]),
                                      (50, [.959, .965, .972, .977, .981]),
                                      (95, [.977, .979, .983, .986, .989])])
    def test_against_filliben_norm_table(self, case):
        # Test against `stats.fit` ref. [7] Table 1
        rng = np.random.default_rng(504569995557928957)
        n, ref = case
        x = rng.random(n)
        known_params = {'loc': 0, 'scale': 1}
        res = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        percentiles = np.array([0.005, 0.01, 0.025, 0.05, 0.1])
        res = stats.scoreatpercentile(res.null_distribution, percentiles*100)
        assert_allclose(res, ref, atol=2e-3)

    @pytest.mark.slow
    @pytest.mark.parametrize('case', [(5, 0.95772790260469, 0.4755),
                                      (6, 0.95398832257958, 0.3848),
                                      (7, 0.9432692889277, 0.2328)])
    def test_against_ppcc(self, case):
        # Test against R ppcc, e.g.
        # library(ppcc)
        # options(digits=16)
        # x < - c(0.52325412, 1.06907699, -0.36084066, 0.15305959, 0.99093194)
        # set.seed(100)
        # ppccTest(x, "qrayleigh", ppos="Filliben")
        n, ref_statistic, ref_pvalue = case
        rng = np.random.default_rng(7777775561439803116)
        x = rng.normal(size=n)
        res = stats.goodness_of_fit(stats.rayleigh, x, statistic="filliben",
                                    random_state=rng)
        assert_allclose(res.statistic, ref_statistic, rtol=1e-4)
        assert_allclose(res.pvalue, ref_pvalue, atol=1.5e-2)

    def test_params_effects(self):
        # Ensure that `guessed_params`, `fit_params`, and `known_params` have
        # the intended effects.
        rng = np.random.default_rng(9121950977643805391)
        x = stats.skewnorm.rvs(-5.044559778383153, loc=1, scale=2, size=50,
                               random_state=rng)

        # Show that `guessed_params` don't fit to the guess,
        # but `fit_params` and `known_params` respect the provided fit
        guessed_params = {'c': 13.4}
        fit_params = {'scale': 13.73}
        known_params = {'loc': -13.85}
        rng = np.random.default_rng(9121950977643805391)
        res1 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)
        assert not np.allclose(res1.fit_result.params.c, 13.4)
        assert_equal(res1.fit_result.params.scale, 13.73)
        assert_equal(res1.fit_result.params.loc, -13.85)

        # Show that changing the guess changes the parameter that gets fit,
        # and it changes the null distribution
        guessed_params = {'c': 2}
        rng = np.random.default_rng(9121950977643805391)
        res2 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)
        assert not np.allclose(res2.fit_result.params.c,
                               res1.fit_result.params.c, rtol=1e-8)
        assert not np.allclose(res2.null_distribution,
                               res1.null_distribution, rtol=1e-8)
        assert_equal(res2.fit_result.params.scale, 13.73)
        assert_equal(res2.fit_result.params.loc, -13.85)

        # If we set all parameters as fit_params and known_params,
        # they're all fixed to those values, but the null distribution
        # varies.
        fit_params = {'c': 13.4, 'scale': 13.73}
        rng = np.random.default_rng(9121950977643805391)
        res3 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)
        assert_equal(res3.fit_result.params.c, 13.4)
        assert_equal(res3.fit_result.params.scale, 13.73)
        assert_equal(res3.fit_result.params.loc, -13.85)
        assert not np.allclose(res3.null_distribution, res1.null_distribution)


class TestFitResult:
    def test_plot_iv(self):
        rng = np.random.default_rng(1769658657308472721)
        data = stats.norm.rvs(0, 1, size=100, random_state=rng)

        def optimizer(*args, **kwargs):
            return differential_evolution(*args, **kwargs, seed=rng)

        bounds = [(0, 30), (0, 1)]
        res = stats.fit(stats.norm, data, bounds, optimizer=optimizer)
        try:
            import matplotlib  # noqa
            message = r"`plot_type` must be one of \{'..."
            with pytest.raises(ValueError, match=message):
                res.plot(plot_type='llama')
        except (ModuleNotFoundError, ImportError):
            message = r"matplotlib must be installed to use method `plot`."
            with pytest.raises(ModuleNotFoundError, match=message):
                res.plot(plot_type='llama')
