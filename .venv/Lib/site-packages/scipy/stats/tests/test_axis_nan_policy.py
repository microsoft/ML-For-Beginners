# Many scipy.stats functions support `axis` and `nan_policy` parameters.
# When the two are combined, it can be tricky to get all the behavior just
# right. This file contains a suite of common tests for scipy.stats functions
# that support `axis` and `nan_policy` and additional tests for some associated
# functions in stats._util.

from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays


def unpack_ttest_result(res):
    low, high = res.confidence_interval()
    return (res.statistic, res.pvalue, res.df, res._standard_error,
            res._estimate, low, high)


def _get_ttest_ci(ttest):
    # get a function that returns the CI bounds of provided `ttest`
    def ttest_ci(*args, **kwargs):
        res = ttest(*args, **kwargs)
        return res.confidence_interval()
    return ttest_ci


axis_nan_policy_cases = [
    # function, args, kwds, number of samples, number of outputs,
    # ... paired, unpacker function
    # args, kwds typically aren't needed; just showing that they work
    (stats.kruskal, tuple(), dict(), 3, 2, False, None),  # 4 samples is slow
    (stats.ranksums, ('less',), dict(), 2, 2, False, None),
    (stats.mannwhitneyu, tuple(), {'method': 'asymptotic'}, 2, 2, False, None),
    (stats.wilcoxon, ('pratt',), {'mode': 'auto'}, 2, 2, True,
     lambda res: (res.statistic, res.pvalue)),
    (stats.wilcoxon, tuple(), dict(), 1, 2, True,
     lambda res: (res.statistic, res.pvalue)),
    (stats.wilcoxon, tuple(), {'mode': 'approx'}, 1, 3, True,
     lambda res: (res.statistic, res.pvalue, res.zstatistic)),
    (stats.gmean, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.hmean, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.pmean, (1.42,), dict(), 1, 1, False, lambda x: (x,)),
    (stats.sem, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.iqr, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.kurtosis, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.skew, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.kstat, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.kstatvar, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.moment, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.moment, tuple(), dict(moment=[1, 2]), 1, 2, False, None),
    (stats.jarque_bera, tuple(), dict(), 1, 2, False, None),
    (stats.ttest_1samp, (np.array([0]),), dict(), 1, 7, False,
     unpack_ttest_result),
    (stats.ttest_rel, tuple(), dict(), 2, 7, True, unpack_ttest_result),
    (stats.ttest_ind, tuple(), dict(), 2, 7, False, unpack_ttest_result),
    (_get_ttest_ci(stats.ttest_1samp), (0,), dict(), 1, 2, False, None),
    (_get_ttest_ci(stats.ttest_rel), tuple(), dict(), 2, 2, True, None),
    (_get_ttest_ci(stats.ttest_ind), tuple(), dict(), 2, 2, False, None),
    (stats.mode, tuple(), dict(), 1, 2, True, lambda x: (x.mode, x.count))
]

# If the message is one of those expected, put nans in
# appropriate places of `statistics` and `pvalues`
too_small_messages = {"The input contains nan",  # for nan_policy="raise"
                      "Degrees of freedom <= 0 for slice",
                      "x and y should have at least 5 elements",
                      "Data must be at least length 3",
                      "The sample must contain at least two",
                      "x and y must contain at least two",
                      "division by zero",
                      "Mean of empty slice",
                      "Data passed to ks_2samp must not be empty",
                      "Not enough test observations",
                      "Not enough other observations",
                      "At least one observation is required",
                      "zero-size array to reduction operation maximum",
                      "`x` and `y` must be of nonzero size.",
                      "The exact distribution of the Wilcoxon test",
                      "Data input must not be empty"}

# If the message is one of these, results of the function may be inaccurate,
# but NaNs are not to be placed
inaccuracy_messages = {"Precision loss occurred in moment calculation",
                       "Sample size too small for normal approximation."}

# For some functions, nan_policy='propagate' should not just return NaNs
override_propagate_funcs = {stats.mode}


def _mixed_data_generator(n_samples, n_repetitions, axis, rng,
                          paired=False):
    # generate random samples to check the response of hypothesis tests to
    # samples with different (but broadcastable) shapes and various
    # nan patterns (e.g. all nans, some nans, no nans) along axis-slices

    data = []
    for i in range(n_samples):
        n_patterns = 6  # number of distinct nan patterns
        n_obs = 20 if paired else 20 + i  # observations per axis-slice
        x = np.ones((n_repetitions, n_patterns, n_obs)) * np.nan

        for j in range(n_repetitions):
            samples = x[j, :, :]

            # case 0: axis-slice with all nans (0 reals)
            # cases 1-3: axis-slice with 1-3 reals (the rest nans)
            # case 4: axis-slice with mostly (all but two) reals
            # case 5: axis slice with all reals
            for k, n_reals in enumerate([0, 1, 2, 3, n_obs-2, n_obs]):
                # for cases 1-3, need paired nansw  to be in the same place
                indices = rng.permutation(n_obs)[:n_reals]
                samples[k, indices] = rng.random(size=n_reals)

            # permute the axis-slices just to show that order doesn't matter
            samples[:] = rng.permutation(samples, axis=0)

        # For multi-sample tests, we want to test broadcasting and check
        # that nan policy works correctly for each nan pattern for each input.
        # This takes care of both simultaneosly.
        new_shape = [n_repetitions] + [1]*n_samples + [n_obs]
        new_shape[1 + i] = 6
        x = x.reshape(new_shape)

        x = np.moveaxis(x, -1, axis)
        data.append(x)
    return data


def _homogeneous_data_generator(n_samples, n_repetitions, axis, rng,
                                paired=False, all_nans=True):
    # generate random samples to check the response of hypothesis tests to
    # samples with different (but broadcastable) shapes and homogeneous
    # data (all nans or all finite)
    data = []
    for i in range(n_samples):
        n_obs = 20 if paired else 20 + i  # observations per axis-slice
        shape = [n_repetitions] + [1]*n_samples + [n_obs]
        shape[1 + i] = 2
        x = np.ones(shape) * np.nan if all_nans else rng.random(shape)
        x = np.moveaxis(x, -1, axis)
        data.append(x)
    return data


def nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=2,
                  nan_policy='raise', paired=False, _no_deco=True, **kwds):
    # Reference implementation for how `nan_policy` should work for 1d samples

    if nan_policy == 'raise':
        for sample in data1d:
            if np.any(np.isnan(sample)):
                raise ValueError("The input contains nan values")

    elif (nan_policy == 'propagate'
          and hypotest not in override_propagate_funcs):
        # For all hypothesis tests tested, returning nans is the right thing.
        # But many hypothesis tests don't propagate correctly (e.g. they treat
        # np.nan the same as np.inf, which doesn't make sense when ranks are
        # involved) so override that behavior here.
        for sample in data1d:
            if np.any(np.isnan(sample)):
                return np.full(n_outputs, np.nan)

    elif nan_policy == 'omit':
        # manually omit nans (or pairs in which at least one element is nan)
        if not paired:
            data1d = [sample[~np.isnan(sample)] for sample in data1d]
        else:
            nan_mask = np.isnan(data1d[0])
            for sample in data1d[1:]:
                nan_mask = np.logical_or(nan_mask, np.isnan(sample))
            data1d = [sample[~nan_mask] for sample in data1d]

    return unpacker(hypotest(*data1d, *args, _no_deco=_no_deco, **kwds))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
@pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
@pytest.mark.parametrize(("axis"), (1,))
@pytest.mark.parametrize(("data_generator"), ("mixed",))
def test_axis_nan_policy_fast(hypotest, args, kwds, n_samples, n_outputs,
                              paired, unpacker, nan_policy, axis,
                              data_generator):
    _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                          unpacker, nan_policy, axis, data_generator)


@pytest.mark.slow
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
@pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
@pytest.mark.parametrize(("axis"), range(-3, 3))
@pytest.mark.parametrize(("data_generator"),
                         ("all_nans", "all_finite", "mixed"))
def test_axis_nan_policy_full(hypotest, args, kwds, n_samples, n_outputs,
                              paired, unpacker, nan_policy, axis,
                              data_generator):
    _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                          unpacker, nan_policy, axis, data_generator)


def _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                          unpacker, nan_policy, axis, data_generator):
    # Tests the 1D and vectorized behavior of hypothesis tests against a
    # reference implementation (nan_policy_1d with np.ndenumerate)

    # Some hypothesis tests return a non-iterable that needs an `unpacker` to
    # extract the statistic and p-value. For those that don't:
    if not unpacker:
        def unpacker(res):
            return res

    rng = np.random.default_rng(0)

    # Generate multi-dimensional test data with all important combinations
    # of patterns of nans along `axis`
    n_repetitions = 3  # number of repetitions of each pattern
    data_gen_kwds = {'n_samples': n_samples, 'n_repetitions': n_repetitions,
                     'axis': axis, 'rng': rng, 'paired': paired}
    if data_generator == 'mixed':
        inherent_size = 6  # number of distinct types of patterns
        data = _mixed_data_generator(**data_gen_kwds)
    elif data_generator == 'all_nans':
        inherent_size = 2  # hard-coded in _homogeneous_data_generator
        data_gen_kwds['all_nans'] = True
        data = _homogeneous_data_generator(**data_gen_kwds)
    elif data_generator == 'all_finite':
        inherent_size = 2  # hard-coded in _homogeneous_data_generator
        data_gen_kwds['all_nans'] = False
        data = _homogeneous_data_generator(**data_gen_kwds)

    output_shape = [n_repetitions] + [inherent_size]*n_samples

    # To generate reference behavior to compare against, loop over the axis-
    # slices in data. Make indexing easier by moving `axis` to the end and
    # broadcasting all samples to the same shape.
    data_b = [np.moveaxis(sample, axis, -1) for sample in data]
    data_b = [np.broadcast_to(sample, output_shape + [sample.shape[-1]])
              for sample in data_b]
    statistics = np.zeros(output_shape)
    pvalues = np.zeros(output_shape)

    for i, _ in np.ndenumerate(statistics):
        data1d = [sample[i] for sample in data_b]
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                res1d = nan_policy_1d(hypotest, data1d, unpacker, *args,
                                      n_outputs=n_outputs,
                                      nan_policy=nan_policy,
                                      paired=paired, _no_deco=True, **kwds)

                # Eventually we'll check the results of a single, vectorized
                # call of `hypotest` against the arrays `statistics` and
                # `pvalues` populated using the reference `nan_policy_1d`.
                # But while we're at it, check the results of a 1D call to
                # `hypotest` against the reference `nan_policy_1d`.
                res1db = unpacker(hypotest(*data1d, *args,
                                           nan_policy=nan_policy, **kwds))
                assert_equal(res1db[0], res1d[0])
                if len(res1db) == 2:
                    assert_equal(res1db[1], res1d[1])

            # When there is not enough data in 1D samples, many existing
            # hypothesis tests raise errors instead of returning nans .
            # For vectorized calls, we put nans in the corresponding elements
            # of the output.
            except (RuntimeWarning, UserWarning, ValueError,
                    ZeroDivisionError) as e:

                # whatever it is, make sure same error is raised by both
                # `nan_policy_1d` and `hypotest`
                with pytest.raises(type(e), match=re.escape(str(e))):
                    nan_policy_1d(hypotest, data1d, unpacker, *args,
                                  n_outputs=n_outputs, nan_policy=nan_policy,
                                  paired=paired, _no_deco=True, **kwds)
                with pytest.raises(type(e), match=re.escape(str(e))):
                    hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)

                if any([str(e).startswith(message)
                        for message in too_small_messages]):
                    res1d = np.full(n_outputs, np.nan)
                elif any([str(e).startswith(message)
                          for message in inaccuracy_messages]):
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning)
                        sup.filter(UserWarning)
                        res1d = nan_policy_1d(hypotest, data1d, unpacker,
                                              *args, n_outputs=n_outputs,
                                              nan_policy=nan_policy,
                                              paired=paired, _no_deco=True,
                                              **kwds)
                else:
                    raise e
        statistics[i] = res1d[0]
        if len(res1d) == 2:
            pvalues[i] = res1d[1]

    # Perform a vectorized call to the hypothesis test.
    # If `nan_policy == 'raise'`, check that it raises the appropriate error.
    # If not, compare against the output against `statistics` and `pvalues`
    if nan_policy == 'raise' and not data_generator == "all_finite":
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            hypotest(*data, axis=axis, nan_policy=nan_policy, *args, **kwds)

    else:
        with suppress_warnings() as sup, \
             np.errstate(divide='ignore', invalid='ignore'):
            sup.filter(RuntimeWarning, "Precision loss occurred in moment")
            sup.filter(UserWarning, "Sample size too small for normal "
                                    "approximation.")
            res = unpacker(hypotest(*data, axis=axis, nan_policy=nan_policy,
                                    *args, **kwds))
        assert_allclose(res[0], statistics, rtol=1e-15)
        assert_equal(res[0].dtype, statistics.dtype)

        if len(res) == 2:
            assert_allclose(res[1], pvalues, rtol=1e-15)
            assert_equal(res[1].dtype, pvalues.dtype)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
@pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
@pytest.mark.parametrize(("data_generator"),
                         ("all_nans", "all_finite", "mixed", "empty"))
def test_axis_nan_policy_axis_is_None(hypotest, args, kwds, n_samples,
                                      n_outputs, paired, unpacker, nan_policy,
                                      data_generator):
    # check for correct behavior when `axis=None`

    if not unpacker:
        def unpacker(res):
            return res

    rng = np.random.default_rng(0)

    if data_generator == "empty":
        data = [rng.random((2, 0)) for i in range(n_samples)]
    else:
        data = [rng.random((2, 20)) for i in range(n_samples)]

    if data_generator == "mixed":
        masks = [rng.random((2, 20)) > 0.9 for i in range(n_samples)]
        for sample, mask in zip(data, masks):
            sample[mask] = np.nan
    elif data_generator == "all_nans":
        data = [sample * np.nan for sample in data]

    data_raveled = [sample.ravel() for sample in data]

    if nan_policy == 'raise' and data_generator not in {"all_finite", "empty"}:
        message = 'The input contains nan values'

        # check for correct behavior whether or not data is 1d to begin with
        with pytest.raises(ValueError, match=message):
            hypotest(*data, axis=None, nan_policy=nan_policy,
                     *args, **kwds)
        with pytest.raises(ValueError, match=message):
            hypotest(*data_raveled, axis=None, nan_policy=nan_policy,
                     *args, **kwds)

    else:
        # behavior of reference implementation with 1d input, hypotest with 1d
        # input, and hypotest with Nd input should match, whether that means
        # that outputs are equal or they raise the same exception

        ea_str, eb_str, ec_str = None, None, None
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                res1da = nan_policy_1d(hypotest, data_raveled, unpacker, *args,
                                       n_outputs=n_outputs,
                                       nan_policy=nan_policy, paired=paired,
                                       _no_deco=True, **kwds)
            except (RuntimeWarning, ValueError, ZeroDivisionError) as ea:
                ea_str = str(ea)

            try:
                res1db = unpacker(hypotest(*data_raveled, *args,
                                           nan_policy=nan_policy, **kwds))
            except (RuntimeWarning, ValueError, ZeroDivisionError) as eb:
                eb_str = str(eb)

            try:
                res1dc = unpacker(hypotest(*data, *args, axis=None,
                                           nan_policy=nan_policy, **kwds))
            except (RuntimeWarning, ValueError, ZeroDivisionError) as ec:
                ec_str = str(ec)

            if ea_str or eb_str or ec_str:
                assert any([str(ea_str).startswith(message)
                            for message in too_small_messages])
                assert ea_str == eb_str == ec_str
            else:
                assert_equal(res1db, res1da)
                assert_equal(res1dc, res1da)
                for item in list(res1da) + list(res1db) + list(res1dc):
                    # Most functions naturally return NumPy numbers, which
                    # are drop-in replacements for the Python versions but with
                    # desirable attributes. Make sure this is consistent.
                    assert np.issubdtype(item.dtype, np.number)

# Test keepdims for:
#     - single-output and multi-output functions (gmean and mannwhitneyu)
#     - Axis negative, positive, None, and tuple
#     - 1D with no NaNs
#     - 1D with NaN propagation
#     - Zero-sized output
@pytest.mark.parametrize("nan_policy", ("omit", "propagate"))
@pytest.mark.parametrize(
    ("hypotest", "args", "kwds", "n_samples", "unpacker"),
    ((stats.gmean, tuple(), dict(), 1, lambda x: (x,)),
     (stats.mannwhitneyu, tuple(), {'method': 'asymptotic'}, 2, None))
)
@pytest.mark.parametrize(
    ("sample_shape", "axis_cases"),
    (((2, 3, 3, 4), (None, 0, -1, (0, 2), (1, -1), (3, 1, 2, 0))),
     ((10, ), (0, -1)),
     ((20, 0), (0, 1)))
)
def test_keepdims(hypotest, args, kwds, n_samples, unpacker,
                  sample_shape, axis_cases, nan_policy):
    # test if keepdims parameter works correctly
    if not unpacker:
        def unpacker(res):
            return res
    rng = np.random.default_rng(0)
    data = [rng.random(sample_shape) for _ in range(n_samples)]
    nan_data = [sample.copy() for sample in data]
    nan_mask = [rng.random(sample_shape) < 0.2 for _ in range(n_samples)]
    for sample, mask in zip(nan_data, nan_mask):
        sample[mask] = np.nan
    for axis in axis_cases:
        expected_shape = list(sample_shape)
        if axis is None:
            expected_shape = np.ones(len(sample_shape))
        else:
            if isinstance(axis, int):
                expected_shape[axis] = 1
            else:
                for ax in axis:
                    expected_shape[ax] = 1
        expected_shape = tuple(expected_shape)
        res = unpacker(hypotest(*data, *args, axis=axis, keepdims=True,
                                **kwds))
        res_base = unpacker(hypotest(*data, *args, axis=axis, keepdims=False,
                                     **kwds))
        nan_res = unpacker(hypotest(*nan_data, *args, axis=axis,
                                    keepdims=True, nan_policy=nan_policy,
                                    **kwds))
        nan_res_base = unpacker(hypotest(*nan_data, *args, axis=axis,
                                         keepdims=False,
                                         nan_policy=nan_policy, **kwds))
        for r, r_base, rn, rn_base in zip(res, res_base, nan_res,
                                          nan_res_base):
            assert r.shape == expected_shape
            r = np.squeeze(r, axis=axis)
            assert_equal(r, r_base)
            assert rn.shape == expected_shape
            rn = np.squeeze(rn, axis=axis)
            assert_equal(rn, rn_base)


@pytest.mark.parametrize(("fun", "nsamp"),
                         [(stats.kstat, 1),
                          (stats.kstatvar, 1)])
def test_hypotest_back_compat_no_axis(fun, nsamp):
    m, n = 8, 9

    rng = np.random.default_rng(0)
    x = rng.random((nsamp, m, n))
    res = fun(*x)
    res2 = fun(*x, _no_deco=True)
    res3 = fun([xi.ravel() for xi in x])
    assert_equal(res, res2)
    assert_equal(res, res3)


@pytest.mark.parametrize(("axis"), (0, 1, 2))
def test_axis_nan_policy_decorated_positional_axis(axis):
    # Test for correct behavior of function decorated with
    # _axis_nan_policy_decorator whether `axis` is provided as positional or
    # keyword argument

    shape = (8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    y = rng.random(shape)
    res1 = stats.mannwhitneyu(x, y, True, 'two-sided', axis)
    res2 = stats.mannwhitneyu(x, y, True, 'two-sided', axis=axis)
    assert_equal(res1, res2)

    message = "mannwhitneyu() got multiple values for argument 'axis'"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.mannwhitneyu(x, y, True, 'two-sided', axis, axis=axis)


def test_axis_nan_policy_decorated_positional_args():
    # Test for correct behavior of function decorated with
    # _axis_nan_policy_decorator when function accepts *args

    shape = (3, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    x[0, 0, 0, 0] = np.nan
    stats.kruskal(*x)

    message = "kruskal() got an unexpected keyword argument 'samples'"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(samples=x)

    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(*x, samples=x)


def test_axis_nan_policy_decorated_keyword_samples():
    # Test for correct behavior of function decorated with
    # _axis_nan_policy_decorator whether samples are provided as positional or
    # keyword arguments

    shape = (2, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    x[0, 0, 0, 0] = np.nan
    res1 = stats.mannwhitneyu(*x)
    res2 = stats.mannwhitneyu(x=x[0], y=x[1])
    assert_equal(res1, res2)

    message = "mannwhitneyu() got multiple values for argument"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.mannwhitneyu(*x, x=x[0], y=x[1])


@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
def test_axis_nan_policy_decorated_pickled(hypotest, args, kwds, n_samples,
                                           n_outputs, paired, unpacker):
    if "ttest_ci" in hypotest.__name__:
        pytest.skip("Can't pickle functions defined within functions.")

    rng = np.random.default_rng(0)

    # Some hypothesis tests return a non-iterable that needs an `unpacker` to
    # extract the statistic and p-value. For those that don't:
    if not unpacker:
        def unpacker(res):
            return res

    data = rng.uniform(size=(n_samples, 2, 30))
    pickled_hypotest = pickle.dumps(hypotest)
    unpickled_hypotest = pickle.loads(pickled_hypotest)
    res1 = unpacker(hypotest(*data, *args, axis=-1, **kwds))
    res2 = unpacker(unpickled_hypotest(*data, *args, axis=-1, **kwds))
    assert_allclose(res1, res2, rtol=1e-12)


def test_check_empty_inputs():
    # Test that _check_empty_inputs is doing its job, at least for single-
    # sample inputs. (Multi-sample functionality is tested below.)
    # If the input sample is not empty, it should return None.
    # If the input sample is empty, it should return an array of NaNs or an
    # empty array of appropriate shape. np.mean is used as a reference for the
    # output because, like the statistics calculated by these functions,
    # it works along and "consumes" `axis` but preserves the other axes.
    for i in range(5):
        for combo in combinations_with_replacement([0, 1, 2], i):
            for axis in range(len(combo)):
                samples = (np.zeros(combo),)
                output = stats._axis_nan_policy._check_empty_inputs(samples,
                                                                    axis)
                if output is not None:
                    with np.testing.suppress_warnings() as sup:
                        sup.filter(RuntimeWarning, "Mean of empty slice.")
                        sup.filter(RuntimeWarning, "invalid value encountered")
                        reference = samples[0].mean(axis=axis)
                    np.testing.assert_equal(output, reference)


def _check_arrays_broadcastable(arrays, axis):
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    # "When operating on two arrays, NumPy compares their shapes element-wise.
    # It starts with the trailing (i.e. rightmost) dimensions and works its
    # way left.
    # Two dimensions are compatible when
    # 1. they are equal, or
    # 2. one of them is 1
    # ...
    # Arrays do not need to have the same number of dimensions."
    # (Clarification: if the arrays are compatible according to the criteria
    #  above and an array runs out of dimensions, it is still compatible.)
    # Below, we follow the rules above except ignoring `axis`

    n_dims = max([arr.ndim for arr in arrays])
    if axis is not None:
        # convert to negative axis
        axis = (-n_dims + axis) if axis >= 0 else axis

    for dim in range(1, n_dims+1):  # we'll index from -1 to -n_dims, inclusive
        if -dim == axis:
            continue  # ignore lengths along `axis`

        dim_lengths = set()
        for arr in arrays:
            if dim <= arr.ndim and arr.shape[-dim] != 1:
                dim_lengths.add(arr.shape[-dim])

        if len(dim_lengths) > 1:
            return False
    return True


@pytest.mark.slow
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
def test_empty(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker):
    # test for correct output shape when at least one input is empty

    if hypotest in override_propagate_funcs:
        reason = "Doesn't follow the usual pattern. Tested separately."
        pytest.skip(reason=reason)

    if unpacker is None:
        unpacker = lambda res: (res[0], res[1])  # noqa: E731

    def small_data_generator(n_samples, n_dims):

        def small_sample_generator(n_dims):
            # return all possible "small" arrays in up to n_dim dimensions
            for i in n_dims:
                # "small" means with size along dimension either 0 or 1
                for combo in combinations_with_replacement([0, 1, 2], i):
                    yield np.zeros(combo)

        # yield all possible combinations of small samples
        gens = [small_sample_generator(n_dims) for i in range(n_samples)]
        yield from product(*gens)

    n_dims = [2, 3]
    for samples in small_data_generator(n_samples, n_dims):

        # this test is only for arrays of zero size
        if not any(sample.size == 0 for sample in samples):
            continue

        max_axis = max(sample.ndim for sample in samples)

        # need to test for all valid values of `axis` parameter, too
        for axis in range(-max_axis, max_axis):

            try:
                # After broadcasting, all arrays are the same shape, so
                # the shape of the output should be the same as a single-
                # sample statistic. Use np.mean as a reference.
                concat = stats._stats_py._broadcast_concatenate(samples, axis)
                with np.testing.suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, "Mean of empty slice.")
                    sup.filter(RuntimeWarning, "invalid value encountered")
                    expected = np.mean(concat, axis=axis) * np.nan

                res = hypotest(*samples, *args, axis=axis, **kwds)
                res = unpacker(res)

                for i in range(n_outputs):
                    assert_equal(res[i], expected)

            except ValueError:
                # confirm that the arrays truly are not broadcastable
                assert not _check_arrays_broadcastable(samples, axis)

                # confirm that _both_ `_broadcast_concatenate` and `hypotest`
                # produce this information.
                message = "Array shapes are incompatible for broadcasting."
                with pytest.raises(ValueError, match=message):
                    stats._stats_py._broadcast_concatenate(samples, axis)
                with pytest.raises(ValueError, match=message):
                    hypotest(*samples, *args, axis=axis, **kwds)


def test_masked_array_2_sentinel_array():
    # prepare arrays
    np.random.seed(0)
    A = np.random.rand(10, 11, 12)
    B = np.random.rand(12)
    mask = A < 0.5
    A = np.ma.masked_array(A, mask)

    # set arbitrary elements to special values
    # (these values might have been considered for use as sentinel values)
    max_float = np.finfo(np.float64).max
    max_float2 = np.nextafter(max_float, -np.inf)
    max_float3 = np.nextafter(max_float2, -np.inf)
    A[3, 4, 1] = np.nan
    A[4, 5, 2] = np.inf
    A[5, 6, 3] = max_float
    B[8] = np.nan
    B[7] = np.inf
    B[6] = max_float2

    # convert masked A to array with sentinel value, don't modify B
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([A, B])
    A_out, B_out = out_arrays

    # check that good sentinel value was chosen (according to intended logic)
    assert (sentinel != max_float) and (sentinel != max_float2)
    assert sentinel == max_float3

    # check that output arrays are as intended
    A_reference = A.data
    A_reference[A.mask] = sentinel
    np.testing.assert_array_equal(A_out, A_reference)
    assert B_out is B


def test_masked_dtype():
    # When _masked_arrays_2_sentinel_arrays was first added, it always
    # upcast the arrays to np.float64. After gh16662, check expected promotion
    # and that the expected sentinel is found.

    # these are important because the max of the promoted dtype is the first
    # candidate to be the sentinel value
    max16 = np.iinfo(np.int16).max
    max128c = np.finfo(np.complex128).max

    # a is a regular array, b has masked elements, and c has no masked elements
    a = np.array([1, 2, max16], dtype=np.int16)
    b = np.ma.array([1, 2, 1], dtype=np.int8, mask=[0, 1, 0])
    c = np.ma.array([1, 2, 1], dtype=np.complex128, mask=[0, 0, 0])

    # check integer masked -> sentinel conversion
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a, b])
    a_out, b_out = out_arrays
    assert sentinel == max16-1  # not max16 because max16 was in the data
    assert b_out.dtype == np.int16  # check expected promotion
    assert_allclose(b_out, [b[0], sentinel, b[-1]])  # check sentinel placement
    assert a_out is a  # not a masked array, so left untouched
    assert not isinstance(b_out, np.ma.MaskedArray)  # b became regular array

    # similarly with complex
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([b, c])
    b_out, c_out = out_arrays
    assert sentinel == max128c  # max128c was not in the data
    assert b_out.dtype == np.complex128  # b got promoted
    assert_allclose(b_out, [b[0], sentinel, b[-1]])  # check sentinel placement
    assert not isinstance(b_out, np.ma.MaskedArray)  # b became regular array
    assert not isinstance(c_out, np.ma.MaskedArray)  # c became regular array

    # Also, check edge case when a sentinel value cannot be found in the data
    min8, max8 = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    a = np.arange(min8, max8+1, dtype=np.int8)  # use all possible values
    mask1 = np.zeros_like(a, dtype=bool)
    mask0 = np.zeros_like(a, dtype=bool)

    # a masked value can be used as the sentinel
    mask1[1] = True
    a1 = np.ma.array(a, mask=mask1)
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a1])
    assert sentinel == min8+1

    # unless it's the smallest possible; skipped for simiplicity (see code)
    mask0[0] = True
    a0 = np.ma.array(a, mask=mask0)
    message = "This function replaces masked elements with sentinel..."
    with pytest.raises(ValueError, match=message):
        _masked_arrays_2_sentinel_arrays([a0])

    # test that dtype is preserved in functions
    a = np.ma.array([1, 2, 3], mask=[0, 1, 0], dtype=np.float32)
    assert stats.gmean(a).dtype == np.float32


def test_masked_stat_1d():
    # basic test of _axis_nan_policy_factory with 1D masked sample
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]
    res = stats.mannwhitneyu(males, females)

    # same result when extra nan is omitted
    females2 = [20, 11, 17, np.nan, 12]
    res2 = stats.mannwhitneyu(males, females2, nan_policy='omit')
    np.testing.assert_array_equal(res2, res)

    # same result when extra element is masked
    females3 = [20, 11, 17, 1000, 12]
    mask3 = [False, False, False, True, False]
    females3 = np.ma.masked_array(females3, mask=mask3)
    res3 = stats.mannwhitneyu(males, females3)
    np.testing.assert_array_equal(res3, res)

    # same result when extra nan is omitted and additional element is masked
    females4 = [20, 11, 17, np.nan, 1000, 12]
    mask4 = [False, False, False, False, True, False]
    females4 = np.ma.masked_array(females4, mask=mask4)
    res4 = stats.mannwhitneyu(males, females4, nan_policy='omit')
    np.testing.assert_array_equal(res4, res)

    # same result when extra elements, including nan, are masked
    females5 = [20, 11, 17, np.nan, 1000, 12]
    mask5 = [False, False, False, True, True, False]
    females5 = np.ma.masked_array(females5, mask=mask5)
    res5 = stats.mannwhitneyu(males, females5, nan_policy='propagate')
    res6 = stats.mannwhitneyu(males, females5, nan_policy='raise')
    np.testing.assert_array_equal(res5, res)
    np.testing.assert_array_equal(res6, res)


@pytest.mark.parametrize(("axis"), range(-3, 3))
def test_masked_stat_3d(axis):
    # basic test of _axis_nan_policy_factory with 3D masked sample
    np.random.seed(0)
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(4, 5)
    c = np.random.rand(4, 1)

    mask_a = a < 0.1
    mask_c = [False, False, False, True]
    a_masked = np.ma.masked_array(a, mask=mask_a)
    c_masked = np.ma.masked_array(c, mask=mask_c)

    a_nans = a.copy()
    a_nans[mask_a] = np.nan
    c_nans = c.copy()
    c_nans[mask_c] = np.nan

    res = stats.kruskal(a_nans, b, c_nans, nan_policy='omit', axis=axis)
    res2 = stats.kruskal(a_masked, b, c_masked, axis=axis)
    np.testing.assert_array_equal(res, res2)


def test_mixed_mask_nan_1():
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # omitting samples with masks and nan_policy='omit' are equivalent
    # also checks paired-sample sentinel value removal
    m, n = 3, 20
    axis = -1

    np.random.seed(0)
    a = np.random.rand(m, n)
    b = np.random.rand(m, n)
    mask_a1 = np.random.rand(m, n) < 0.2
    mask_a2 = np.random.rand(m, n) < 0.1
    mask_b1 = np.random.rand(m, n) < 0.15
    mask_b2 = np.random.rand(m, n) < 0.15
    mask_a1[2, :] = True

    a_nans = a.copy()
    b_nans = b.copy()
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan

    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan

    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan

    a_masked3 = np.ma.masked_array(a, mask=(mask_a1 | mask_a2))
    b_masked3 = np.ma.masked_array(b, mask=(mask_b1 | mask_b2))

    res = stats.wilcoxon(a_nans, b_nans, nan_policy='omit', axis=axis)
    res1 = stats.wilcoxon(a_masked1, b_masked1, nan_policy='omit', axis=axis)
    res2 = stats.wilcoxon(a_masked2, b_masked2, nan_policy='omit', axis=axis)
    res3 = stats.wilcoxon(a_masked3, b_masked3, nan_policy='raise', axis=axis)
    res4 = stats.wilcoxon(a_masked3, b_masked3,
                          nan_policy='propagate', axis=axis)

    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)


def test_mixed_mask_nan_2():
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # check for expected interaction between masks and nans

    # Cases here are
    # [mixed nan/mask, all nans, all masked,
    # unmasked nan, masked nan, unmasked non-nan]
    a = [[1, np.nan, 2], [np.nan, np.nan, np.nan], [1, 2, 3],
         [1, np.nan, 3], [1, np.nan, 3], [1, 2, 3]]
    mask = [[1, 0, 1], [0, 0, 0], [1, 1, 1],
            [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    a_masked = np.ma.masked_array(a, mask=mask)
    b = [[4, 5, 6]]
    ref1 = stats.ranksums([1, 3], [4, 5, 6])
    ref2 = stats.ranksums([1, 2, 3], [4, 5, 6])

    # nan_policy = 'omit'
    # all elements are removed from first three rows
    # middle element is removed from fourth and fifth rows
    # no elements removed from last row
    res = stats.ranksums(a_masked, b, nan_policy='omit', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan,
                ref1.statistic, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan,
             ref1.pvalue, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)

    # nan_policy = 'propagate'
    # nans propagate in first, second, and fourth row
    # all elements are removed by mask from third row
    # middle element is removed from fifth row
    # no elements removed from last row
    res = stats.ranksums(a_masked, b, nan_policy='propagate', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan,
                np.nan, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan,
             np.nan, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)


def test_axis_None_vs_tuple():
    # `axis` `None` should be equivalent to tuple with all axes
    shape = (3, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    res = stats.kruskal(*x, axis=None)
    res2 = stats.kruskal(*x, axis=(0, 1, 2))
    np.testing.assert_array_equal(res, res2)


def test_axis_None_vs_tuple_with_broadcasting():
    # `axis` `None` should be equivalent to tuple with all axes,
    # which should be equivalent to raveling the arrays before passing them
    rng = np.random.default_rng(0)
    x = rng.random((5, 1))
    y = rng.random((1, 5))
    x2, y2 = np.broadcast_arrays(x, y)

    res0 = stats.mannwhitneyu(x.ravel(), y.ravel())
    res1 = stats.mannwhitneyu(x, y, axis=None)
    res2 = stats.mannwhitneyu(x, y, axis=(0, 1))
    res3 = stats.mannwhitneyu(x2.ravel(), y2.ravel())

    assert res1 == res0
    assert res2 == res0
    assert res3 != res0


@pytest.mark.parametrize(("axis"),
                         list(permutations(range(-3, 3), 2)) + [(-4, 1)])
def test_other_axis_tuples(axis):
    # Check that _axis_nan_policy_factory treates all `axis` tuples as expected
    rng = np.random.default_rng(0)
    shape_x = (4, 5, 6)
    shape_y = (1, 6)
    x = rng.random(shape_x)
    y = rng.random(shape_y)
    axis_original = axis

    # convert axis elements to positive
    axis = tuple([(i if i >= 0 else 3 + i) for i in axis])
    axis = sorted(axis)

    if len(set(axis)) != len(axis):
        message = "`axis` must contain only distinct elements"
        with pytest.raises(np.AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return

    if axis[0] < 0 or axis[-1] > 2:
        message = "`axis` is out of bounds for array of dimension 3"
        with pytest.raises(np.AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return

    res = stats.mannwhitneyu(x, y, axis=axis_original)

    # reference behavior
    not_axis = {0, 1, 2} - set(axis)  # which axis is not part of `axis`
    not_axis = next(iter(not_axis))  # take it out of the set

    x2 = x
    shape_y_broadcasted = [1, 1, 6]
    shape_y_broadcasted[not_axis] = shape_x[not_axis]
    y2 = np.broadcast_to(y, shape_y_broadcasted)

    m = x2.shape[not_axis]
    x2 = np.moveaxis(x2, axis, (1, 2))
    y2 = np.moveaxis(y2, axis, (1, 2))
    x2 = np.reshape(x2, (m, -1))
    y2 = np.reshape(y2, (m, -1))
    res2 = stats.mannwhitneyu(x2, y2, axis=1)

    np.testing.assert_array_equal(res, res2)


@pytest.mark.parametrize(("weighted_fun_name"), ["gmean", "hmean", "pmean"])
def test_mean_mixed_mask_nan_weights(weighted_fun_name):
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # omitting samples with masks and nan_policy='omit' are equivalent
    # also checks paired-sample sentinel value removal

    if weighted_fun_name == 'pmean':
        def weighted_fun(a, **kwargs):
            return stats.pmean(a, p=0.42, **kwargs)
    else:
        weighted_fun = getattr(stats, weighted_fun_name)

    m, n = 3, 20
    axis = -1

    rng = np.random.default_rng(6541968121)
    a = rng.uniform(size=(m, n))
    b = rng.uniform(size=(m, n))
    mask_a1 = rng.uniform(size=(m, n)) < 0.2
    mask_a2 = rng.uniform(size=(m, n)) < 0.1
    mask_b1 = rng.uniform(size=(m, n)) < 0.15
    mask_b2 = rng.uniform(size=(m, n)) < 0.15
    mask_a1[2, :] = True

    a_nans = a.copy()
    b_nans = b.copy()
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan

    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan

    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan

    a_masked3 = np.ma.masked_array(a, mask=(mask_a1 | mask_a2))
    b_masked3 = np.ma.masked_array(b, mask=(mask_b1 | mask_b2))

    mask_all = (mask_a1 | mask_a2 | mask_b1 | mask_b2)
    a_masked4 = np.ma.masked_array(a, mask=mask_all)
    b_masked4 = np.ma.masked_array(b, mask=mask_all)

    with np.testing.suppress_warnings() as sup:
        message = 'invalid value encountered'
        sup.filter(RuntimeWarning, message)
        res = weighted_fun(a_nans, weights=b_nans,
                           nan_policy='omit', axis=axis)
        res1 = weighted_fun(a_masked1, weights=b_masked1,
                            nan_policy='omit', axis=axis)
        res2 = weighted_fun(a_masked2, weights=b_masked2,
                            nan_policy='omit', axis=axis)
        res3 = weighted_fun(a_masked3, weights=b_masked3,
                            nan_policy='raise', axis=axis)
        res4 = weighted_fun(a_masked3, weights=b_masked3,
                            nan_policy='propagate', axis=axis)
        # Would test with a_masked3/b_masked3, but there is a bug in np.average
        # that causes a bug in _no_deco mean with masked weights. Would use
        # np.ma.average, but that causes other problems. See numpy/numpy#7330.
        if weighted_fun_name not in {'pmean', 'gmean'}:
            weighted_fun_ma = getattr(stats.mstats, weighted_fun_name)
            res5 = weighted_fun_ma(a_masked4, weights=b_masked4,
                                   axis=axis, _no_deco=True)

    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)
    if weighted_fun_name not in {'pmean', 'gmean'}:
        # _no_deco mean returns masked array, last element was masked
        np.testing.assert_allclose(res5.compressed(), res[~np.isnan(res)])


def test_raise_invalid_args_g17713():
    # other cases are handled in:
    # test_axis_nan_policy_decorated_positional_axis - multiple values for arg
    # test_axis_nan_policy_decorated_positional_args - unexpected kwd arg
    message = "got an unexpected keyword argument"
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], invalid_arg=True)

    message = " got multiple values for argument"
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], a=True)

    message = "missing 1 required positional argument"
    with pytest.raises(TypeError, match=message):
        stats.gmean()

    message = "takes from 1 to 4 positional arguments but 5 were given"
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], 0, float, [1, 1, 1], 10)

@pytest.mark.parametrize(
    'dtype',
    (list(np.typecodes['Float']
          + np.typecodes['Integer']
          + np.typecodes['Complex'])))
def test_array_like_input(dtype):
    # Check that `_axis_nan_policy`-decorated functions work with custom
    # containers that are coercible to numeric arrays

    class ArrLike():
        def __init__(self, x):
            self._x = x

        def __array__(self):
            return np.asarray(x, dtype=dtype)

    x = [1]*2 + [3, 4, 5]
    res = stats.mode(ArrLike(x))
    assert res.mode == 1
    assert res.count == 2
