import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation


class TestVariation:
    """
    Test class for scipy.stats.variation
    """

    def test_ddof(self):
        x = np.arange(9.0)
        assert_allclose(variation(x, ddof=1), np.sqrt(60/8)/4)

    @pytest.mark.parametrize('sgn', [1, -1])
    def test_sign(self, sgn):
        x = np.array([1, 2, 3, 4, 5])
        v = variation(sgn*x)
        expected = sgn*np.sqrt(2)/3
        assert_allclose(v, expected, rtol=1e-10)

    def test_scalar(self):
        # A scalar is treated like a 1-d sequence with length 1.
        assert_equal(variation(4.0), 0.0)

    @pytest.mark.parametrize('nan_policy, expected',
                             [('propagate', np.nan),
                              ('omit', np.sqrt(20/3)/4)])
    def test_variation_nan(self, nan_policy, expected):
        x = np.arange(10.)
        x[9] = np.nan
        assert_allclose(variation(x, nan_policy=nan_policy), expected)

    def test_nan_policy_raise(self):
        x = np.array([1.0, 2.0, np.nan, 3.0])
        with pytest.raises(ValueError, match='input contains nan'):
            variation(x, nan_policy='raise')

    def test_bad_nan_policy(self):
        with pytest.raises(ValueError, match='must be one of'):
            variation([1, 2, 3], nan_policy='foobar')

    def test_keepdims(self):
        x = np.arange(10).reshape(2, 5)
        y = variation(x, axis=1, keepdims=True)
        expected = np.array([[np.sqrt(2)/2],
                             [np.sqrt(2)/7]])
        assert_allclose(y, expected)

    @pytest.mark.parametrize('axis, expected',
                             [(0, np.empty((1, 0))),
                              (1, np.full((5, 1), fill_value=np.nan))])
    def test_keepdims_size0(self, axis, expected):
        x = np.zeros((5, 0))
        y = variation(x, axis=axis, keepdims=True)
        assert_equal(y, expected)

    @pytest.mark.parametrize('incr, expected_fill', [(0, np.inf), (1, np.nan)])
    def test_keepdims_and_ddof_eq_len_plus_incr(self, incr, expected_fill):
        x = np.array([[1, 1, 2, 2], [1, 2, 3, 3]])
        y = variation(x, axis=1, ddof=x.shape[1] + incr, keepdims=True)
        assert_equal(y, np.full((2, 1), fill_value=expected_fill))

    def test_propagate_nan(self):
        # Check that the shape of the result is the same for inputs
        # with and without nans, cf gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        v = variation(a, axis=1, nan_policy="propagate")
        assert_allclose(v, [np.sqrt(5/4)/1.5, np.nan], atol=1e-15)

    def test_axis_none(self):
        # Check that `variation` computes the result on the flattened
        # input when axis is None.
        y = variation([[0, 1], [2, 3]], axis=None)
        assert_allclose(y, np.sqrt(5/4)/1.5)

    def test_bad_axis(self):
        # Check that an invalid axis raises np.AxisError.
        x = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(np.AxisError):
            variation(x, axis=10)

    def test_mean_zero(self):
        # Check that `variation` returns inf for a sequence that is not
        # identically zero but whose mean is zero.
        x = np.array([10, -3, 1, -4, -4])
        y = variation(x)
        assert_equal(y, np.inf)

        x2 = np.array([x, -10*x])
        y2 = variation(x2, axis=1)
        assert_equal(y2, [np.inf, np.inf])

    @pytest.mark.parametrize('x', [np.zeros(5), [], [1, 2, np.inf, 9]])
    def test_return_nan(self, x):
        # Test some cases where `variation` returns nan.
        y = variation(x)
        assert_equal(y, np.nan)

    @pytest.mark.parametrize('axis, expected',
                             [(0, []), (1, [np.nan]*3), (None, np.nan)])
    def test_2d_size_zero_with_axis(self, axis, expected):
        x = np.empty((3, 0))
        y = variation(x, axis=axis)
        assert_equal(y, expected)

    def test_neg_inf(self):
        # Edge case that produces -inf: ddof equals the number of non-nan
        # values, the values are not constant, and the mean is negative.
        x1 = np.array([-3, -5])
        assert_equal(variation(x1, ddof=2), -np.inf)

        x2 = np.array([[np.nan, 1, -10, np.nan],
                       [-20, -3, np.nan, np.nan]])
        assert_equal(variation(x2, axis=1, ddof=2, nan_policy='omit'),
                     [-np.inf, -np.inf])

    @pytest.mark.parametrize("nan_policy", ['propagate', 'omit'])
    def test_combined_edge_cases(self, nan_policy):
        x = np.array([[0, 10, np.nan, 1],
                      [0, -5, np.nan, 2],
                      [0, -5, np.nan, 3]])
        y = variation(x, axis=0, nan_policy=nan_policy)
        assert_allclose(y, [np.nan, np.inf, np.nan, np.sqrt(2/3)/2])

    @pytest.mark.parametrize(
        'ddof, expected',
        [(0, [np.sqrt(1/6), np.sqrt(5/8), np.inf, 0, np.nan, 0.0, np.nan]),
         (1, [0.5, np.sqrt(5/6), np.inf, 0, np.nan, 0, np.nan]),
         (2, [np.sqrt(0.5), np.sqrt(5/4), np.inf, np.nan, np.nan, 0, np.nan])]
    )
    def test_more_nan_policy_omit_tests(self, ddof, expected):
        # The slightly strange formatting in the follow array is my attempt to
        # maintain a clean tabular arrangement of the data while satisfying
        # the demands of pycodestyle.  Currently, E201 and E241 are not
        # disabled by the `# noqa` annotation.
        nan = np.nan
        x = np.array([[1.0, 2.0, nan, 3.0],
                      [0.0, 4.0, 3.0, 1.0],
                      [nan, -.5, 0.5, nan],
                      [nan, 9.0, 9.0, nan],
                      [nan, nan, nan, nan],
                      [3.0, 3.0, 3.0, 3.0],
                      [0.0, 0.0, 0.0, 0.0]])
        v = variation(x, axis=1, ddof=ddof, nan_policy='omit')
        assert_allclose(v, expected)

    def test_variation_ddof(self):
        # test variation with delta degrees of freedom
        # regression test for gh-13341
        a = np.array([1, 2, 3, 4, 5])
        nan_a = np.array([1, 2, 3, np.nan, 4, 5, np.nan])
        y = variation(a, ddof=1)
        nan_y = variation(nan_a, nan_policy="omit", ddof=1)
        assert_allclose(y, np.sqrt(5/2)/3)
        assert y == nan_y
