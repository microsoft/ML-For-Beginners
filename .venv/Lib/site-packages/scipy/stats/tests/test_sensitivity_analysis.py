import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest

from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
    BootstrapSobolResult, f_ishigami, sample_AB, sample_A_B
)


@pytest.fixture(scope='session')
def ishigami_ref_indices():
    """Reference values for Ishigami from Saltelli2007.

    Chapter 4, exercise 5 pages 179-182.
    """
    a = 7.
    b = 0.1

    var = 0.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18
    v1 = 0.5 + b*np.pi**4/5 + b**2*np.pi**8/50
    v2 = a**2/8
    v3 = 0
    v12 = 0
    # v13: mistake in the book, see other derivations e.g. in 10.1002/nme.4856
    v13 = b**2*np.pi**8*8/225
    v23 = 0

    s_first = np.array([v1, v2, v3])/var
    s_second = np.array([
        [0., 0., v13],
        [v12, 0., v23],
        [v13, v23, 0.]
    ])/var
    s_total = s_first + s_second.sum(axis=1)

    return s_first, s_total


def f_ishigami_vec(x):
    """Output of shape (2, n)."""
    res = f_ishigami(x)
    return res, res


class TestSobolIndices:

    dists = [
        stats.uniform(loc=-np.pi, scale=2*np.pi)  # type: ignore[attr-defined]
    ] * 3

    def test_sample_AB(self):
        # (d, n)
        A = np.array(
            [[1, 4, 7, 10],
             [2, 5, 8, 11],
             [3, 6, 9, 12]]
        )
        B = A + 100
        # (d, d, n)
        ref = np.array(
            [[[101, 104, 107, 110],
              [2, 5, 8, 11],
              [3, 6, 9, 12]],
             [[1, 4, 7, 10],
              [102, 105, 108, 111],
              [3, 6, 9, 12]],
             [[1, 4, 7, 10],
              [2, 5, 8, 11],
              [103, 106, 109, 112]]]
        )
        AB = sample_AB(A=A, B=B)
        assert_allclose(AB, ref)

    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    @pytest.mark.parametrize(
        'func',
        [f_ishigami, pytest.param(f_ishigami_vec, marks=pytest.mark.slow)],
        ids=['scalar', 'vector']
    )
    def test_ishigami(self, ishigami_ref_indices, func):
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        res = sobol_indices(
            func=func, n=4096,
            dists=self.dists,
            random_state=rng
        )

        if func.__name__ == 'f_ishigami_vec':
            ishigami_ref_indices = [
                    [ishigami_ref_indices[0], ishigami_ref_indices[0]],
                    [ishigami_ref_indices[1], ishigami_ref_indices[1]]
            ]

        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)

        assert res._bootstrap_result is None
        bootstrap_res = res.bootstrap(n_resamples=99)
        assert isinstance(bootstrap_res, BootstrapSobolResult)
        assert isinstance(res._bootstrap_result, BootstrapResult)

        assert res._bootstrap_result.confidence_interval.low.shape[0] == 2
        assert res._bootstrap_result.confidence_interval.low[1].shape \
               == res.first_order.shape

        assert bootstrap_res.first_order.confidence_interval.low.shape \
               == res.first_order.shape
        assert bootstrap_res.total_order.confidence_interval.low.shape \
               == res.total_order.shape

        assert_array_less(
            bootstrap_res.first_order.confidence_interval.low, res.first_order
        )
        assert_array_less(
            res.first_order, bootstrap_res.first_order.confidence_interval.high
        )
        assert_array_less(
            bootstrap_res.total_order.confidence_interval.low, res.total_order
        )
        assert_array_less(
            res.total_order, bootstrap_res.total_order.confidence_interval.high
        )

        # call again to use previous results and change a param
        assert isinstance(
            res.bootstrap(confidence_level=0.9, n_resamples=99),
            BootstrapSobolResult
        )
        assert isinstance(res._bootstrap_result, BootstrapResult)

    def test_func_dict(self, ishigami_ref_indices):
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        n = 4096
        dists = [
            stats.uniform(loc=-np.pi, scale=2*np.pi),
            stats.uniform(loc=-np.pi, scale=2*np.pi),
            stats.uniform(loc=-np.pi, scale=2*np.pi)
        ]

        A, B = sample_A_B(n=n, dists=dists, random_state=rng)
        AB = sample_AB(A=A, B=B)

        func = {
            'f_A': f_ishigami(A).reshape(1, -1),
            'f_B': f_ishigami(B).reshape(1, -1),
            'f_AB': f_ishigami(AB).reshape((3, 1, -1))
        }

        res = sobol_indices(
            func=func, n=n,
            dists=dists,
            random_state=rng
        )
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)

        res = sobol_indices(
            func=func, n=n,
            random_state=rng
        )
        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)

    def test_method(self, ishigami_ref_indices):
        def jansen_sobol(f_A, f_B, f_AB):
            """Jansen for S and Sobol' for St.

            From Saltelli2010, table 2 formulations (c) and (e)."""
            var = np.var([f_A, f_B], axis=(0, -1))

            s = (var - 0.5*np.mean((f_B - f_AB)**2, axis=-1)) / var
            st = np.mean(f_A*(f_A - f_AB), axis=-1) / var

            return s.T, st.T

        rng = np.random.default_rng(28631265345463262246170309650372465332)
        res = sobol_indices(
            func=f_ishigami, n=4096,
            dists=self.dists,
            method=jansen_sobol,
            random_state=rng
        )

        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)

        def jansen_sobol_typed(
            f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            return jansen_sobol(f_A, f_B, f_AB)

        _ = sobol_indices(
            func=f_ishigami, n=8,
            dists=self.dists,
            method=jansen_sobol_typed,
            random_state=rng
        )

    def test_normalization(self, ishigami_ref_indices):
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        res = sobol_indices(
            func=lambda x: f_ishigami(x) + 1000, n=4096,
            dists=self.dists,
            random_state=rng
        )

        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-2)

    def test_constant_function(self, ishigami_ref_indices):

        def f_ishigami_vec_const(x):
            """Output of shape (3, n)."""
            res = f_ishigami(x)
            return res, res * 0 + 10, res

        rng = np.random.default_rng(28631265345463262246170309650372465332)
        res = sobol_indices(
            func=f_ishigami_vec_const, n=4096,
            dists=self.dists,
            random_state=rng
        )

        ishigami_vec_indices = [
                [ishigami_ref_indices[0], [0, 0, 0], ishigami_ref_indices[0]],
                [ishigami_ref_indices[1], [0, 0, 0], ishigami_ref_indices[1]]
        ]

        assert_allclose(res.first_order, ishigami_vec_indices[0], atol=1e-2)
        assert_allclose(res.total_order, ishigami_vec_indices[1], atol=1e-2)

    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    def test_more_converged(self, ishigami_ref_indices):
        rng = np.random.default_rng(28631265345463262246170309650372465332)
        res = sobol_indices(
            func=f_ishigami, n=2**19,  # 524288
            dists=self.dists,
            random_state=rng
        )

        assert_allclose(res.first_order, ishigami_ref_indices[0], atol=1e-4)
        assert_allclose(res.total_order, ishigami_ref_indices[1], atol=1e-4)

    def test_raises(self):

        message = r"Each distribution in `dists` must have method `ppf`"
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, dists="uniform")

        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, dists=[lambda x: x])

        message = r"The balance properties of Sobol'"
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=7, func=f_ishigami, dists=[stats.uniform()])

        with pytest.raises(ValueError, match=message):
            sobol_indices(n=4.1, func=f_ishigami, dists=[stats.uniform()])

        message = r"'toto' is not a valid 'method'"
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, method='toto')

        message = r"must have the following signature"
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami, method=lambda x: x)

        message = r"'dists' must be defined when 'func' is a callable"
        with pytest.raises(ValueError, match=message):
            sobol_indices(n=0, func=f_ishigami)

        def func_wrong_shape_output(x):
            return x.reshape(-1, 1)

        message = r"'func' output should have a shape"
        with pytest.raises(ValueError, match=message):
            sobol_indices(
                n=2, func=func_wrong_shape_output, dists=[stats.uniform()]
            )

        message = r"When 'func' is a dictionary"
        with pytest.raises(ValueError, match=message):
            sobol_indices(
                n=2, func={'f_A': [], 'f_AB': []}, dists=[stats.uniform()]
            )

        with pytest.raises(ValueError, match=message):
            # f_B malformed
            sobol_indices(
                n=2,
                func={'f_A': [1, 2], 'f_B': [3], 'f_AB': [5, 6, 7, 8]},
            )

        with pytest.raises(ValueError, match=message):
            # f_AB malformed
            sobol_indices(
                n=2,
                func={'f_A': [1, 2], 'f_B': [3, 4], 'f_AB': [5, 6, 7]},
            )
