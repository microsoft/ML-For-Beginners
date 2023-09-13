import os
from collections import Counter
from itertools import combinations, product

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal

from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
    van_der_corput, n_primes, primes_from_2_to,
    update_discrepancy, QMCEngine, _l1_norm,
    _perturb_discrepancy, _lloyd_centroidal_voronoi_tessellation
)  # noqa


class TestUtils:
    def test_scale(self):
        # 1d scalar
        space = [[0], [1], [0.5]]
        out = [[-2], [6], [2]]
        scaled_space = qmc.scale(space, l_bounds=-2, u_bounds=6)

        assert_allclose(scaled_space, out)

        # 2d space
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        bounds = np.array([[-2, 0], [6, 5]])
        out = [[-2, 0], [6, 5], [2, 2.5]]

        scaled_space = qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        assert_allclose(scaled_space, out)

        scaled_back_space = qmc.scale(scaled_space, l_bounds=bounds[0],
                                      u_bounds=bounds[1], reverse=True)
        assert_allclose(scaled_back_space, space)

        # broadcast
        space = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]
        l_bounds, u_bounds = 0, [6, 5, 3]
        out = [[0, 0, 0], [6, 5, 3], [3, 2.5, 1.5]]

        scaled_space = qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        assert_allclose(scaled_space, out)

    def test_scale_random(self):
        rng = np.random.default_rng(317589836511269190194010915937762468165)
        sample = rng.random((30, 10))
        a = -rng.random(10) * 10
        b = rng.random(10) * 10
        scaled = qmc.scale(sample, a, b, reverse=False)
        unscaled = qmc.scale(scaled, a, b, reverse=True)
        assert_allclose(unscaled, sample)

    def test_scale_errors(self):
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            space = [0, 1, 0.5]
            qmc.scale(space, l_bounds=-2, u_bounds=6)

        with pytest.raises(ValueError, match=r"Bounds are not consistent"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 6], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"'l_bounds' and 'u_bounds'"
                                             r" must be broadcastable"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            l_bounds, u_bounds = [-2, 0, 2], [6, 5]
            qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        with pytest.raises(ValueError, match=r"'l_bounds' and 'u_bounds'"
                                             r" must be broadcastable"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 0, 2], [6, 5, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"Sample is not in unit "
                                             r"hypercube"):
            space = [[0, 0], [1, 1.5], [0.5, 0.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"Sample is out of bounds"):
            out = [[-2, 0], [6, 5], [8, 2.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(out, l_bounds=bounds[0], u_bounds=bounds[1],
                      reverse=True)

    def test_discrepancy(self):
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
        space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])
        space_2 = (2.0 * space_2 - 1.0) / (2.0 * 6.0)

        # From Fang et al. Design and modeling for computer experiments, 2006
        assert_allclose(qmc.discrepancy(space_1), 0.0081, atol=1e-4)
        assert_allclose(qmc.discrepancy(space_2), 0.0105, atol=1e-4)

        # From Zhou Y.-D. et al. Mixture discrepancy for quasi-random point
        # sets. Journal of Complexity, 29 (3-4), pp. 283-301, 2013.
        # Example 4 on Page 298
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        assert_allclose(qmc.discrepancy(sample, method='MD'), 2.5000,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD'), 1.3680,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD'), 0.3172,
                        atol=1e-4)

        # From Tim P. et al. Minimizing the L2 and Linf star discrepancies
        # of a single point in the unit hypercube. JCAM, 2005
        # Table 1 on Page 283
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3**(-dim))
            assert_allclose(qmc.discrepancy(np.array([[1]*dim]),
                                            method='L2-star'), ref)

    def test_discrepancy_errors(self):
        sample = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])

        with pytest.raises(
            ValueError, match=r"Sample is not in unit hypercube"
        ):
            qmc.discrepancy(sample)

        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            qmc.discrepancy([1, 3])

        sample = [[0, 0], [1, 1], [0.5, 0.5]]
        with pytest.raises(ValueError, match=r"'toto' is not a valid ..."):
            qmc.discrepancy(sample, method="toto")

    def test_discrepancy_parallel(self, monkeypatch):
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        assert_allclose(qmc.discrepancy(sample, method='MD', workers=8),
                        2.5000,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD', workers=8),
                        1.3680,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD', workers=8),
                        0.3172,
                        atol=1e-4)

        # From Tim P. et al. Minimizing the L2 and Linf star discrepancies
        # of a single point in the unit hypercube. JCAM, 2005
        # Table 1 on Page 283
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3 ** (-dim))
            assert_allclose(qmc.discrepancy(np.array([[1] * dim]),
                                            method='L2-star', workers=-1), ref)

        monkeypatch.setattr(os, 'cpu_count', lambda: None)
        with pytest.raises(NotImplementedError, match="Cannot determine the"):
            qmc.discrepancy(sample, workers=-1)

        with pytest.raises(ValueError, match="Invalid number of workers..."):
            qmc.discrepancy(sample, workers=-2)

    def test_update_discrepancy(self):
        # From Fang et al. Design and modeling for computer experiments, 2006
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)

        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        assert_allclose(disc_iter, 0.0081, atol=1e-4)

        # n<d
        rng = np.random.default_rng(241557431858162136881731220526394276199)
        space_1 = rng.random((4, 10))

        disc_ref = qmc.discrepancy(space_1)
        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        assert_allclose(disc_iter, disc_ref, atol=1e-4)

        # errors
        with pytest.raises(ValueError, match=r"Sample is not in unit "
                                             r"hypercube"):
            update_discrepancy(space_1[-1], space_1[:-1] + 1, disc_init)

        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            update_discrepancy(space_1[-1], space_1[0], disc_init)

        x_new = [1, 3]
        with pytest.raises(ValueError, match=r"x_new is not in unit "
                                             r"hypercube"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        x_new = [[0.5, 0.5]]
        with pytest.raises(ValueError, match=r"x_new is not a 1D array"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        x_new = [0.3, 0.1, 0]
        with pytest.raises(ValueError, match=r"x_new and sample must be "
                                             r"broadcastable"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

    def test_perm_discrepancy(self):
        rng = np.random.default_rng(46449423132557934943847369749645759997)
        qmc_gen = qmc.LatinHypercube(5, seed=rng)
        sample = qmc_gen.random(10)
        disc = qmc.discrepancy(sample)

        for i in range(100):
            row_1 = rng.integers(10)
            row_2 = rng.integers(10)
            col = rng.integers(5)

            disc = _perturb_discrepancy(sample, row_1, row_2, col, disc)
            sample[row_1, col], sample[row_2, col] = (
                sample[row_2, col], sample[row_1, col])
            disc_reference = qmc.discrepancy(sample)
            assert_allclose(disc, disc_reference)

    def test_discrepancy_alternative_implementation(self):
        """Alternative definitions from Matt Haberland."""

        def disc_c2(x):
            n, s = x.shape
            xij = x
            disc1 = np.sum(np.prod((1
                                    + 1/2*np.abs(xij-0.5)
                                    - 1/2*np.abs(xij-0.5)**2), axis=1))
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc2 = np.sum(np.sum(np.prod(1
                                          + 1/2*np.abs(xij - 0.5)
                                          + 1/2*np.abs(xkj - 0.5)
                                          - 1/2*np.abs(xij - xkj), axis=2),
                                  axis=0))
            return (13/12)**s - 2/n * disc1 + 1/n**2*disc2

        def disc_wd(x):
            n, s = x.shape
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc = np.sum(np.sum(np.prod(3/2
                                         - np.abs(xij - xkj)
                                         + np.abs(xij - xkj)**2, axis=2),
                                 axis=0))
            return -(4/3)**s + 1/n**2 * disc

        def disc_md(x):
            n, s = x.shape
            xij = x
            disc1 = np.sum(np.prod((5/3
                                    - 1/4*np.abs(xij-0.5)
                                    - 1/4*np.abs(xij-0.5)**2), axis=1))
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc2 = np.sum(np.sum(np.prod(15/8
                                          - 1/4*np.abs(xij - 0.5)
                                          - 1/4*np.abs(xkj - 0.5)
                                          - 3/4*np.abs(xij - xkj)
                                          + 1/2*np.abs(xij - xkj)**2,
                                          axis=2), axis=0))
            return (19/12)**s - 2/n * disc1 + 1/n**2*disc2

        def disc_star_l2(x):
            n, s = x.shape
            return np.sqrt(
                3 ** (-s) - 2 ** (1 - s) / n
                * np.sum(np.prod(1 - x ** 2, axis=1))
                + np.sum([
                    np.prod(1 - np.maximum(x[k, :], x[j, :]))
                    for k in range(n) for j in range(n)
                ]) / n ** 2
            )

        rng = np.random.default_rng(117065081482921065782761407107747179201)
        sample = rng.random((30, 10))

        disc_curr = qmc.discrepancy(sample, method='CD')
        disc_alt = disc_c2(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='WD')
        disc_alt = disc_wd(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='MD')
        disc_alt = disc_md(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='L2-star')
        disc_alt = disc_star_l2(sample)
        assert_allclose(disc_curr, disc_alt)

    def test_n_primes(self):
        primes = n_primes(10)
        assert primes[-1] == 29

        primes = n_primes(168)
        assert primes[-1] == 997

        primes = n_primes(350)
        assert primes[-1] == 2357

    def test_primes(self):
        primes = primes_from_2_to(50)
        out = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        assert_allclose(primes, out)


class TestVDC:
    def test_van_der_corput(self):
        sample = van_der_corput(10)
        out = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625,
               0.375, 0.875, 0.0625, 0.5625]
        assert_allclose(sample, out)

        sample = van_der_corput(10, workers=4)
        assert_allclose(sample, out)

        sample = van_der_corput(10, workers=8)
        assert_allclose(sample, out)

        sample = van_der_corput(7, start_index=3)
        assert_allclose(sample, out[3:])

    def test_van_der_corput_scramble(self):
        seed = 338213789010180879520345496831675783177
        out = van_der_corput(10, scramble=True, seed=seed)

        sample = van_der_corput(7, start_index=3, scramble=True, seed=seed)
        assert_allclose(sample, out[3:])

        sample = van_der_corput(
            7, start_index=3, scramble=True, seed=seed, workers=4
        )
        assert_allclose(sample, out[3:])

        sample = van_der_corput(
            7, start_index=3, scramble=True, seed=seed, workers=8
        )
        assert_allclose(sample, out[3:])

    def test_invalid_base_error(self):
        with pytest.raises(ValueError, match=r"'base' must be at least 2"):
            van_der_corput(10, base=1)


class RandomEngine(qmc.QMCEngine):
    def __init__(self, d, optimization=None, seed=None):
        super().__init__(d=d, optimization=optimization, seed=seed)

    def _random(self, n=1, *, workers=1):
        sample = self.rng.random((n, self.d))
        return sample


def test_subclassing_QMCEngine():
    engine = RandomEngine(2, seed=175180605424926556207367152557812293274)

    sample_1 = engine.random(n=5)
    sample_2 = engine.random(n=7)
    assert engine.num_generated == 12

    # reset and re-sample
    engine.reset()
    assert engine.num_generated == 0

    sample_1_test = engine.random(n=5)
    assert_equal(sample_1, sample_1_test)

    # repeat reset and fast forward
    engine.reset()
    engine.fast_forward(n=5)
    sample_2_test = engine.random(n=7)

    assert_equal(sample_2, sample_2_test)
    assert engine.num_generated == 12


def test_raises():
    # input validation
    with pytest.raises(ValueError, match=r"d must be a non-negative integer"):
        RandomEngine((2,))  # noqa

    with pytest.raises(ValueError, match=r"d must be a non-negative integer"):
        RandomEngine(-1)  # noqa

    msg = r"'u_bounds' and 'l_bounds' must be integers"
    with pytest.raises(ValueError, match=msg):
        engine = RandomEngine(1)
        engine.integers(l_bounds=1, u_bounds=1.1)


def test_integers():
    engine = RandomEngine(1, seed=231195739755290648063853336582377368684)

    # basic tests
    sample = engine.integers(1, n=10)
    assert_equal(np.unique(sample), [0])

    assert sample.dtype == np.dtype('int64')

    sample = engine.integers(1, n=10, endpoint=True)
    assert_equal(np.unique(sample), [0, 1])

    low = -5
    high = 7

    # scaling logic
    engine.reset()
    ref_sample = engine.random(20)
    ref_sample = ref_sample * (high - low) + low
    ref_sample = np.floor(ref_sample).astype(np.int64)

    engine.reset()
    sample = engine.integers(low, u_bounds=high, n=20, endpoint=False)

    assert_equal(sample, ref_sample)

    # up to bounds, no less, no more
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=False)
    assert_equal((sample.min(), sample.max()), (low, high-1))

    sample = engine.integers(low, u_bounds=high, n=100, endpoint=True)
    assert_equal((sample.min(), sample.max()), (low, high))


def test_integers_nd():
    d = 10
    rng = np.random.default_rng(3716505122102428560615700415287450951)
    low = rng.integers(low=-5, high=-1, size=d)
    high = rng.integers(low=1, high=5, size=d, endpoint=True)
    engine = RandomEngine(d, seed=rng)

    sample = engine.integers(low, u_bounds=high, n=100, endpoint=False)
    assert_equal(sample.min(axis=0), low)
    assert_equal(sample.max(axis=0), high-1)

    sample = engine.integers(low, u_bounds=high, n=100, endpoint=True)
    assert_equal(sample.min(axis=0), low)
    assert_equal(sample.max(axis=0), high)


class QMCEngineTests:
    """Generic tests for QMC engines."""
    qmce = NotImplemented
    can_scramble = NotImplemented
    unscramble_nd = NotImplemented
    scramble_nd = NotImplemented

    scramble = [True, False]
    ids = ["Scrambled", "Unscrambled"]

    def engine(
        self, scramble: bool,
        seed=170382760648021597650530316304495310428,
        **kwargs
    ) -> QMCEngine:
        if self.can_scramble:
            return self.qmce(scramble=scramble, seed=seed, **kwargs)
        else:
            if scramble:
                pytest.skip()
            else:
                return self.qmce(seed=seed, **kwargs)

    def reference(self, scramble: bool) -> np.ndarray:
        return self.scramble_nd if scramble else self.unscramble_nd

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_0dim(self, scramble):
        engine = self.engine(d=0, scramble=scramble)
        sample = engine.random(4)
        assert_array_equal(np.empty((4, 0)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_0sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(0)
        assert_array_equal(np.empty((0, 2)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_1sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(1)
        assert (1, 2) == sample.shape

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_bounds(self, scramble):
        engine = self.engine(d=100, scramble=scramble)
        sample = engine.random(512)
        assert np.all(sample >= 0)
        assert np.all(sample <= 1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_sample(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(n=len(ref_sample))

        assert_allclose(sample, ref_sample, atol=1e-1)
        assert engine.num_generated == len(ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_continuing(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        engine = self.engine(d=2, scramble=scramble)

        n_half = len(ref_sample) // 2

        _ = engine.random(n=n_half)
        sample = engine.random(n=n_half)
        assert_allclose(sample, ref_sample[n_half:], atol=1e-1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    @pytest.mark.parametrize(
        "seed",
        (
            170382760648021597650530316304495310428,
            np.random.default_rng(170382760648021597650530316304495310428),
            None,
        ),
    )
    def test_reset(self, scramble, seed):
        engine = self.engine(d=2, scramble=scramble, seed=seed)
        ref_sample = engine.random(n=8)

        engine.reset()
        assert engine.num_generated == 0

        sample = engine.random(n=8)
        assert_allclose(sample, ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_fast_forward(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        engine = self.engine(d=2, scramble=scramble)

        engine.fast_forward(4)
        sample = engine.random(n=4)

        assert_allclose(sample, ref_sample[4:], atol=1e-1)

        # alternate fast forwarding with sampling
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.random())
            else:
                engine.fast_forward(1)
        assert_allclose(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            np.concatenate(even_draws),
            atol=1e-5
        )

    @pytest.mark.parametrize("scramble", [True])
    def test_distribution(self, scramble):
        d = 50
        engine = self.engine(d=d, scramble=scramble)
        sample = engine.random(1024)
        assert_allclose(
            np.mean(sample, axis=0), np.repeat(0.5, d), atol=1e-2
        )
        assert_allclose(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=1e-2
        )
        assert_allclose(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=1e-2
        )

    def test_raises_optimizer(self):
        message = r"'toto' is not a valid optimization method"
        with pytest.raises(ValueError, match=message):
            self.engine(d=1, scramble=False, optimization="toto")

    @pytest.mark.parametrize(
        "optimization,metric",
        [
            ("random-CD", qmc.discrepancy),
            ("lloyd", lambda sample: -_l1_norm(sample))]
    )
    def test_optimizers(self, optimization, metric):
        engine = self.engine(d=2, scramble=False)
        sample_ref = engine.random(n=64)
        metric_ref = metric(sample_ref)

        optimal_ = self.engine(d=2, scramble=False, optimization=optimization)
        sample_ = optimal_.random(n=64)
        metric_ = metric(sample_)

        assert metric_ < metric_ref

    def test_consume_prng_state(self):
        rng = np.random.default_rng(0xa29cabb11cfdf44ff6cac8bec254c2a0)
        sample = []
        for i in range(3):
            engine = self.engine(d=2, scramble=True, seed=rng)
            sample.append(engine.random(4))

        with pytest.raises(AssertionError, match="Arrays are not equal"):
            assert_equal(sample[0], sample[1])
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            assert_equal(sample[0], sample[2])


class TestHalton(QMCEngineTests):
    qmce = qmc.Halton
    can_scramble = True
    # theoretical values known from Van der Corput
    unscramble_nd = np.array([[0, 0], [1 / 2, 1 / 3],
                              [1 / 4, 2 / 3], [3 / 4, 1 / 9],
                              [1 / 8, 4 / 9], [5 / 8, 7 / 9],
                              [3 / 8, 2 / 9], [7 / 8, 5 / 9]])
    # theoretical values unknown: convergence properties checked
    scramble_nd = np.array([[0.50246036, 0.93382481],
                            [0.00246036, 0.26715815],
                            [0.75246036, 0.60049148],
                            [0.25246036, 0.8227137 ],
                            [0.62746036, 0.15604704],
                            [0.12746036, 0.48938037],
                            [0.87746036, 0.71160259],
                            [0.37746036, 0.04493592]])

    def test_workers(self):
        ref_sample = self.reference(scramble=True)
        engine = self.engine(d=2, scramble=True)
        sample = engine.random(n=len(ref_sample), workers=8)

        assert_allclose(sample, ref_sample, atol=1e-3)

        # worker + integers
        engine.reset()
        ref_sample = engine.integers(10)
        engine.reset()
        sample = engine.integers(10, workers=8)
        assert_equal(sample, ref_sample)


class TestLHS(QMCEngineTests):
    qmce = qmc.LatinHypercube
    can_scramble = True

    def test_continuing(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    def test_fast_forward(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    def test_sample(self, *args):
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    @pytest.mark.parametrize("strength", [1, 2])
    @pytest.mark.parametrize("scramble", [False, True])
    @pytest.mark.parametrize("optimization", [None, "random-CD"])
    def test_sample_stratified(self, optimization, scramble, strength):
        seed = np.random.default_rng(37511836202578819870665127532742111260)
        p = 5
        n = p**2
        d = 6

        engine = qmc.LatinHypercube(d=d, scramble=scramble,
                                    strength=strength,
                                    optimization=optimization,
                                    seed=seed)
        sample = engine.random(n=n)
        assert sample.shape == (n, d)
        assert engine.num_generated == n

        # centering stratifies samples in the middle of equal segments:
        # * inter-sample distance is constant in 1D sub-projections
        # * after ordering, columns are equal
        expected1d = (np.arange(n) + 0.5) / n
        expected = np.broadcast_to(expected1d, (d, n)).T
        assert np.any(sample != expected)

        sorted_sample = np.sort(sample, axis=0)
        tol = 0.5 / n if scramble else 0

        assert_allclose(sorted_sample, expected, atol=tol)
        assert np.any(sample - expected > tol)

        if strength == 2 and optimization is None:
            unique_elements = np.arange(p)
            desired = set(product(unique_elements, unique_elements))

            for i, j in combinations(range(engine.d), 2):
                samples_2d = sample[:, [i, j]]
                res = (samples_2d * p).astype(int)
                res_set = {tuple(row) for row in res}
                assert_equal(res_set, desired)

    def test_optimizer_1d(self):
        # discrepancy measures are invariant under permuting factors and runs
        engine = self.engine(d=1, scramble=False)
        sample_ref = engine.random(n=64)

        optimal_ = self.engine(d=1, scramble=False, optimization="random-CD")
        sample_ = optimal_.random(n=64)

        assert_array_equal(sample_ref, sample_)

    def test_raises(self):
        message = r"not a valid strength"
        with pytest.raises(ValueError, match=message):
            qmc.LatinHypercube(1, strength=3)

        message = r"n is not the square of a prime number"
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(16)

        message = r"n is not the square of a prime number"
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(5)  # because int(sqrt(5)) would result in 2

        message = r"n is too small for d"
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=5, strength=2)
            engine.random(9)

        message = r"'centered' is deprecated"
        with pytest.warns(UserWarning, match=message):
            qmc.LatinHypercube(1, centered=True)


class TestSobol(QMCEngineTests):
    qmce = qmc.Sobol
    can_scramble = True
    # theoretical values from Joe Kuo2010
    unscramble_nd = np.array([[0., 0.],
                              [0.5, 0.5],
                              [0.75, 0.25],
                              [0.25, 0.75],
                              [0.375, 0.375],
                              [0.875, 0.875],
                              [0.625, 0.125],
                              [0.125, 0.625]])

    # theoretical values unknown: convergence properties checked
    scramble_nd = np.array([[0.25331921, 0.41371179],
                            [0.8654213, 0.9821167],
                            [0.70097554, 0.03664616],
                            [0.18027647, 0.60895735],
                            [0.10521339, 0.21897069],
                            [0.53019685, 0.66619033],
                            [0.91122276, 0.34580743],
                            [0.45337471, 0.78912079]])

    def test_warning(self):
        with pytest.warns(UserWarning, match=r"The balance properties of "
                                             r"Sobol' points"):
            engine = qmc.Sobol(1)
            engine.random(10)

    def test_random_base2(self):
        engine = qmc.Sobol(2, scramble=False)
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[:4], sample)

        # resampling still having N=2**n
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[4:8], sample)

        # resampling again but leading to N!=2**n
        with pytest.raises(ValueError, match=r"The balance properties of "
                                             r"Sobol' points"):
            engine.random_base2(2)

    def test_raise(self):
        with pytest.raises(ValueError, match=r"Maximum supported "
                                             r"dimensionality"):
            qmc.Sobol(qmc.Sobol.MAXDIM + 1)

        with pytest.raises(ValueError, match=r"Maximum supported "
                                             r"'bits' is 64"):
            qmc.Sobol(1, bits=65)

    def test_high_dim(self):
        engine = qmc.Sobol(1111, scramble=False)
        count1 = Counter(engine.random().flatten().tolist())
        count2 = Counter(engine.random().flatten().tolist())
        assert_equal(count1, Counter({0.0: 1111}))
        assert_equal(count2, Counter({0.5: 1111}))

    @pytest.mark.parametrize("bits", [2, 3])
    def test_bits(self, bits):
        engine = qmc.Sobol(2, scramble=False, bits=bits)
        ns = 2**bits
        sample = engine.random(ns)
        assert_array_equal(self.unscramble_nd[:ns], sample)

        with pytest.raises(ValueError, match="increasing `bits`"):
            engine.random()

    def test_64bits(self):
        engine = qmc.Sobol(2, scramble=False, bits=64)
        sample = engine.random(8)
        assert_array_equal(self.unscramble_nd, sample)


class TestPoisson(QMCEngineTests):
    qmce = qmc.PoissonDisk
    can_scramble = False

    def test_bounds(self, *args):
        pytest.skip("Too costly in memory.")

    def test_fast_forward(self, *args):
        pytest.skip("Not applicable: recursive process.")

    def test_sample(self, *args):
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    def test_continuing(self, *args):
        # can continue a sampling, but will not preserve the same order
        # because candidates are lost, so we will not select the same center
        radius = 0.05
        ns = 6
        engine = self.engine(d=2, radius=radius, scramble=False)

        sample_init = engine.random(n=ns)
        assert len(sample_init) <= ns
        assert l2_norm(sample_init) >= radius

        sample_continued = engine.random(n=ns)
        assert len(sample_continued) <= ns
        assert l2_norm(sample_continued) >= radius

        sample = np.concatenate([sample_init, sample_continued], axis=0)
        assert len(sample) <= ns * 2
        assert l2_norm(sample) >= radius

    def test_mindist(self):
        rng = np.random.default_rng(132074951149370773672162394161442690287)
        ns = 50

        low, high = 0.08, 0.2
        radii = (high - low) * rng.random(5) + low

        dimensions = [1, 3, 4]
        hypersphere_methods = ["volume", "surface"]

        gen = product(dimensions, radii, hypersphere_methods)

        for d, radius, hypersphere in gen:
            engine = self.qmce(
                d=d, radius=radius, hypersphere=hypersphere, seed=rng
            )
            sample = engine.random(ns)

            assert len(sample) <= ns
            assert l2_norm(sample) >= radius

    def test_fill_space(self):
        radius = 0.2
        engine = self.qmce(d=2, radius=radius)

        sample = engine.fill_space()
        # circle packing problem is np complex
        assert l2_norm(sample) >= radius

    def test_raises(self):
        message = r"'toto' is not a valid hypersphere sampling"
        with pytest.raises(ValueError, match=message):
            qmc.PoissonDisk(1, hypersphere="toto")


class TestMultinomialQMC:
    def test_validations(self):
        # negative Ps
        p = np.array([0.12, 0.26, -0.05, 0.35, 0.22])
        with pytest.raises(ValueError, match=r"Elements of pvals must "
                                             r"be non-negative."):
            qmc.MultinomialQMC(p, n_trials=10)

        # sum of P too large
        p = np.array([0.12, 0.26, 0.1, 0.35, 0.22])
        message = r"Elements of pvals must sum to 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10)

        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])

        message = r"Dimension of `engine` must be 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=qmc.Sobol(d=2))

        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=np.random.default_rng())

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_MultinomialBasicDraw(self):
        seed = np.random.default_rng(6955663962957011631562466584467607969)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)

    def test_MultinomialDistribution(self):
        seed = np.random.default_rng(77797854505813727292048130876699859000)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        engine = qmc.MultinomialQMC(p, n_trials=8192, seed=seed)
        draws = engine.random(1)
        assert_allclose(draws / np.sum(draws), np.atleast_2d(p), atol=1e-4)

    def test_FindIndex(self):
        p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
        size = len(p_cumulative)
        assert_equal(_test_find_index(p_cumulative, size, 0.0), 0)
        assert_equal(_test_find_index(p_cumulative, size, 0.4), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.44999), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.45001), 3)
        assert_equal(_test_find_index(p_cumulative, size, 1.0), size - 1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_other_engine(self):
        # same as test_MultinomialBasicDraw with different engine
        seed = np.random.default_rng(283753519042773243071753037669078065412)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        base_engine = qmc.Sobol(1, scramble=True, seed=seed)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, engine=base_engine,
                                    seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)


class TestNormalQMC:
    def test_NormalQMC(self):
        # d = 1
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(1))
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        # d = 2
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2))
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCInvTransform(self):
        # d = 1
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(1), inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        # d = 2
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCSeeded(self):
        # test even dimension
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923],
                                     [-1.477655, 0.846851]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # test odd dimension
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578],
                                     [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # same test with another engine
        seed = np.random.default_rng(274600237797326520096085022671371676017)
        base_engine = qmc.Sobol(4, scramble=True, seed=seed)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), inv_transform=False,
            engine=base_engine, seed=seed
        )
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.932001, -0.522923, 0.036578],
                                     [-1.778011, 0.912428, -0.065421]])
        assert_allclose(samples, samples_expected, atol=1e-4)

    def test_NormalQMCSeededInvTransform(self):
        # test even dimension
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.913237, -0.964026],
                                     [0.255904, 0.003068]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # test odd dimension
        seed = np.random.default_rng(288527772707286126646493545351112463929)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.913237, -0.964026, 0.355501],
                                     [0.699261, 2.90213 , -0.6418]])
        assert_allclose(samples, samples_expected, atol=1e-4)

    def test_other_engine(self):
        for d in (0, 1, 2):
            base_engine = qmc.Sobol(d=d, scramble=False)
            engine = qmc.MultivariateNormalQMC(mean=np.zeros(d),
                                               engine=base_engine,
                                               inv_transform=True)
            samples = engine.random()
            assert_equal(samples.shape, (1, d))

    def test_NormalQMCShapiro(self):
        rng = np.random.default_rng(13242)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=rng)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

    def test_NormalQMCShapiroInvTransform(self):
        rng = np.random.default_rng(32344554)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=True, seed=rng)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2


class TestMultivariateNormalQMC:

    def test_validations(self):

        message = r"Dimension of `engine` must be consistent"
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], engine=qmc.Sobol(d=2))

        message = r"Dimension of `engine` must be consistent"
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0, 0], engine=qmc.Sobol(d=4))

        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], engine=np.random.default_rng())

        message = r"Covariance matrix not PSD."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 2], [2, 1]])

        message = r"Covariance matrix is not symmetric."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 0], [2, 1]])

        message = r"Dimension mismatch between mean and covariance."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], [[1, 0], [0, 1]])

    def test_MultivariateNormalQMCNonPD(self):
        # try with non-pd but psd cov; should work
        engine = qmc.MultivariateNormalQMC(
            [0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]],
        )
        assert engine._corr_matrix is not None

    def test_MultivariateNormalQMC(self):
        # d = 1 scalar
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # d = 2 list
        engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]])
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCInvTransform(self):
        # d = 1 scalar
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5, inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # d = 2 list
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True,
        )
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov, inv_transform=True)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCSeeded(self):
        # test even dimension
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((2, 2))
        A = a @ a.transpose() + np.diag(rng.random(2))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A,
                                           inv_transform=False, seed=rng)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.64419, -0.882413],
                                     [0.837199, 2.045301]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # test odd dimension
        rng = np.random.default_rng(180182791534511062935571481899241825000)
        a = rng.standard_normal((3, 3))
        A = a @ a.transpose() + np.diag(rng.random(3))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A,
                                           inv_transform=False, seed=rng)
        samples = engine.random(n=2)
        samples_expected = np.array([[-0.693853, -1.265338, -0.088024],
                                     [1.620193, 2.679222, 0.457343]])
        assert_allclose(samples, samples_expected, atol=1e-4)

    def test_MultivariateNormalQMCSeededInvTransform(self):
        # test even dimension
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        a = rng.standard_normal((2, 2))
        A = a @ a.transpose() + np.diag(rng.random(2))
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0]), A, seed=rng, inv_transform=True
        )
        samples = engine.random(n=2)
        samples_expected = np.array([[0.682171, -3.114233],
                                     [-0.098463, 0.668069]])
        assert_allclose(samples, samples_expected, atol=1e-4)

        # test odd dimension
        rng = np.random.default_rng(224125808928297329711992996940871155974)
        a = rng.standard_normal((3, 3))
        A = a @ a.transpose() + np.diag(rng.random(3))
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0, 0]), A, seed=rng, inv_transform=True
        )
        samples = engine.random(n=2)
        samples_expected = np.array([[0.988061, -1.644089, -0.877035],
                                     [-1.771731, 1.096988, 2.024744]])
        assert_allclose(samples, samples_expected, atol=1e-4)

    def test_MultivariateNormalQMCShapiro(self):
        # test the standard case
        seed = np.random.default_rng(188960007281846377164494575845971640)
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

        # test the correlated, non-zero mean case
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # check covariance
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 1e-2

    def test_MultivariateNormalQMCShapiroInvTransform(self):
        # test the standard case
        seed = np.random.default_rng(200089821034563288698994840831440331329)
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed, inv_transform=True
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

        # test the correlated, non-zero mean case
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0],
            cov=[[1.5, 0.5], [0.5, 1.5]],
            seed=seed,
            inv_transform=True,
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # check covariance
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 1e-2

    def test_MultivariateNormalQMCDegenerate(self):
        # X, Y iid standard Normal and Z = X + Y, random vector (X, Y, Z)
        seed = np.random.default_rng(16320637417581448357869821654290448620)
        engine = qmc.MultivariateNormalQMC(
            mean=[0.0, 0.0, 0.0],
            cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
            seed=seed,
        )
        samples = engine.random(n=512)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert np.abs(np.std(samples[:, 0]) - 1) < 1e-2
        assert np.abs(np.std(samples[:, 1]) - 1) < 1e-2
        assert np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 1e-2
        for i in (0, 1, 2):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.8
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2
        assert np.abs(cov[0, 2] - 1) < 1e-2
        # check to see if X + Y = Z almost exactly
        assert all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2])
                   < 1e-5)


class TestLloyd:
    def test_lloyd(self):
        # quite sensible seed as it can go up before going further down
        rng = np.random.RandomState(1809831)
        sample = rng.uniform(0, 1, size=(128, 2))
        base_l1 = _l1_norm(sample)
        base_l2 = l2_norm(sample)

        for _ in range(4):
            sample_lloyd = _lloyd_centroidal_voronoi_tessellation(
                    sample, maxiter=1,
            )
            curr_l1 = _l1_norm(sample_lloyd)
            curr_l2 = l2_norm(sample_lloyd)

            # higher is better for the distance measures
            assert base_l1 < curr_l1
            assert base_l2 < curr_l2

            base_l1 = curr_l1
            base_l2 = curr_l2

            sample = sample_lloyd

    def test_lloyd_non_mutating(self):
        """
        Verify that the input samples are not mutated in place and that they do
        not share memory with the output.
        """
        sample_orig = np.array([[0.1, 0.1],
                                [0.1, 0.2],
                                [0.2, 0.1],
                                [0.2, 0.2]])
        sample_copy = sample_orig.copy()
        new_sample = _lloyd_centroidal_voronoi_tessellation(
            sample=sample_orig
        )
        assert_allclose(sample_orig, sample_copy)
        assert not np.may_share_memory(sample_orig, new_sample)

    def test_lloyd_errors(self):
        with pytest.raises(ValueError, match=r"`sample` is not a 2D array"):
            sample = [0, 1, 0.5]
            _lloyd_centroidal_voronoi_tessellation(sample)

        msg = r"`sample` dimension is not >= 2"
        with pytest.raises(ValueError, match=msg):
            sample = [[0], [0.4], [1]]
            _lloyd_centroidal_voronoi_tessellation(sample)

        msg = r"`sample` is not in unit hypercube"
        with pytest.raises(ValueError, match=msg):
            sample = [[-1.1, 0], [0.1, 0.4], [1, 2]]
            _lloyd_centroidal_voronoi_tessellation(sample)


# mindist
def l2_norm(sample):
    return distance.pdist(sample).min()
