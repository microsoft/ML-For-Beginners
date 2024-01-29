import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.optimize import nnls


class TestNNLS:
    def setup_method(self):
        self.rng = np.random.default_rng(1685225766635251)

    def test_nnls(self):
        a = np.arange(25.0).reshape(-1, 5)
        x = np.arange(5.0)
        y = a @ x
        x, res = nnls(a, y)
        assert res < 1e-7
        assert np.linalg.norm((a @ x) - y) < 1e-7

    def test_nnls_tall(self):
        a = self.rng.uniform(low=-10, high=10, size=[50, 10])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[10]))
        x[::2] = 0
        b = a @ x
        xact, rnorm = nnls(a, b, atol=500*np.linalg.norm(a, 1)*np.spacing(1.))
        assert_allclose(xact, x, rtol=0., atol=1e-10)
        assert rnorm < 1e-12

    def test_nnls_wide(self):
        # If too wide then problem becomes too ill-conditioned ans starts
        # emitting warnings, hence small m, n difference.
        a = self.rng.uniform(low=-10, high=10, size=[100, 120])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[120]))
        x[::2] = 0
        b = a @ x
        xact, rnorm = nnls(a, b, atol=500*np.linalg.norm(a, 1)*np.spacing(1.))
        assert_allclose(xact, x, rtol=0., atol=1e-10)
        assert rnorm < 1e-12

    def test_maxiter(self):
        # test that maxiter argument does stop iterations
        a = self.rng.uniform(size=(5, 10))
        b = self.rng.uniform(size=5)
        with assert_raises(RuntimeError):
            nnls(a, b, maxiter=1)
