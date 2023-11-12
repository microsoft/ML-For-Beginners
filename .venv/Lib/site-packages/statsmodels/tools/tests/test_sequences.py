import numpy as np
import numpy.testing as npt
from statsmodels.tools import sequences


def test_discrepancy():
    space_0 = [[0.1, 0.5], [0.2, 0.4], [0.3, 0.3], [0.4, 0.2], [0.5, 0.1]]
    space_1 = [[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]]
    space_2 = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]]

    corners = np.array([[0.5, 0.5], [6.5, 6.5]])

    npt.assert_allclose(sequences.discrepancy(space_0), 0.1353, atol=1e-4)

    # From Fang et al. Design and modeling for computer experiments, 2006
    npt.assert_allclose(sequences.discrepancy(space_1, corners), 0.0081, atol=1e-4)
    npt.assert_allclose(sequences.discrepancy(space_2, corners), 0.0105, atol=1e-4)


def test_van_der_corput():
    sample = sequences.van_der_corput(10)
    out = [0., 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625]
    npt.assert_almost_equal(sample, out)

    sample = sequences.van_der_corput(5, start_index=3)
    out = [0.75, 0.125, 0.625, 0.375, 0.875]
    npt.assert_almost_equal(sample, out)


def test_primes():
    primes = sequences.primes_from_2_to(50)
    out = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    npt.assert_allclose(primes, out)


def test_halton():
    corners = np.array([[0, 2], [10, 5]])
    sample = sequences.halton(dim=2, n_sample=5, bounds=corners)

    out = np.array([[5., 3.], [2.5, 4.], [7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)

    sample = sequences.halton(dim=2, n_sample=3, bounds=corners, start_index=2)
    out = np.array([[7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)
