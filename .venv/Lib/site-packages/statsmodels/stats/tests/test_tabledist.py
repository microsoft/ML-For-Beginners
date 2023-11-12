"""
Test data from Lilliefors test for normality
An Analytic Approximation to the Distribution of Lilliefors's
Test Statistic for Normality
Author(s): Gerard E. Dallal and Leland WilkinsonSource: The American
Statistician, Vol. 40, No. 4 (Nov., 1986), pp. 294-296
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from statsmodels.stats.tabledist import TableDist


def test_tabledist():
    # for this test alpha is sf probability, i.e. right tail probability
    alpha = np.array([0.2, 0.15, 0.1, 0.05, 0.01, 0.001])[::-1]
    size = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18, 19, 20, 25, 30, 40, 100, 400, 900], float)

    # critical values, rows are by sample size, columns are by alpha
    crit_lf = np.array([[303, 321, 346, 376, 413, 433],
                        [289, 303, 319, 343, 397, 439],
                        [269, 281, 297, 323, 371, 424],
                        [252, 264, 280, 304, 351, 402],
                        [239, 250, 265, 288, 333, 384],
                        [227, 238, 252, 274, 317, 365],
                        [217, 228, 241, 262, 304, 352],
                        [208, 218, 231, 251, 291, 338],
                        [200, 210, 222, 242, 281, 325],
                        [193, 202, 215, 234, 271, 314],
                        [187, 196, 208, 226, 262, 305],
                        [181, 190, 201, 219, 254, 296],
                        [176, 184, 195, 213, 247, 287],
                        [171, 179, 190, 207, 240, 279],
                        [167, 175, 185, 202, 234, 273],
                        [163, 170, 181, 197, 228, 266],
                        [159, 166, 176, 192, 223, 260],
                        [143, 150, 159, 173, 201, 236],
                        [131, 138, 146, 159, 185, 217],
                        [115, 120, 128, 139, 162, 189],
                        [74, 77, 82, 89, 104, 122],
                        [37, 39, 41, 45, 52, 61],
                        [25, 26, 28, 30, 35, 42]])[:, ::-1] / 1000.

    lf = TableDist(alpha, size, crit_lf)

    assert_almost_equal(lf.prob(0.166, 20), 0.15)
    assert_almost_equal(lf.crit(0.15, 20), 0.166)
    assert_almost_equal(lf.crit3(0.15, 20), 0.166)

    assert .159 <= lf.crit(0.17, 20) <= 166
    assert .159 <= lf.crit3(0.17, 20) <= .166

    assert .159 <= lf.crit(0.19, 20) <= .166
    assert .159 <= lf.crit3(0.19, 20) <= .166

    assert .159 <= lf.crit(0.199, 20) <= .166
    assert .159 <= lf.crit3(0.199, 20) <= .166

    # testing
    vals = [lf.prob(c, size[i]) for i in range(len(size)) for c in crit_lf[i]]
    vals = np.array(vals).reshape(-1, lf.n_alpha)
    delta = np.abs(vals) - lf.alpha
    assert_allclose(delta, np.zeros_like(delta))

    # 1.6653345369377348e-16
    vals = [lf.crit(c, size[i]) for i in range(len(size)) for c in lf.alpha]
    vals = np.array(vals).reshape(-1, lf.n_alpha)
    delta = np.abs(vals - crit_lf)
    assert_allclose(delta, np.zeros_like(delta))

    # 6.9388939039072284e-18)
    print(np.max(np.abs(np.array(
        [lf.crit3(c, size[i]) for i in range(len(size)) for c in
         lf.alpha]).reshape(-1, lf.n_alpha) - crit_lf)))
    # 4.0615705243496336e-12)
    vals = [lf.crit3(c, size[i]) for i in range(len(size))
            for c in lf.alpha[:-1] * 1.1]
    vals = np.array(vals).reshape(-1, lf.n_alpha - 1)
    assert (vals < crit_lf[:, :-1]).all()

    vals = [lf.crit3(c, size[i]) for i in range(len(size)) for c in
            lf.alpha[:-1] * 1.1]
    vals = np.array(vals).reshape(-1, lf.n_alpha - 1)
    assert (vals > crit_lf[:, 1:]).all()

    vals = [lf.prob(c * 0.9, size[i]) for i in range(len(size))
            for c in crit_lf[i, :-1]]
    vals = np.array(vals).reshape(-1, lf.n_alpha - 1)
    assert (vals > lf.alpha[:-1]).all()

    vals = [lf.prob(c * 1.1, size[i]) for i in range(len(size)) for c in
            crit_lf[i, 1:]]
    vals = np.array(vals).reshape(-1, lf.n_alpha - 1)
    assert (vals < lf.alpha[1:]).all()

    # start at size_idx=2 because of non-monotonicity of lf_crit
    vals = [lf.prob(c, size[i] * 0.9) for i in range(2, len(size))
            for c in crit_lf[i, :-1]]
    vals = np.array(vals).reshape(-1, lf.n_alpha - 1)
    assert (vals > lf.alpha[:-1]).all()
