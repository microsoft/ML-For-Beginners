import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.contrast import Contrast
import statsmodels.stats.contrast as smc


class TestContrast:
    @classmethod
    def setup_class(cls):
        numpy.random.seed(54321)
        cls.X = numpy.random.standard_normal((40,10))

    def test_contrast1(self):
        term = np.column_stack((self.X[:,0], self.X[:,2]))
        c = Contrast(term, self.X)
        test_contrast = [[1] + [0]*9, [0]*2 + [1] + [0]*7]
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast2(self):
        zero = np.zeros((40,))
        term = np.column_stack((zero, self.X[:,2]))
        c = Contrast(term, self.X)
        test_contrast = [0]*2 + [1] + [0]*7
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast3(self):
        P = np.dot(self.X, np.linalg.pinv(self.X))
        resid = np.identity(40) - P
        noise = np.dot(resid, numpy.random.standard_normal((40, 5)))
        term = np.column_stack((noise, self.X[:,2]))
        c = Contrast(term, self.X)
        assert_equal(c.contrast_matrix.shape, (10,))
#TODO: this should actually test the value of the contrast, not only its dimension

    def test_estimable(self):
        X2 = np.column_stack((self.X, self.X[:,5]))
        c = Contrast(self.X[:,5],X2)
        #TODO: I do not think this should be estimable?  isestimable correct?


def test_constraints():
    cm_ = np.eye(4, 3, k=-1)
    cpairs = np.array([[ 1.,  0.,  0.],
                       [ 0.,  1.,  0.],
                       [ 0.,  0.,  1.],
                       [-1.,  1.,  0.],
                       [-1.,  0.,  1.],
                       [ 0., -1.,  1.]])
    c0 = smc._constraints_factor(cm_)
    assert_equal(c0, cpairs)

    c1 = smc._contrast_pairs(3, 4, 0)
    assert_equal(c1, cpairs)

    # embedded
    cpairs2 = np.array([[ 0.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  1.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  1.,  0.,  0.],
                        [ 0., -1.,  1.,  0.,  0.,  0.],
                        [ 0., -1.,  0.,  1.,  0.,  0.],
                        [ 0.,  0., -1.,  1.,  0.,  0.]])

    c0 = smc._constraints_factor(cm_, k_params=6, idx_start=1)
    assert_equal(c0, cpairs2)

    c1 = smc._contrast_pairs(6, 4, 1)  # k_params, k_level, idx_start
    assert_equal(c1, cpairs2)
