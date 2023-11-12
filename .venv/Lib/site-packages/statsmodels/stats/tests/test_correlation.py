import numpy as np
from statsmodels.stats.correlation_tools import (
        kernel_covariance, GaussianMultivariateKernel)
from numpy.testing import assert_allclose


def test_kernel_covariance():

    np.random.seed(342)

    # Number of independent observations
    ng = 1000

    # Dimension of the process
    p = 3

    # Each component of the process in an AR(r) with 10 values
    # observed on a grid
    r = 0.5
    ii = np.arange(10)
    qm = r**np.abs(np.subtract.outer(ii, ii))
    qm = np.linalg.cholesky(qm)

    exog, groups, pos = [], [], []
    for j in range(ng):
        pos1 = np.arange(10)[:, None]
        groups1 = j * np.ones(10)

        # The components are independent AR processes
        ex1 = np.random.normal(size=(10, 3))

        ex1 = np.dot(qm, ex1)
        pos.append(pos1)
        groups.append(groups1)
        exog.append(ex1)

    groups = np.concatenate(groups)
    pos = np.concatenate(pos, axis=0)
    exog = np.concatenate(exog, axis=0)

    for j in range(4):

        if j == 0:
            kernel = None
            bw = None
        elif j == 1:
            kernel = GaussianMultivariateKernel()
            bw = None
        elif j == 2:
            kernel = GaussianMultivariateKernel()
            bw = 1
        elif j == 3:
            kernel = GaussianMultivariateKernel()
            bw = kernel.set_default_bw(pos)

        cv = kernel_covariance(exog, pos, groups, kernel=kernel, bw=bw)
        assert_allclose(cv(0, 0), np.eye(p), atol=0.1, rtol=0.01)
        assert_allclose(cv(0, 1), 0.5*np.eye(p), atol=0.1, rtol=0.01)
        assert_allclose(cv(0, 2), 0.25*np.eye(p), atol=0.1, rtol=0.01)
        assert_allclose(cv(1, 2), 0.5*np.eye(p), atol=0.1, rtol=0.01)
