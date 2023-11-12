# -*- coding: utf-8 -*-
from statsmodels.compat.python import lzip

import numpy as np

from statsmodels.tools.validation import array_like
from . import kernels


#TODO: should this be a function?
class KDE:
    """
    Kernel Density Estimator

    Parameters
    ----------
    x : array_like
        N-dimensional array from which the density is to be estimated
    kernel : Kernel Class
        Should be a class from *
    """
    #TODO: amend docs for Nd case?
    def __init__(self, x, kernel=None):
        x = array_like(x, "x", maxdim=2, contiguous=True)
        if x.ndim == 1:
            x = x[:,None]

        nobs, n_series = x.shape

        if kernel is None:
            kernel = kernels.Gaussian()  # no meaningful bandwidth yet

        if n_series > 1:
            if isinstance( kernel, kernels.CustomKernel ):
                kernel = kernels.NdKernel(n_series, kernels = kernel)

        self.kernel = kernel
        self.n = n_series  #TODO change attribute
        self.x = x

    def density(self, x):
        return self.kernel.density(self.x, x)

    def __call__(self, x, h="scott"):
        return np.array([self.density(xx) for xx in x])

    def evaluate(self, x, h="silverman"):
        density = self.kernel.density
        return np.array([density(xx) for xx in x])


if __name__ == "__main__":
    from numpy import random
    import matplotlib.pyplot as plt
    import statsmodels.nonparametric.bandwidths as bw
    from statsmodels.sandbox.nonparametric.testdata import kdetest

    # 1-D case
    random.seed(142)
    x = random.standard_t(4.2, size = 50)
    h = bw.bw_silverman(x)
    #NOTE: try to do it with convolution
    support = np.linspace(-10,10,512)


    kern = kernels.Gaussian(h = h)
    kde = KDE( x, kern)
    print(kde.density(1.015469))
    print(0.2034675)
    Xs = np.arange(-10,10,0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Xs, kde(Xs), "-")
    ax.set_ylim(-10, 10)
    ax.set_ylim(0,0.4)


    # 2-D case
    x = lzip(kdetest.faithfulData["eruptions"], kdetest.faithfulData["waiting"])
    x = np.array(x)
    x = (x - x.mean(0))/x.std(0)
    nobs = x.shape[0]
    H = kdetest.Hpi
    kern = kernels.NdKernel( 2 )
    kde = KDE( x, kern )
    print(kde.density( np.matrix( [1,2 ]))) #.T
    plt.figure()
    plt.plot(x[:,0], x[:,1], 'o')


    n_grid = 50
    xsp = np.linspace(x.min(0)[0], x.max(0)[0], n_grid)
    ysp = np.linspace(x.min(0)[1], x.max(0)[1], n_grid)
#    xsorted = np.sort(x)
#    xlow = xsorted[nobs/4]
#    xupp = xsorted[3*nobs/4]
#    xsp = np.linspace(xlow[0], xupp[0], n_grid)
#    ysp = np.linspace(xlow[1], xupp[1], n_grid)
    xr, yr = np.meshgrid(xsp, ysp)
    kde_vals = np.array([kde.density( np.matrix( [xi, yi ]) ) for xi, yi in
               zip(xr.ravel(), yr.ravel())])
    plt.contour(xsp, ysp, kde_vals.reshape(n_grid, n_grid))

    plt.show()


    # 5 D case
#    random.seed(142)
#    mu = [1.0, 4.0, 3.5, -2.4, 0.0]
#    sigma = np.matrix(
#        [[ 0.6 - 0.1*abs(i-j) if i != j else 1.0 for j in xrange(5)] for i in xrange(5)])
#    x = random.multivariate_normal(mu, sigma, size = 100)
#    kern = kernel.Gaussian()
#    kde = KernelEstimate( x, kern )
