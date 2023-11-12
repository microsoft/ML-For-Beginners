'''subclassing kde

Author: josef pktd
'''

import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt


class gaussian_kde_set_covariance(stats.gaussian_kde):
    '''
    from Anne Archibald in mailinglist:
    http://www.nabble.com/Width-of-the-gaussian-in-stats.kde.gaussian_kde---td19558924.html#a19558924
    '''
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance(self):
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) * self.n


class gaussian_kde_covfact(stats.gaussian_kde):
    def __init__(self, dataset, covfact = 'scotts'):
        self.covfact = covfact
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance_(self):
        '''not used'''
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) * self.n

    def covariance_factor(self):
        if self.covfact in ['sc', 'scotts']:
            return self.scotts_factor()
        if self.covfact in ['si', 'silverman']:
            return self.silverman_factor()
        elif self.covfact:
            return float(self.covfact)
        else:
            raise ValueError('covariance factor has to be scotts, silverman or a number')

    def reset_covfact(self, covfact):
        self.covfact = covfact
        self.covariance_factor()
        self._compute_covariance()

def plotkde(covfact):
    gkde.reset_covfact(covfact)
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    # plot histgram of sample
    plt.hist(xn, bins=20, normed=1)
    # plot estimated density
    plt.plot(ind, kdepdf, label='kde', color="g")
    # plot data generating density
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) +
                  (1-alpha) * stats.norm.pdf(ind, loc=mhigh),
                  color="r", label='DGP: normal mix')
    plt.title('Kernel Density Estimation - ' + str(gkde.covfact))
    plt.legend()


def test_kde_1d():
    np.random.seed(8765678)
    n_basesample = 500
    xn = np.random.randn(n_basesample)
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)
    print(xnmean, xnstd)

    # get kde for original sample
    gkde = stats.gaussian_kde(xn)

    # evaluate the density function for the kde for some points
    xs = np.linspace(-7,7,501)
    kdepdf = gkde.evaluate(xs)
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    print('MSE', np.sum((kdepdf - normpdf)**2))
    print('axabserror', np.max(np.abs(kdepdf - normpdf)))
    intervall = xs[1] - xs[0]
    assert_(np.sum((kdepdf - normpdf)**2)*intervall < 0.01)
    #assert_array_almost_equal(kdepdf, normpdf, decimal=2)
    print(gkde.integrate_gaussian(0.0, 1.0))
    print(gkde.integrate_box_1d(-np.inf, 0.0))
    print(gkde.integrate_box_1d(0.0, np.inf))
    print(gkde.integrate_box_1d(-np.inf, xnmean))
    print(gkde.integrate_box_1d(xnmean, np.inf))

    assert_almost_equal(gkde.integrate_box_1d(xnmean, np.inf), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box_1d(-np.inf, xnmean), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), 0.5, decimal=1)

    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*intervall, decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf*normpdf).sum()*intervall, decimal=2)
##    assert_almost_equal(gkde.integrate_gaussian(0.0, 1.0),
##                        (kdepdf*normpdf).sum()*intervall, decimal=2)




if __name__ == '__main__':
    # generate a sample
    n_basesample = 1000
    np.random.seed(8765678)
    alpha = 0.6 #weight for (prob of) lower distribution
    mlow, mhigh = (-3,3)  #mean locations for gaussian mixture
    xn =  np.concatenate([mlow + np.random.randn(alpha * n_basesample),
                       mhigh + np.random.randn((1-alpha) * n_basesample)])

    # get kde for original sample
    #gkde = stats.gaussian_kde(xn)
    gkde = gaussian_kde_covfact(xn, 0.1)
    # evaluate the density function for the kde for some points
    ind = np.linspace(-7,7,101)
    kdepdf = gkde.evaluate(ind)

    plt.figure()
    # plot histgram of sample
    plt.hist(xn, bins=20, normed=1)
    # plot estimated density
    plt.plot(ind, kdepdf, label='kde', color="g")
    # plot data generating density
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) +
                  (1-alpha) * stats.norm.pdf(ind, loc=mhigh),
                  color="r", label='DGP: normal mix')
    plt.title('Kernel Density Estimation')
    plt.legend()

    gkde = gaussian_kde_covfact(xn, 'scotts')
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    # plot histgram of sample
    plt.hist(xn, bins=20, normed=1)
    # plot estimated density
    plt.plot(ind, kdepdf, label='kde', color="g")
    # plot data generating density
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) +
                  (1-alpha) * stats.norm.pdf(ind, loc=mhigh),
                  color="r", label='DGP: normal mix')
    plt.title('Kernel Density Estimation')
    plt.legend()
    #plt.show()
    for cv in ['scotts', 'silverman', 0.05, 0.1, 0.5]:
        plotkde(cv)

    test_kde_1d()


    np.random.seed(8765678)
    n_basesample = 1000
    xn = np.random.randn(n_basesample)
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)

    # get kde for original sample
    gkde = stats.gaussian_kde(xn)
