# -*- coding: utf-8 -*-
# some cut and paste characters are not ASCII
'''density estimation based on orthogonal polynomials


Author: Josef Perktold
Created: 2011-05017
License: BSD

2 versions work: based on Fourier, FPoly, and chebychev T, ChebyTPoly
also hermite polynomials, HPoly, works
other versions need normalization


TODO:

* check fourier case again:  base is orthonormal,
  but needs offsetfact = 0 and does not integrate to 1, rescaled looks good
* hermite: works but DensityOrthoPoly requires currently finite bounds
  I use it with offsettfactor 0.5 in example
* not implemented methods:
  - add bonafide density correction
  - add transformation to domain of polynomial base - DONE
    possible problem: what is the behavior at the boundary,
    offsetfact requires more work, check different cases, add as option
    moved to polynomial class by default, as attribute
* convert examples to test cases
* need examples with large density on boundary, beta ?
* organize poly classes in separate module, check new numpy.polynomials,
  polyvander
* MISE measures, order selection, ...

enhancements:
  * other polynomial bases: especially for open and half open support
  * wavelets
  * local or piecewise approximations


'''
from scipy import stats, integrate, special

import numpy as np


sqr2 = np.sqrt(2.)

class FPoly:
    '''Orthonormal (for weight=1) Fourier Polynomial on [0,1]

    orthonormal polynomial but density needs corfactor that I do not see what
    it is analytically

    parameterization on [0,1] from

    Sam Efromovich: Orthogonal series density estimation,
    2010 John Wiley & Sons, Inc. WIREs Comp Stat 2010 2 467-476


    '''

    def __init__(self, order):
        self.order = order
        self.domain = (0, 1)
        self.intdomain = self.domain

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x)
        else:
            return sqr2 * np.cos(np.pi * self.order * x)

class F2Poly:
    '''Orthogonal (for weight=1) Fourier Polynomial on [0,pi]

    is orthogonal but first component does not square-integrate to 1
    final result seems to need a correction factor of sqrt(pi)
    _corfactor = sqrt(pi) from integrating the density

    Parameterization on [0, pi] from

    Peter Hall, Cross-Validation and the Smoothing of Orthogonal Series Density
    Estimators, JOURNAL OF MULTIVARIATE ANALYSIS 21, 189-206 (1987)

    '''

    def __init__(self, order):
        self.order = order
        self.domain = (0, np.pi)
        self.intdomain = self.domain
        self.offsetfactor = 0

    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / np.sqrt(np.pi)
        else:
            return sqr2 * np.cos(self.order * x) / np.sqrt(np.pi)

class ChebyTPoly:
    '''Orthonormal (for weight=1) Chebychev Polynomial on (-1,1)


    Notes
    -----
    integration requires to stay away from boundary, offsetfactor > 0
    maybe this implies that we cannot use it for densities that are > 0 at
    boundary ???

    or maybe there is a mistake close to the boundary, sometimes integration works.

    '''

    def __init__(self, order):
        self.order = order
        from scipy.special import chebyt
        self.poly = chebyt(order)
        self.domain = (-1, 1)
        self.intdomain = (-1+1e-6, 1-1e-6)
        #not sure if I need this, in integration nans are possible  on the boundary
        self.offsetfactor = 0.01  #required for integration


    def __call__(self, x):
        if self.order == 0:
            return np.ones_like(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi)

        else:
            return self.poly(x) / (1-x**2)**(1/4.) /np.sqrt(np.pi) *np.sqrt(2)


logpi2 = np.log(np.pi)/2

class HPoly:
    '''Orthonormal (for weight=1) Hermite Polynomial, uses finite bounds

    for current use with DensityOrthoPoly domain is defined as [-6,6]

    '''
    def __init__(self, order):
        self.order = order
        from scipy.special import hermite
        self.poly = hermite(order)
        self.domain = (-6, +6)
        self.offsetfactor = 0.5  # note this is

    def __call__(self, x):
        k = self.order

        lnfact = -(1./2)*(k*np.log(2.) + special.gammaln(k+1) + logpi2) - x*x/2
        fact = np.exp(lnfact)

        return self.poly(x) * fact

def polyvander(x, polybase, order=5):
    polyarr = np.column_stack([polybase(i)(x) for i in range(order)])
    return polyarr

def inner_cont(polys, lower, upper, weight=None):
    '''inner product of continuous function (with weight=1)

    Parameters
    ----------
    polys : list of callables
        polynomial instances
    lower : float
        lower integration limit
    upper : float
        upper integration limit
    weight : callable or None
        weighting function

    Returns
    -------
    innp : ndarray
        symmetric 2d square array with innerproduct of all function pairs
    err : ndarray
        numerical error estimate from scipy.integrate.quad, same dimension as innp

    Examples
    --------
    >>> from scipy.special import chebyt
    >>> polys = [chebyt(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[ 2.        ,  0.        , -0.66666667,  0.        ],
           [ 0.        ,  0.66666667,  0.        , -0.4       ],
           [-0.66666667,  0.        ,  0.93333333,  0.        ],
           [ 0.        , -0.4       ,  0.        ,  0.97142857]])

    '''
    n_polys = len(polys)
    innerprod = np.empty((n_polys, n_polys))
    innerprod.fill(np.nan)
    interr = np.zeros((n_polys, n_polys))

    for i in range(n_polys):
        for j in range(i+1):
            p1 = polys[i]
            p2 = polys[j]
            if weight is not None:
                innp, err = integrate.quad(lambda x: p1(x)*p2(x)*weight(x),
                                       lower, upper)
            else:
                innp, err = integrate.quad(lambda x: p1(x)*p2(x), lower, upper)
            innerprod[i,j] = innp
            interr[i,j] = err
            if not i == j:
                innerprod[j,i] = innp
                interr[j,i] = err

    return innerprod, interr


def is_orthonormal_cont(polys, lower, upper, rtol=0, atol=1e-08):
    '''check whether functions are orthonormal

    Parameters
    ----------
    polys : list of polynomials or function

    Returns
    -------
    is_orthonormal : bool
        is False if the innerproducts are not close to 0 or 1

    Notes
    -----
    this stops as soon as the first deviation from orthonormality is found.

    Examples
    --------
    >>> from scipy.special import chebyt
    >>> polys = [chebyt(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[ 2.        ,  0.        , -0.66666667,  0.        ],
           [ 0.        ,  0.66666667,  0.        , -0.4       ],
           [-0.66666667,  0.        ,  0.93333333,  0.        ],
           [ 0.        , -0.4       ,  0.        ,  0.97142857]])
    >>> is_orthonormal_cont(polys, -1, 1, atol=1e-6)
    False

    >>> polys = [ChebyTPoly(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[  1.00000000e+00,   0.00000000e+00,  -9.31270888e-14,
              0.00000000e+00],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
             -9.47850712e-15],
           [ -9.31270888e-14,   0.00000000e+00,   1.00000000e+00,
              0.00000000e+00],
           [  0.00000000e+00,  -9.47850712e-15,   0.00000000e+00,
              1.00000000e+00]])
    >>> is_orthonormal_cont(polys, -1, 1, atol=1e-6)
    True

    '''
    for i in range(len(polys)):
        for j in range(i+1):
            p1 = polys[i]
            p2 = polys[j]
            innerprod = integrate.quad(lambda x: p1(x)*p2(x), lower, upper)[0]
            #print i,j, innerprod
            if not np.allclose(innerprod, i==j, rtol=rtol, atol=atol):
                return False
    return True



#new versions


class DensityOrthoPoly:
    '''Univariate density estimation by orthonormal series expansion


    Uses an orthonormal polynomial basis to approximate a univariate density.


    currently all arguments can be given to fit, I might change it to requiring
    arguments in __init__ instead.
    '''

    def __init__(self, polybase=None, order=5):
        if polybase is not None:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]
        #try:
        #self.offsetfac = 0.05
        #self.offsetfac = polys[0].offsetfactor #polys maybe not defined yet
        self._corfactor = 1
        self._corshift = 0


    def fit(self, x, polybase=None, order=5, limits=None):
        '''estimate the orthogonal polynomial approximation to the density

        '''
        if polybase is None:
            polys = self.polys[:order]
        else:
            self.polybase = polybase
            self.polys = polys = [polybase(i) for i in range(order)]

        #move to init ?
        if not hasattr(self, 'offsetfac'):
            self.offsetfac = polys[0].offsetfactor


        xmin, xmax = x.min(), x.max()
        if limits is None:
            self.offset = offset = (xmax - xmin) * self.offsetfac
            limits = self.limits = (xmin - offset, xmax + offset)

        interval_length = limits[1] - limits[0]
        xinterval = xmax - xmin
        # need to cover (half-)open intervalls
        self.shrink = 1. / interval_length #xinterval/interval_length
        offset = (interval_length - xinterval ) / 2.
        self.shift = xmin - offset

        self.x = x = self._transform(x)

        coeffs = [(p(x)).mean() for p in polys]
        self.coeffs = coeffs
        self.polys = polys
        self._verify()  #verify that it is a proper density

        return self #coeffs, polys

    def evaluate(self, xeval, order=None):
        xeval = self._transform(xeval)
        if order is None:
            order = len(self.polys)
        res = sum(c*p(xeval) for c, p in list(zip(self.coeffs, self.polys))[:order])
        res = self._correction(res)
        return res

    def __call__(self, xeval):
        '''alias for evaluate, except no order argument'''
        return self.evaluate(xeval)

    def _verify(self):
        '''check for bona fide density correction

        currently only checks that density integrates to 1

`       non-negativity - NotImplementedYet
        '''
        #watch out for circular/recursive usage

        #evaluate uses domain of data, we stay offset away from bounds
        intdomain = self.limits #self.polys[0].intdomain
        self._corfactor = 1./integrate.quad(self.evaluate, *intdomain)[0]
        #self._corshift = 0
        #self._corfactor
        return self._corfactor



    def _correction(self, x):
        '''bona fide density correction

        affine shift of density to make it into a proper density

        '''
        if self._corfactor != 1:
            x *= self._corfactor

        if self._corshift != 0:
            x += self._corshift

        return x

    def _transform(self, x): # limits=None):
        '''transform observation to the domain of the density


        uses shrink and shift attribute which are set in fit to stay


        '''

        #use domain from first instance
        #class does not have domain  self.polybase.domain[0] AttributeError
        domain = self.polys[0].domain

        ilen = (domain[1] - domain[0])
        shift = self.shift - domain[0]/self.shrink/ilen
        shrink = self.shrink * ilen

        return (x - shift) * shrink


#old version as a simple function
def density_orthopoly(x, polybase, order=5, xeval=None):
    #polybase = legendre  #chebyt #hermitenorm#
    #polybase = chebyt
    #polybase = FPoly
    #polybase = ChtPoly
    #polybase = hermite
    #polybase = HPoly

    if xeval is None:
        xeval = np.linspace(x.min(),x.max(),50)

    #polys = [legendre(i) for i in range(order)]
    polys = [polybase(i) for i in range(order)]
    #coeffs = [(p(x)*(1-x**2)**(-1/2.)).mean() for p in polys]
    #coeffs = [(p(x)*np.exp(-x*x)).mean() for p in polys]
    coeffs = [(p(x)).mean() for p in polys]
    res = sum(c*p(xeval) for c, p in zip(coeffs, polys))
    #res *= (1-xeval**2)**(-1/2.)
    #res *= np.exp(-xeval**2./2)
    return res, xeval, coeffs, polys



if __name__ == '__main__':

    examples = ['chebyt', 'fourier', 'hermite']#[2]

    nobs = 10000

    import matplotlib.pyplot as plt
    from statsmodels.distributions.mixture_rvs import (
                                                mixture_rvs, MixtureDistribution)

    #np.random.seed(12345)
##    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
##                   kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.75)))
    mix_kwds = (dict(loc=-0.5,scale=.5),dict(loc=1,scale=.2))
    obs_dist = mixture_rvs([1/3.,2/3.], size=nobs, dist=[stats.norm, stats.norm],
                   kwargs=mix_kwds)
    mix = MixtureDistribution()

    #obs_dist = np.random.randn(nobs)/4. #np.sqrt(2)


    if "chebyt_" in examples: # needed for Cheby example below
        #obs_dist = np.clip(obs_dist, -2, 2)/2.01
        #chebyt [0,1]
        obs_dist = obs_dist[(obs_dist>-2) & (obs_dist<2)]/2.0 #/4. + 2/4.0
        #fourier [0,1]
        #obs_dist = obs_dist[(obs_dist>-2) & (obs_dist<2)]/4. + 2/4.0
        f_hat, grid, coeffs, polys = density_orthopoly(obs_dist, ChebyTPoly, order=20, xeval=None)
        #f_hat /= f_hat.sum() * (grid.max() - grid.min())/len(grid)
        f_hat0 = f_hat
        fint = integrate.trapz(f_hat, grid)# dx=(grid.max() - grid.min())/len(grid))
        #f_hat -= fint/2.
        print('f_hat.min()', f_hat.min())
        f_hat = (f_hat - f_hat.min()) #/ f_hat.max() - f_hat.min
        fint2 = integrate.trapz(f_hat, grid)# dx=(grid.max() - grid.min())/len(grid))
        print('fint2', fint, fint2)
        f_hat /= fint2

        # note that this uses a *huge* grid by default
        #f_hat, grid = kdensityfft(emp_dist, kernel="gauss", bw="scott")

        # check the plot

        doplot = 0
        if doplot:
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.plot(grid, f_hat, lw=2, color='black')
            plt.plot(grid, f_hat0, lw=2, color='g')
            plt.show()

        for i,p in enumerate(polys[:5]):
            for j,p2 in enumerate(polys[:5]):
                print(i,j,integrate.quad(lambda x: p(x)*p2(x), -1,1)[0])

        for p in polys:
            print(integrate.quad(lambda x: p(x)**2, -1,1))


    #examples using the new class

    if "chebyt" in examples:
        dop = DensityOrthoPoly().fit(obs_dist, ChebyTPoly, order=20)
        grid = np.linspace(obs_dist.min(), obs_dist.max())
        xf = dop(grid)
        #print('np.max(np.abs(xf - f_hat0))', np.max(np.abs(xf - f_hat0))
        dopint = integrate.quad(dop, *dop.limits)[0]
        print('dop F integral', dopint)
        mpdf = mix.pdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm],
                   kwargs=mix_kwds)

        doplot = 1
        if doplot:
            plt.figure()
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.plot(grid, xf, lw=2, color='black')
            plt.plot(grid, mpdf, lw=2, color='green')
            plt.title('using Chebychev polynomials')
            #plt.show()

    if "fourier" in examples:
        dop = DensityOrthoPoly()
        dop.offsetfac = 0.5
        dop = dop.fit(obs_dist, F2Poly, order=30)
        grid = np.linspace(obs_dist.min(), obs_dist.max())
        xf = dop(grid)
        #print(np.max(np.abs(xf - f_hat0))
        dopint = integrate.quad(dop, *dop.limits)[0]
        print('dop F integral', dopint)
        mpdf = mix.pdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm],
                   kwargs=mix_kwds)

        doplot = 1
        if doplot:
            plt.figure()
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.title('using Fourier polynomials')
            plt.plot(grid, xf, lw=2, color='black')
            plt.plot(grid, mpdf, lw=2, color='green')
            #plt.show()

        #check orthonormality:
        print(np.max(np.abs(inner_cont(dop.polys[:5], 0, 1)[0] -np.eye(5))))

    if "hermite" in examples:
        dop = DensityOrthoPoly()
        dop.offsetfac = 0
        dop = dop.fit(obs_dist, HPoly, order=20)
        grid = np.linspace(obs_dist.min(), obs_dist.max())
        xf = dop(grid)
        #print(np.max(np.abs(xf - f_hat0))
        dopint = integrate.quad(dop, *dop.limits)[0]
        print('dop F integral', dopint)

        mpdf = mix.pdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm],
                   kwargs=mix_kwds)

        doplot = 1
        if doplot:
            plt.figure()
            plt.hist(obs_dist, bins=50, normed=True, color='red')
            plt.plot(grid, xf, lw=2, color='black')
            plt.plot(grid, mpdf, lw=2, color='green')
            plt.title('using Hermite polynomials')
            plt.show()

        #check orthonormality:
        print(np.max(np.abs(inner_cont(dop.polys[:5], 0, 1)[0] -np.eye(5))))


    #check orthonormality

    hpolys = [HPoly(i) for i in range(5)]
    inn = inner_cont(hpolys, -6, 6)[0]
    print(np.max(np.abs(inn - np.eye(5))))
    print((inn*100000).astype(int))

    from scipy.special import hermite, chebyt
    htpolys = [hermite(i) for i in range(5)]
    innt = inner_cont(htpolys, -10, 10)[0]
    print((innt*100000).astype(int))

    polysc = [chebyt(i) for i in range(4)]
    r, e = inner_cont(polysc, -1, 1, weight=lambda x: (1-x*x)**(-1/2.))
    print(np.max(np.abs(r - np.diag(np.diag(r)))))
