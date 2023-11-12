'''Kernel Ridge Regression for local non-parametric regression'''


import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt


def kernel_rbf(x,y,scale=1, **kwds):
    #scale = kwds.get('scale',1)
    dist = ssp.minkowski_distance_p(x[:,np.newaxis,:],y[np.newaxis,:,:],2)
    return np.exp(-0.5/scale*(dist))


def kernel_euclid(x,y,p=2, **kwds):
    return ssp.minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)


class GaussProcess:
    '''class to perform kernel ridge regression (gaussian process)

    Warning: this class is memory intensive, it creates nobs x nobs distance
    matrix and its inverse, where nobs is the number of rows (observations).
    See sparse version for larger number of observations


    Notes
    -----

    Todo:
    * normalize multidimensional x array on demand, either by var or cov
    * add confidence band
    * automatic selection or proposal of smoothing parameters

    Note: this is different from kernel smoothing regression,
       see for example https://en.wikipedia.org/wiki/Kernel_smoother

    In this version of the kernel ridge regression, the training points
    are fitted exactly.
    Needs a fast version for leave-one-out regression, for fitting each
    observation on all the other points.
    This version could be numerically improved for the calculation for many
    different values of the ridge coefficient. see also short summary by
    Isabelle Guyon (ETHZ) in a manuscript KernelRidge.pdf

    Needs verification and possibly additional statistical results or
    summary statistics for interpretation, but this is a problem with
    non-parametric, non-linear methods.

    Reference
    ---------

    Rasmussen, C.E. and C.K.I. Williams, 2006, Gaussian Processes for Machine
    Learning, the MIT Press, www.GaussianProcess.org/gpal, chapter 2

    a short summary of the kernel ridge regression is at
    http://www.ics.uci.edu/~welling/teaching/KernelsICS273B/Kernel-Ridge.pdf
    '''

    def __init__(self, x, y=None, kernel=kernel_rbf,
                 scale=0.5, ridgecoeff = 1e-10, **kwds ):
        '''
        Parameters
        ----------
        x : 2d array (N,K)
           data array of explanatory variables, columns represent variables
           rows represent observations
        y : 2d array (N,1) (optional)
           endogenous variable that should be fitted or predicted
           can alternatively be specified as parameter to fit method
        kernel : function, default: kernel_rbf
           kernel: (x1,x2)->kernel matrix is a function that takes as parameter
           two column arrays and return the kernel or distance matrix
        scale : float (optional)
           smoothing parameter for the rbf kernel
        ridgecoeff : float (optional)
           coefficient that is multiplied with the identity matrix in the
           ridge regression

        Notes
        -----
        After initialization, kernel matrix is calculated and if y is given
        as parameter then also the linear regression parameter and the
        fitted or estimated y values, yest, are calculated. yest is available
        as an attribute in this case.

        Both scale and the ridge coefficient smooth the fitted curve.

        '''

        self.x = x
        self.kernel = kernel
        self.scale = scale
        self.ridgecoeff = ridgecoeff
        self.distxsample = kernel(x,x,scale=scale)
        self.Kinv = np.linalg.inv(self.distxsample +
                             np.eye(*self.distxsample.shape)*ridgecoeff)
        if y is not None:
            self.y = y
            self.yest = self.fit(y)


    def fit(self,y):
        '''fit the training explanatory variables to a sample ouput variable'''
        self.parest = np.dot(self.Kinv, y) #self.kernel(y,y,scale=self.scale))
        yhat = np.dot(self.distxsample,self.parest)
        return yhat

##        print ds33.shape
##        ds33_2 = kernel(x,x[::k,:],scale=scale)
##        dsinv = np.linalg.inv(ds33+np.eye(*distxsample.shape)*ridgecoeff)
##        B = np.dot(dsinv,y[::k,:])
    def predict(self,x):
        '''predict new y values for a given array of explanatory variables'''
        self.xpredict = x
        distxpredict = self.kernel(x, self.x, scale=self.scale)
        self.ypredict = np.dot(distxpredict, self.parest)
        return self.ypredict

    def plot(self, y, plt=plt ):
        '''some basic plots'''
        #todo return proper graph handles
        plt.figure()
        plt.plot(self.x,self.y, 'bo-', self.x, self.yest, 'r.-')
        plt.title('sample (training) points')
        plt.figure()
        plt.plot(self.xpredict,y,'bo-',self.xpredict,self.ypredict,'r.-')
        plt.title('all points')



def example1():
    m,k = 500,4
    upper = 6
    scale=10
    xs1a = np.linspace(1,upper,m)[:,np.newaxis]
    xs1 = xs1a*np.ones((1,4)) + 1/(1.0+np.exp(np.random.randn(m,k)))
    xs1 /= np.std(xs1[::k,:],0)   # normalize scale, could use cov to normalize
    y1true = np.sum(np.sin(xs1)+np.sqrt(xs1),1)[:,np.newaxis]
    y1 = y1true + 0.250 * np.random.randn(m,1)

    stride = 2 #use only some points as trainig points e.g 2 means every 2nd
    gp1 = GaussProcess(xs1[::stride,:],y1[::stride,:], kernel=kernel_euclid,
                       ridgecoeff=1e-10)
    yhatr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, yhatr1,'r.')
    plt.title('euclid kernel: true y versus noisy y and estimated y')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',yhatr1,'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')

    gp2 = GaussProcess(xs1[::stride,:],y1[::stride,:], kernel=kernel_rbf,
                       scale=scale, ridgecoeff=1e-1)
    yhatr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, yhatr2,'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',yhatr2,'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')
    #gp2.plot(y1)


def example2(m=100, scale=0.01, stride=2):
    #m,k = 100,1
    upper = 6
    xs1 = np.linspace(1,upper,m)[:,np.newaxis]
    y1true = np.sum(np.sin(xs1**2),1)[:,np.newaxis]/xs1
    y1 = y1true + 0.05*np.random.randn(m,1)

    ridgecoeff = 1e-10
    #stride = 2 #use only some points as trainig points e.g 2 means every 2nd
    gp1 = GaussProcess(xs1[::stride,:],y1[::stride,:], kernel=kernel_euclid,
                       ridgecoeff=1e-10)
    yhatr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, yhatr1,'r.')
    plt.title('euclid kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',yhatr1,'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')

    gp2 = GaussProcess(xs1[::stride,:],y1[::stride,:], kernel=kernel_rbf,
                       scale=scale, ridgecoeff=1e-2)
    yhatr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, yhatr2,'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',yhatr2,'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')
    #gp2.plot(y1)

if __name__ == '__main__':
    example2()
    #example2(m=1000, scale=0.01)
    #example2(m=100, scale=0.5)   # oversmoothing
    #example2(m=2000, scale=0.005) # this looks good for rbf, zoom in
    #example2(m=200, scale=0.01,stride=4)
    example1()
    #plt.show()
