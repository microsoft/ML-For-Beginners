'''
from David Huard's scipy sandbox, also attached to a ticket and
in the matplotlib-user mailinglist  (links ???)


Notes
=====

out of bounds interpolation raises exception and would not be completely
defined ::

>>> scoreatpercentile(x, [0,25,50,100])
Traceback (most recent call last):
...
    raise ValueError("A value in x_new is below the interpolation "
ValueError: A value in x_new is below the interpolation range.
>>> percentileofscore(x, [-50, 50])
Traceback (most recent call last):
...
    raise ValueError("A value in x_new is below the interpolation "
ValueError: A value in x_new is below the interpolation range.


idea
====

histogram and empirical interpolated distribution
-------------------------------------------------

dual constructor
* empirical cdf : cdf on all observations through linear interpolation
* binned cdf : based on histogram
both should work essentially the same, although pdf of empirical has
many spikes, fluctuates a lot
- alternative: binning based on interpolated cdf : example in script
* ppf: quantileatscore based on interpolated cdf
* rvs : generic from ppf
* stats, expectation ? how does integration wrt cdf work - theory?

Problems
* limits, lower and upper bound of support
  does not work or is undefined with empirical cdf and interpolation
* extending bounds ?
  matlab has pareto tails for empirical distribution, breaks linearity

empirical distribution with higher order interpolation
------------------------------------------------------

* should work easily enough with interpolating splines
* not piecewise linear
* can use pareto (or other) tails
* ppf how do I get the inverse function of a higher order spline?
  Chuck: resample and fit spline to inverse function
  this will have an approximation error in the inverse function
* -> does not work: higher order spline does not preserve monotonicity
  see mailing list for response to my question
* pmf from derivative available in spline

-> forget this and use kernel density estimator instead


bootstrap/empirical distribution:
---------------------------------

discrete distribution on real line given observations
what's defined?
* cdf : step function
* pmf : points with equal weight 1/nobs
* rvs : resampling
* ppf : quantileatscore on sample?
* moments : from data ?
* expectation ? sum_{all observations x} [func(x) * pmf(x)]
* similar for discrete distribution on real line
* References : ?
* what's the point? most of it is trivial, just for the record ?


Created on Monday, May 03, 2010, 11:47:03 AM
Author: josef-pktd, parts based on David Huard
License: BSD

'''
import scipy.interpolate as interpolate
import numpy as np

def scoreatpercentile(data, percentile):
    """Return the score at the given percentile of the data.

    Example:
        >>> data = randn(100)
            >>> scoreatpercentile(data, 50)

        will return the median of sample `data`.
    """
    per = np.array(percentile)
    cdf = empiricalcdf(data)
    interpolator = interpolate.interp1d(np.sort(cdf), np.sort(data))
    return interpolator(per/100.)

def percentileofscore(data, score):
    """Return the percentile-position of score relative to data.

    score: Array of scores at which the percentile is computed.

    Return percentiles (0-100).

    Example
            r = randn(50)
        x = linspace(-2,2,100)
        percentileofscore(r,x)

    Raise an error if the score is outside the range of data.
    """
    cdf = empiricalcdf(data)
    interpolator = interpolate.interp1d(np.sort(data), np.sort(cdf))
    return interpolator(score)*100.

def empiricalcdf(data, method='Hazen'):
    """Return the empirical cdf.

    Methods available:
        Hazen:       (i-0.5)/N
            Weibull:     i/(N+1)
        Chegodayev:  (i-.3)/(N+.4)
        Cunnane:     (i-.4)/(N+.2)
        Gringorten:  (i-.44)/(N+.12)
        California:  (i-1)/N

    Where i goes from 1 to N.
    """

    i = np.argsort(np.argsort(data)) + 1.
    N = len(data)
    method = method.lower()
    if method == 'hazen':
        cdf = (i-0.5)/N
    elif method == 'weibull':
        cdf = i/(N+1.)
    elif method == 'california':
        cdf = (i-1.)/N
    elif method == 'chegodayev':
        cdf = (i-.3)/(N+.4)
    elif method == 'cunnane':
        cdf = (i-.4)/(N+.2)
    elif method == 'gringorten':
        cdf = (i-.44)/(N+.12)
    else:
        raise ValueError('Unknown method. Choose among Weibull, Hazen,'
                         'Chegodayev, Cunnane, Gringorten and California.')

    return cdf


class HistDist:
    '''Distribution with piecewise linear cdf, pdf is step function

    can be created from empiricial distribution or from a histogram (not done yet)

    work in progress, not finished


    '''

    def __init__(self, data):
        self.data = np.atleast_1d(data)
        self.binlimit = np.array([self.data.min(), self.data.max()])
        sortind = np.argsort(data)
        self._datasorted = data[sortind]
        self.ranking = np.argsort(sortind)

        cdf = self.empiricalcdf()
        self._empcdfsorted = np.sort(cdf)
        self.cdfintp = interpolate.interp1d(self._datasorted, self._empcdfsorted)
        self.ppfintp = interpolate.interp1d(self._empcdfsorted, self._datasorted)

    def empiricalcdf(self, data=None, method='Hazen'):
        """Return the empirical cdf.

        Methods available:
            Hazen:       (i-0.5)/N
                Weibull:     i/(N+1)
            Chegodayev:  (i-.3)/(N+.4)
            Cunnane:     (i-.4)/(N+.2)
            Gringorten:  (i-.44)/(N+.12)
            California:  (i-1)/N

        Where i goes from 1 to N.
        """

        if data is None:
            data = self.data
            i = self.ranking
        else:
            i = np.argsort(np.argsort(data)) + 1.

        N = len(data)
        method = method.lower()
        if method == 'hazen':
            cdf = (i-0.5)/N
        elif method == 'weibull':
            cdf = i/(N+1.)
        elif method == 'california':
            cdf = (i-1.)/N
        elif method == 'chegodayev':
            cdf = (i-.3)/(N+.4)
        elif method == 'cunnane':
            cdf = (i-.4)/(N+.2)
        elif method == 'gringorten':
            cdf = (i-.44)/(N+.12)
        else:
            raise ValueError('Unknown method. Choose among Weibull, Hazen,'
                             'Chegodayev, Cunnane, Gringorten and California.')

        return cdf


    def cdf_emp(self, score):
        '''
        this is score in dh

        '''
        return self.cdfintp(score)
        #return percentileofscore(self.data, score)

    def ppf_emp(self, quantile):
        '''
        this is score in dh

        '''
        return self.ppfintp(quantile)
        #return scoreatpercentile(self.data, quantile*100)


    #from DHuard http://old.nabble.com/matplotlib-f2903.html
    def optimize_binning(self, method='Freedman'):
        """Find the optimal number of bins and update the bin countaccordingly.
        Available methods : Freedman
                            Scott
        """

        nobs = len(self.data)
        if method=='Freedman':
            IQR = self.ppf_emp(0.75) - self.ppf_emp(0.25) # Interquantile range(75% -25%)
            width = 2* IQR* nobs**(-1./3)

        elif method=='Scott':
            width = 3.49 * np.std(self.data) * nobs**(-1./3)

        self.nbin = (np.ptp(self.binlimit)/width)
        return self.nbin


#changes: josef-pktd
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nobs = 100
    x = np.random.randn(nobs)

    examples = [2]
    if 1 in examples:
        empiricalcdf(x)
        print(percentileofscore(x, 0.5))
        print(scoreatpercentile(x, 50))
        xsupp = np.linspace(x.min(), x.max())
        pos = percentileofscore(x, xsupp)
        plt.plot(xsupp, pos)
        #perc = np.linspace(2.5, 97.5)
        #plt.plot(scoreatpercentile(x, perc), perc)
        plt.plot(scoreatpercentile(x, pos), pos+1)


        #emp = interpolate.PiecewisePolynomial(np.sort(empiricalcdf(x)), np.sort(x))
        emp=interpolate.InterpolatedUnivariateSpline(np.sort(x),np.sort(empiricalcdf(x)),k=1)
        pdfemp = np.array([emp.derivatives(xi)[1] for xi in xsupp])
        plt.figure()
        plt.plot(xsupp,pdfemp)
        cdf_ongrid = emp(xsupp)
        plt.figure()
        plt.plot(xsupp, cdf_ongrid)

        #get pdf from interpolated cdf on a regular grid
        plt.figure()
        plt.step(xsupp[:-1],np.diff(cdf_ongrid)/np.diff(xsupp))

        #reduce number of bins/steps
        xsupp2 = np.linspace(x.min(), x.max(), 25)
        plt.figure()
        plt.step(xsupp2[:-1],np.diff(emp(xsupp2))/np.diff(xsupp2))

        #pdf using 25 original observations, every (nobs/25)th
        xso = np.sort(x)
        xs = xso[::nobs/25]
        plt.figure()
        plt.step(xs[:-1],np.diff(emp(xs))/np.diff(xs))
        #lower end looks strange


    histd = HistDist(x)
    print(histd.optimize_binning())
    print(histd.cdf_emp(histd.binlimit))
    print(histd.ppf_emp([0.25, 0.5, 0.75]))
    print(histd.cdf_emp([-0.5, -0.25, 0, 0.25, 0.5]))


    xsupp = np.linspace(x.min(), x.max(), 500)
    emp=interpolate.InterpolatedUnivariateSpline(np.sort(x),np.sort(empiricalcdf(x)),k=1)
    #pdfemp = np.array([emp.derivatives(xi)[1] for xi in xsupp])
    #plt.figure()
    #plt.plot(xsupp,pdfemp)
    cdf_ongrid = emp(xsupp)
    plt.figure()
    plt.plot(xsupp, cdf_ongrid)
    ppfintp = interpolate.InterpolatedUnivariateSpline(cdf_ongrid,xsupp,k=3)

    ppfs = ppfintp(cdf_ongrid)
    plt.plot(ppfs, cdf_ongrid)
    #ppfemp=interpolate.InterpolatedUnivariateSpline(np.sort(empiricalcdf(x)),np.sort(x),k=3)
    #Do not use interpolating splines for function approximation
    #with s=0.03 the spline is monotonic at the evaluated values
    ppfemp=interpolate.UnivariateSpline(np.sort(empiricalcdf(x)),np.sort(x),k=3, s=0.03)
    ppfe = ppfemp(cdf_ongrid)
    plt.plot(ppfe, cdf_ongrid)

    print('negative density')
    print('(np.diff(ppfs)).min()', (np.diff(ppfs)).min())
    print('(np.diff(cdf_ongrid)).min()', (np.diff(cdf_ongrid)).min())
    #plt.show()
