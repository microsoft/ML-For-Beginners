'''using scipy signal and numpy correlate to calculate some time series
statistics

original developer notes

see also scikits.timeseries  (movstat is partially inspired by it)
added 2009-08-29
timeseries moving stats are in c, autocorrelation similar to here
I thought I saw moving stats somewhere in python, maybe not)


TODO

moving statistics
- filters do not handle boundary conditions nicely (correctly ?)
e.g. minimum order filter uses 0 for out of bounds value
-> append and prepend with last resp. first value
- enhance for nd arrays, with axis = 0



Note: Equivalence for 1D signals
>>> np.all(signal.correlate(x,[1,1,1],'valid')==np.correlate(x,[1,1,1]))
True
>>> np.all(ndimage.filters.correlate(x,[1,1,1], origin = -1)[:-3+1]==np.correlate(x,[1,1,1]))
True

# multidimensional, but, it looks like it uses common filter across time series, no VAR
ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)
ndimage.filters.correlate(x,[1,1,1],origin = 1))
ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)

>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)[0]==\
ndimage.filters.correlate(x,[1,1,1],origin = 1))
True
>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)[0]==ndimage.filters.correlate(x,[1,1,1],origin = 1))


update
2009-09-06: cosmetic changes, rearrangements
'''

import numpy as np
from scipy import signal

from numpy.testing import assert_array_equal, assert_array_almost_equal


def expandarr(x,k):
    #make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd)*x[0],x,np.ones(kadd)*x[-1]]

def movorder(x, order = 'med', windsize=3, lag='lagged'):
    '''moving order statistics

    Parameters
    ----------
    x : ndarray
       time series data
    order : float or 'med', 'min', 'max'
       which order statistic to calculate
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    filtered array


    '''

    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        lead = windsize//2
    elif lag == 'centered':
        lead = 0
    elif lag == 'leading':
        lead = -windsize//2 +1
    else:
        raise ValueError
    if np.isfinite(order): #if np.isnumber(order):
        ord = order   # note: ord is a builtin function
    elif order == 'med':
        ord = (windsize - 1)/2
    elif order == 'min':
        ord = 0
    elif order == 'max':
        ord = windsize - 1
    else:
        raise ValueError

    #return signal.order_filter(x,np.ones(windsize),ord)[:-lead]
    xext = expandarr(x, windsize)
    #np.r_[np.ones(windsize)*x[0],x,np.ones(windsize)*x[-1]]
    return signal.order_filter(xext,np.ones(windsize),ord)[windsize-lead:-(windsize+lead)]

def check_movorder():
    '''graphical test for movorder'''
    import matplotlib.pylab as plt
    x = np.arange(1,10)
    xo = movorder(x, order='max')
    assert_array_equal(xo, x)
    x = np.arange(10,1,-1)
    xo = movorder(x, order='min')
    assert_array_equal(xo, x)
    assert_array_equal(movorder(x, order='min', lag='centered')[:-1], x[1:])

    tt = np.linspace(0,2*np.pi,15)
    x = np.sin(tt) + 1
    xo = movorder(x, order='max')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max lagged')
    xo = movorder(x, order='max', lag='centered')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max centered')
    xo = movorder(x, order='max', lag='leading')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max leading')

# identity filter
##>>> signal.order_filter(x,np.ones(1),0)
##array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
# median filter
##signal.medfilt(np.sin(x), kernel_size=3)
##>>> plt.figure()
##<matplotlib.figure.Figure object at 0x069BBB50>
##>>> x=np.linspace(0,3,100);plt.plot(x,np.sin(x),x,signal.medfilt(np.sin(x), kernel_size=3))

# remove old version
##def movmeanvar(x, windowsize=3, valid='same'):
##    '''
##    this should also work along axis or at least for columns
##    '''
##    n = x.shape[0]
##    x = expandarr(x, windowsize - 1)
##    takeslice = slice(windowsize-1, n + windowsize-1)
##    avgkern = (np.ones(windowsize)/float(windowsize))
##    m = np.correlate(x, avgkern, 'same')#[takeslice]
##    print(m.shape)
##    print(x.shape)
##    xm = x - m
##    v = np.correlate(x*x, avgkern, 'same') - m**2
##    v1 = np.correlate(xm*xm, avgkern, valid) #not correct for var of window
###>>> np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')-np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')**2
##    return m[takeslice], v[takeslice], v1

def movmean(x, windowsize=3, lag='lagged'):
    '''moving window mean


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving mean, with same shape as x


    Notes
    -----
    for leading and lagging the data array x is extended by the closest value of the array


    '''
    return movmoment(x, 1, windowsize=windowsize, lag=lag)

def movvar(x, windowsize=3, lag='lagged'):
    '''moving window variance


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        moving variance, with same shape as x


    '''
    m1 = movmoment(x, 1, windowsize=windowsize, lag=lag)
    m2 = movmoment(x, 2, windowsize=windowsize, lag=lag)
    return m2 - m1*m1

def movmoment(x, k, windowsize=3, lag='lagged'):
    '''non-central moment


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        k-th moving non-central moment, with same shape as x


    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.

    '''

    windsize = windowsize
    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        #lead = -0 + windsize #windsize//2
        lead = -0# + (windsize-1) + windsize//2
        sl = slice((windsize-1) or None, -2*(windsize-1) or None)
    elif lag == 'centered':
        lead = -windsize//2  #0#-1 #+ #(windsize-1)
        sl = slice((windsize-1)+windsize//2 or None, -(windsize-1)-windsize//2 or None)
    elif lag == 'leading':
        #lead = -windsize +1#+1 #+ (windsize-1)#//2 +1
        lead = -windsize +2 #-windsize//2 +1
        sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    else:
        raise ValueError

    avgkern = (np.ones(windowsize)/float(windowsize))
    xext = expandarr(x, windsize-1)
    #Note: expandarr increases the array size by 2*(windsize-1)

    #sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    print(sl)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, 'full')[sl]
        #return np.correlate(xext**k, avgkern, 'same')[windsize-lead:-(windsize+lead)]
    else:
        print(xext.shape)
        print(avgkern[:,None].shape)

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:,None], 'full')[sl,:]







#x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,[1],'full')
#x=0.5**np.arange(3);np.correlate(x,x,'same')
##>>> x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')
##
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> xo
##xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> x=np.ones(10);xo=x-x.mean();a=np.correlate(xo,xo,'full')
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> d
##array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,   9.,
##         8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])


##def ccovf():
##    pass
##    #x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')

__all__ = ['movorder', 'movmean', 'movvar', 'movmoment']

if __name__ == '__main__':

    print('\ncheckin moving mean and variance')
    nobs = 10
    x = np.arange(nobs)
    ws = 3
    ave = np.array([ 0., 1/3., 1., 2., 3., 4., 5., 6., 7., 8.,
                  26/3., 9])
    va = np.array([[ 0.        ,  0.        ],
                   [ 0.22222222,  0.88888889],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.66666667,  2.66666667],
                   [ 0.22222222,  0.88888889],
                   [ 0.        ,  0.        ]])
    ave2d = np.c_[ave, 2*ave]
    print(movmean(x, windowsize=ws, lag='lagged'))
    print(movvar(x, windowsize=ws, lag='lagged'))
    print([np.var(x[i-ws:i]) for i in range(ws, nobs)])
    m1 = movmoment(x, 1, windowsize=3, lag='lagged')
    m2 = movmoment(x, 2, windowsize=3, lag='lagged')
    print(m1)
    print(m2)
    print(m2 - m1*m1)

    # this implicitly also tests moment
    assert_array_almost_equal(va[ws-1:,0],
                    movvar(x, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws//2:-ws//2+1,0],
                    movvar(x, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws+1,0],
                    movvar(x, windowsize=ws, lag='lagged'))



    print('\nchecking moving moment for 2d (columns only)')
    x2d = np.c_[x, 2*x]
    print(movmoment(x2d, 1, windowsize=3, lag='centered'))
    print(movmean(x2d, windowsize=ws, lag='lagged'))
    print(movvar(x2d, windowsize=ws, lag='lagged'))
    assert_array_almost_equal(va[ws-1:,:],
                    movvar(x2d, windowsize=3, lag='leading'))
    assert_array_almost_equal(va[ws//2:-ws//2+1,:],
                    movvar(x2d, windowsize=3, lag='centered'))
    assert_array_almost_equal(va[:-ws+1,:],
                    movvar(x2d, windowsize=ws, lag='lagged'))

    assert_array_almost_equal(ave2d[ws-1:],
                    movmoment(x2d, 1, windowsize=3, lag='leading'))
    assert_array_almost_equal(ave2d[ws//2:-ws//2+1],
                    movmoment(x2d, 1, windowsize=3, lag='centered'))
    assert_array_almost_equal(ave2d[:-ws+1],
                    movmean(x2d, windowsize=ws, lag='lagged'))

    from scipy import ndimage
    print(ndimage.filters.correlate1d(x2d, np.array([1,1,1])/3., axis=0))


    #regression test check

    xg = np.array([  0. ,   0.1,   0.3,   0.6,   1. ,   1.5,   2.1,   2.8,   3.6,
                 4.5,   5.5,   6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,
                13.5,  14.5,  15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,
                22.5,  23.5,  24.5,  25.5,  26.5,  27.5,  28.5,  29.5,  30.5,
                31.5,  32.5,  33.5,  34.5,  35.5,  36.5,  37.5,  38.5,  39.5,
                40.5,  41.5,  42.5,  43.5,  44.5,  45.5,  46.5,  47.5,  48.5,
                49.5,  50.5,  51.5,  52.5,  53.5,  54.5,  55.5,  56.5,  57.5,
                58.5,  59.5,  60.5,  61.5,  62.5,  63.5,  64.5,  65.5,  66.5,
                67.5,  68.5,  69.5,  70.5,  71.5,  72.5,  73.5,  74.5,  75.5,
                76.5,  77.5,  78.5,  79.5,  80.5,  81.5,  82.5,  83.5,  84.5,
                85.5,  86.5,  87.5,  88.5,  89.5,  90.5,  91.5,  92.5,  93.5,
                94.5])

    assert_array_almost_equal(xg, movmean(np.arange(100), 10,'lagged'))

    xd = np.array([  0.3,   0.6,   1. ,   1.5,   2.1,   2.8,   3.6,   4.5,   5.5,
                 6.5,   7.5,   8.5,   9.5,  10.5,  11.5,  12.5,  13.5,  14.5,
                15.5,  16.5,  17.5,  18.5,  19.5,  20.5,  21.5,  22.5,  23.5,
                24.5,  25.5,  26.5,  27.5,  28.5,  29.5,  30.5,  31.5,  32.5,
                33.5,  34.5,  35.5,  36.5,  37.5,  38.5,  39.5,  40.5,  41.5,
                42.5,  43.5,  44.5,  45.5,  46.5,  47.5,  48.5,  49.5,  50.5,
                51.5,  52.5,  53.5,  54.5,  55.5,  56.5,  57.5,  58.5,  59.5,
                60.5,  61.5,  62.5,  63.5,  64.5,  65.5,  66.5,  67.5,  68.5,
                69.5,  70.5,  71.5,  72.5,  73.5,  74.5,  75.5,  76.5,  77.5,
                78.5,  79.5,  80.5,  81.5,  82.5,  83.5,  84.5,  85.5,  86.5,
                87.5,  88.5,  89.5,  90.5,  91.5,  92.5,  93.5,  94.5,  95.4,
                96.2,  96.9,  97.5,  98. ,  98.4,  98.7,  98.9,  99. ])
    assert_array_almost_equal(xd, movmean(np.arange(100), 10,'leading'))

    xc = np.array([ 1.36363636,   1.90909091,   2.54545455,   3.27272727,
                 4.09090909,   5.        ,   6.        ,   7.        ,
                 8.        ,   9.        ,  10.        ,  11.        ,
                12.        ,  13.        ,  14.        ,  15.        ,
                16.        ,  17.        ,  18.        ,  19.        ,
                20.        ,  21.        ,  22.        ,  23.        ,
                24.        ,  25.        ,  26.        ,  27.        ,
                28.        ,  29.        ,  30.        ,  31.        ,
                32.        ,  33.        ,  34.        ,  35.        ,
                36.        ,  37.        ,  38.        ,  39.        ,
                40.        ,  41.        ,  42.        ,  43.        ,
                44.        ,  45.        ,  46.        ,  47.        ,
                48.        ,  49.        ,  50.        ,  51.        ,
                52.        ,  53.        ,  54.        ,  55.        ,
                56.        ,  57.        ,  58.        ,  59.        ,
                60.        ,  61.        ,  62.        ,  63.        ,
                64.        ,  65.        ,  66.        ,  67.        ,
                68.        ,  69.        ,  70.        ,  71.        ,
                72.        ,  73.        ,  74.        ,  75.        ,
                76.        ,  77.        ,  78.        ,  79.        ,
                80.        ,  81.        ,  82.        ,  83.        ,
                84.        ,  85.        ,  86.        ,  87.        ,
                88.        ,  89.        ,  90.        ,  91.        ,
                92.        ,  93.        ,  94.        ,  94.90909091,
                95.72727273,  96.45454545,  97.09090909,  97.63636364])
    assert_array_almost_equal(xc, movmean(np.arange(100), 11,'centered'))
