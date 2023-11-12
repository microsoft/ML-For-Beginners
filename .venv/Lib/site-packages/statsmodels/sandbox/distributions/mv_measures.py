'''using multivariate dependence and divergence measures

The standard correlation coefficient measures only linear dependence between
random variables.
kendall's tau measures any monotonic relationship also non-linear.

mutual information measures any kind of dependence, but does not distinguish
between positive and negative relationship


mutualinfo_kde and mutualinfo_binning follow Khan et al. 2007

Shiraj Khan, Sharba Bandyopadhyay, Auroop R. Ganguly, Sunil Saigal,
David J. Erickson, III, Vladimir Protopopescu, and George Ostrouchov,
Relative performance of mutual information estimation methods for
quantifying the dependence among short and noisy data,
Phys. Rev. E 76, 026209 (2007)
http://pre.aps.org/abstract/PRE/v76/i2/e026209


'''

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

import statsmodels.sandbox.infotheo as infotheo


def mutualinfo_kde(y, x, normed=True):
    '''mutual information of two random variables estimated with kde

    '''
    nobs = len(x)
    if not len(y) == nobs:
        raise ValueError('both data arrays need to have the same size')
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yx = np.vstack((y,x))
    kde_x = gaussian_kde(x)(x)
    kde_y = gaussian_kde(y)(y)
    kde_yx = gaussian_kde(yx)(yx)

    mi_obs = np.log(kde_yx) - np.log(kde_x) - np.log(kde_y)
    mi = mi_obs.sum() / nobs
    if normed:
        mi_normed = np.sqrt(1. - np.exp(-2 * mi))
        return mi_normed
    else:
        return mi

def mutualinfo_kde_2sample(y, x, normed=True):
    '''mutual information of two random variables estimated with kde

    '''
    nobs = len(x)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    #yx = np.vstack((y,x))
    kde_x = gaussian_kde(x.T)(x.T)
    kde_y = gaussian_kde(y.T)(x.T)
    #kde_yx = gaussian_kde(yx)(yx)

    mi_obs = np.log(kde_x) - np.log(kde_y)
    if len(mi_obs) != nobs:
        raise ValueError("Wrong number of observations")
    mi = mi_obs.mean()
    if normed:
        mi_normed = np.sqrt(1. - np.exp(-2 * mi))
        return mi_normed
    else:
        return mi

def mutualinfo_binned(y, x, bins, normed=True):
    '''mutual information of two random variables estimated with kde



    Notes
    -----
    bins='auto' selects the number of bins so that approximately 5 observations
    are expected to be in each bin under the assumption of independence. This
    follows roughly the description in Kahn et al. 2007

    '''
    nobs = len(x)
    if not len(y) == nobs:
        raise ValueError('both data arrays need to have the same size')
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    #yx = np.vstack((y,x))


##    fyx, binsy, binsx = np.histogram2d(y, x, bins=bins)
##    fx, binsx_ = np.histogram(x, bins=binsx)
##    fy, binsy_ = np.histogram(y, bins=binsy)

    if bins == 'auto':
        ys = np.sort(y)
        xs = np.sort(x)
        #quantiles = np.array([0,0.25, 0.4, 0.6, 0.75, 1])
        qbin_sqr = np.sqrt(5./nobs)
        quantiles = np.linspace(0, 1, 1./qbin_sqr)
        quantile_index = ((nobs-1)*quantiles).astype(int)
        #move edges so that they do not coincide with an observation
        shift = 1e-6 + np.ones(quantiles.shape)
        shift[0] -= 2*1e-6
        binsy = ys[quantile_index] + shift
        binsx = xs[quantile_index] + shift

    elif np.size(bins) == 1:
        binsy = bins
        binsx = bins
    elif (len(bins) == 2):
        binsy, binsx = bins
##        if np.size(bins[0]) == 1:
##            binsx = bins[0]
##        if np.size(bins[1]) == 1:
##            binsx = bins[1]

    fx, binsx = np.histogram(x, bins=binsx)
    fy, binsy = np.histogram(y, bins=binsy)
    fyx, binsy, binsx = np.histogram2d(y, x, bins=(binsy, binsx))

    pyx = fyx * 1. / nobs
    px = fx * 1. / nobs
    py = fy * 1. / nobs


    mi_obs = pyx * (np.log(pyx+1e-10) - np.log(py)[:,None] - np.log(px))
    mi = mi_obs.sum()

    if normed:
        mi_normed = np.sqrt(1. - np.exp(-2 * mi))
        return mi_normed, (pyx, py, px, binsy, binsx), mi_obs
    else:
        return mi


if __name__ == '__main__':
    import statsmodels.api as sm

    funtype = ['linear', 'quadratic'][1]
    nobs = 200
    sig = 2#5.
    #x = np.linspace(-3, 3, nobs) + np.random.randn(nobs)
    x = np.sort(3*np.random.randn(nobs))
    exog = sm.add_constant(x, prepend=True)
    #y = 0 + np.log(1+x**2) + sig * np.random.randn(nobs)
    if funtype == 'quadratic':
        y = 0 + x**2 + sig * np.random.randn(nobs)
    if funtype == 'linear':
        y = 0 + x + sig * np.random.randn(nobs)

    print('correlation')
    print(np.corrcoef(y,x)[0, 1])
    print('pearsonr', stats.pearsonr(y,x))
    print('spearmanr', stats.spearmanr(y,x))
    print('kendalltau', stats.kendalltau(y,x))

    pxy, binsx, binsy = np.histogram2d(x,y, bins=5)
    px, binsx_ = np.histogram(x, bins=binsx)
    py, binsy_ = np.histogram(y, bins=binsy)
    print('mutualinfo', infotheo.mutualinfo(px*1./nobs, py*1./nobs,
                                            1e-15+pxy*1./nobs, logbase=np.e))

    print('mutualinfo_kde normed', mutualinfo_kde(y,x))
    print('mutualinfo_kde       ', mutualinfo_kde(y,x, normed=False))
    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = \
               mutualinfo_binned(y, x, 5, normed=True)
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())

    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = \
               mutualinfo_binned(y, x, 'auto', normed=True)
    print('auto')
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())

    ys = np.sort(y)
    xs = np.sort(x)
    by = ys[((nobs-1)*np.array([0, 0.25, 0.4, 0.6, 0.75, 1])).astype(int)]
    bx = xs[((nobs-1)*np.array([0, 0.25, 0.4, 0.6, 0.75, 1])).astype(int)]
    mi_normed, (pyx2, py2, px2, binsy2, binsx2), mi_obs = \
               mutualinfo_binned(y, x, (by,bx), normed=True)
    print('quantiles')
    print('mutualinfo_binned normed', mi_normed)
    print('mutualinfo_binned       ', mi_obs.sum())

    doplot = 1#False
    if doplot:
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'o')
        olsres = sm.OLS(y, exog).fit()
        plt.plot(x, olsres.fittedvalues)
