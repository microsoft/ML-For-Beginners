# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:20:45 2011

@author: josef
"""
from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.sandbox.tools.mctools import StatTestMC
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

def normalnoisesim(nobs=500, loc=0.0):
    return (loc+np.random.randn(nobs))


def lb(x):
    s,p = acorr_ljungbox(x, lags=4)
    return np.r_[s, p]


mc1 = StatTestMC(normalnoisesim, lb)
mc1.run(5000, statindices=lrange(4))

print(mc1.summary_quantiles([1,2,3], stats.chi2([2,3,4]).ppf,
                            varnames=['lag 1', 'lag 2', 'lag 3'],
                            title='acorr_ljungbox'))
print('\n\n')

frac = [0.01, 0.025, 0.05, 0.1, 0.975]
crit = stats.chi2([2,3,4]).ppf(np.atleast_2d(frac).T)
print(mc1.summary_cdf([1,2,3], frac, crit,
                      varnames=['lag 1', 'lag 2', 'lag 3'],
                      title='acorr_ljungbox'))
print(mc1.cdf(crit, [1,2,3])[1])

#----------------------

def randwalksim(nobs=500, drift=0.0):
    return (drift+np.random.randn(nobs)).cumsum()


def adf20(x):
    return adfuller(x, 2, regression="n", autolag=None)

print(adf20(np.random.randn(100)))

mc2 = StatTestMC(randwalksim, adf20)
mc2.run(10000, statindices=[0,1])
frac = [0.01, 0.05, 0.1]
#bug
crit = np.array([-3.4996365338407074, -2.8918307730370025, -2.5829283377617176])[:,None]
print(mc2.summary_cdf([0], frac, crit,
                      varnames=['adf'],
                      title='adf'))
#bug
#crit2 = np.column_stack((crit, frac))
#print mc2.summary_cdf([0, 1], frac, crit,
#                      varnames=['adf'],
#                      title='adf')

print(mc2.quantiles([0]))
print(mc2.cdf(crit, [0]))

doplot=1
if doplot:
    import matplotlib.pyplot as plt
    mc1.plot_hist([3],stats.chi2([4]).pdf)
    plt.title('acorr_ljungbox - MC versus chi2')
    plt.show()
